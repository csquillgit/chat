#!/usr/bin/env python3
"""Voice assistant that listens continuously and sends prompts to a local LLM."""

from __future__ import annotations

import argparse
import array
import collections
import json
import math
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import requests
    import sounddevice as sd
    from vosk import KaldiRecognizer, Model
except ModuleNotFoundError as exc:
    missing_package = exc.name or "a required package"
    print(
        f"Missing Python dependency: {missing_package}\n"
        "Install dependencies with:\n\n"
        "  python3 -m pip install -r requirements.txt\n",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc

try:
    import pyttsx3
except ModuleNotFoundError:
    pyttsx3 = None


class TextSpeaker:
    def __init__(self) -> None:
        self.macos_say = shutil.which("say")

    def say(self, text: str) -> None:
        if self.macos_say:
            subprocess.run([self.macos_say, text], check=False)
            return

        if pyttsx3 is None:
            print("No text-to-speech backend available. Install pyttsx3.", file=sys.stderr)
            return

        speaker = pyttsx3.init()
        try:
            speaker.say(text)
            speaker.runAndWait()
        finally:
            speaker.stop()


@dataclass(frozen=True)
class Config:
    llm_provider: str
    llm_url: str
    llm_model: str
    vosk_model_path: str
    sample_rate: int
    device: int | None
    timeout: int
    console: bool


@dataclass(frozen=True)
class AssistantEvent:
    kind: str
    value: Any


class JarvisAssistant:
    def __init__(self, config: Config, events: queue.Queue[AssistantEvent] | None = None) -> None:
        self.config = config
        self.events = events
        self.audio_queue: queue.Queue[bytes] = queue.Queue()
        self.model = Model(str(resolve_vosk_model_path(config.vosk_model_path)))
        self.recognizer = KaldiRecognizer(self.model, config.sample_rate)
        self.speaker = TextSpeaker()
        self.running = True
        self.speaking = threading.Event()
        self.listening_enabled = threading.Event()
        self.listening_enabled.set()

    def audio_callback(self, indata: bytes, frames: int, time, status) -> None:  # noqa: ANN001
        if status:
            self.emit("warning", f"Audio warning: {status}")
        if self.speaking.is_set() or not self.listening_enabled.is_set():
            return
        data = bytes(indata)
        self.emit("audio_level", audio_level(data))
        self.emit("audio_data", bytes(data))
        self.audio_queue.put(data)

    def listen(self) -> Iterable[str]:
        with sd.RawInputStream(
            samplerate=self.config.sample_rate,
            blocksize=8000,
            dtype="int16",
            channels=1,
            callback=self.audio_callback,
            device=self.config.device,
        ):
            self.emit("status", "Listening" if self.listening_enabled.is_set() else "Paused")
            while self.running:
                data = self.audio_queue.get()
                if not self.running:
                    break
                if self.speaking.is_set() or not self.listening_enabled.is_set():
                    continue
                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    text = normalize(result.get("text", ""))
                    if text:
                        yield text

    def run(self) -> None:
        for text in self.listen():
            self.emit("user_message", text)
            self.respond_to(text)

    def respond_to(self, prompt: str) -> None:
        self.emit("status", "Processing")
        try:
            response = self.ask_llm(prompt)
        except requests.RequestException as exc:
            fallback = "I could not reach the local language model."
            self.emit("error", f"LLM request failed: {exc}")
            self.emit("assistant_message", fallback)
            self.say(fallback)
            return

        self.emit("assistant_message", response)
        self.say(response)
        self.emit("status", "Listening" if self.listening_enabled.is_set() else "Paused")

    def ask_llm(self, prompt: str) -> str:
        if self.config.llm_provider == "ollama":
            return self.ask_ollama(prompt)
        if self.config.llm_provider == "lmstudio":
            return self.ask_lmstudio(prompt)
        raise ValueError(f"Unsupported LLM provider: {self.config.llm_provider}")

    def chat_messages(self, prompt: str) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": "You are Jarvis, a concise and helpful voice assistant.",
            },
            {"role": "user", "content": prompt},
        ]

    def ask_ollama(self, prompt: str) -> str:
        endpoint = self.config.llm_url.rstrip("/") + "/api/chat"
        payload = {
            "model": self.config.llm_model,
            "stream": False,
            "messages": self.chat_messages(prompt),
        }
        response = requests.post(endpoint, json=payload, timeout=self.config.timeout)
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "").strip() or "I have no response."

    def ask_lmstudio(self, prompt: str) -> str:
        endpoint = self.config.llm_url.rstrip("/") + "/v1/chat/completions"
        payload = {
            "model": self.config.llm_model,
            "messages": self.chat_messages(prompt),
            "temperature": 0.7,
            "stream": False,
        }
        response = requests.post(endpoint, json=payload, timeout=self.config.timeout)
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "").strip() or "I have no response."
        return "I have no response."

    def say(self, text: str) -> None:
        self.speaking.set()
        self.emit("speaking", "start")
        try:
            self.speaker.say(text)
        finally:
            time.sleep(0.25)
            self.discard_pending_audio()
            self.recognizer.Reset()
            self.speaking.clear()
            self.emit("speaking", "stop")

    def set_listening_enabled(self, enabled: bool) -> None:
        if enabled:
            self.discard_pending_audio()
            self.recognizer.Reset()
            self.listening_enabled.set()
            self.emit("listening_enabled", True)
            self.emit("status", "Listening")
            return

        self.listening_enabled.clear()
        self.discard_pending_audio()
        self.recognizer.Reset()
        self.emit("listening_enabled", False)
        self.emit("status", "Paused")

    def toggle_listening(self) -> bool:
        enabled = not self.listening_enabled.is_set()
        self.set_listening_enabled(enabled)
        return enabled

    def is_listening_enabled(self) -> bool:
        return self.listening_enabled.is_set()

    def stop(self) -> None:
        self.running = False
        self.audio_queue.put(b"")

    def discard_pending_audio(self) -> None:
        while True:
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

    def emit(self, kind: str, value: Any) -> None:
        if self.events is not None:
            self.events.put(AssistantEvent(kind, value))
            return

        if kind == "status" and value == "Listening":
            print("Listening. Speak to ask a question.")
        elif kind == "status" and value == "Paused":
            print("Listening paused.")
        elif kind == "user_message":
            print(f"Heard: {value}")
        elif kind == "assistant_message":
            print(f"Jarvis: {value}")
        elif kind in {"error", "warning"}:
            print(value, file=sys.stderr)


class JarvisUI:
    def __init__(self, assistant: JarvisAssistant, events: queue.Queue[AssistantEvent]) -> None:
        try:
            from PySide6 import QtCore, QtGui, QtWidgets
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "PySide6 is required for the desktop UI. Install dependencies with: "
                "python3 -m pip install -r requirements.txt"
            ) from exc

        self.QtCore = QtCore
        self.QtGui = QtGui
        self.QtWidgets = QtWidgets
        self.assistant = assistant
        self.events = events
        self.worker: threading.Thread | None = None
        self.pcm_buffer: collections.deque[int] = collections.deque(maxlen=24000)
        self.closed = False
        self.status = "Starting"
        self.wave_mode = "idle"
        self.audio_strength = 0.0
        self.last_user_audio = 0.0
        self.phase = 0.0

        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
        self.app.aboutToQuit.connect(self.close)

        class WaveWidget(QtWidgets.QWidget):
            def __init__(wave_self, ui: JarvisUI) -> None:
                super().__init__()
                wave_self.ui = ui
                wave_self.setMinimumHeight(110)
                wave_self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)

            def paintEvent(wave_self, event) -> None:  # noqa: ANN001
                wave_self.ui.paint_wave(wave_self)

        self.window = QtWidgets.QWidget()
        self.window.setWindowTitle("Max Voice Assistant")
        self.window.resize(760, 620)
        self.window.setMinimumSize(520, 460)
        self.window.setStyleSheet(
            """
            QWidget {
                background: #f5f7fb;
                color: #111827;
                font-family: Helvetica, Arial, sans-serif;
            }
            QScrollArea {
                border: 0;
                background: #f5f7fb;
            }
            QScrollArea > QWidget > QWidget {
                background: #f5f7fb;
            }
            QScrollBar:vertical {
                background: #f5f7fb;
                width: 12px;
            }
            QScrollBar::handle:vertical {
                background: #cbd5e1;
                min-height: 28px;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0;
            }
            """
        )

        shell = QtWidgets.QVBoxLayout(self.window)
        shell.setContentsMargins(18, 18, 18, 18)
        shell.setSpacing(14)

        header = QtWidgets.QFrame()
        header.setStyleSheet("background: #111827;")
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(18, 14, 18, 14)

        title = QtWidgets.QLabel("Max")
        title.setStyleSheet("background: #111827; color: #f9fafb; font-size: 22px; font-weight: 700;")
        header_layout.addWidget(title)
        header_layout.addStretch(1)

        self.status_label = QtWidgets.QLabel(self.status)
        self.status_label.setStyleSheet("background: #111827; color: #a7f3d0; font-size: 12px; font-weight: 700;")
        header_layout.addWidget(self.status_label)

        self.listen_button = QtWidgets.QPushButton()
        self.listen_button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.listen_button.setStyleSheet(
            """
            QPushButton {
                background: #f9fafb;
                color: #111827;
                border: 0;
                padding: 7px 12px;
                font-size: 12px;
                font-weight: 700;
            }
            QPushButton:hover {
                background: #e5e7eb;
            }
            """
        )
        self.listen_button.clicked.connect(self.toggle_listening)
        header_layout.addWidget(self.listen_button)
        self.update_listen_button()
        shell.addWidget(header)

        self.wave = WaveWidget(self)
        self.wave.setStyleSheet("background: #111827;")
        shell.addWidget(self.wave)

        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.messages = QtWidgets.QWidget()
        self.messages_layout = QtWidgets.QVBoxLayout(self.messages)
        self.messages_layout.setContentsMargins(0, 0, 0, 0)
        self.messages_layout.setSpacing(12)
        self.messages_layout.addStretch(1)
        self.scroll_area.setWidget(self.messages)
        shell.addWidget(self.scroll_area, 1)

        self.poll_timer = QtCore.QTimer()
        self.poll_timer.timeout.connect(self.poll_events)
        self.wave_timer = QtCore.QTimer()
        self.wave_timer.timeout.connect(self.animate_wave)

        self.add_message("Jarvis", "Listening. Speak to ask a question.", "assistant")

    def start(self) -> None:
        self.worker = threading.Thread(target=self.run_assistant, daemon=True)
        self.worker.start()
        self.poll_timer.start(50)
        self.wave_timer.start(40)
        self.window.show()
        self.app.exec()

    def run_assistant(self) -> None:
        try:
            self.assistant.run()
        except Exception as exc:  # noqa: BLE001
            self.events.put(AssistantEvent("error", f"Assistant stopped: {exc}"))

    def poll_events(self) -> None:
        for _ in range(100):
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            self.handle_event(event)

    def handle_event(self, event: AssistantEvent) -> None:
        if event.kind == "audio_data":
            raw = event.value
            if isinstance(raw, bytes):
                samples = array.array("h")
                samples.frombytes(raw[: len(raw) - (len(raw) % 2)])
                self.pcm_buffer.extend(samples)
            return

        value = str(event.value)
        if event.kind == "status":
            self.set_status(value)
        elif event.kind == "user_message":
            self.add_message("You", value, "user")
            self.set_status("Processing")
        elif event.kind == "assistant_message":
            self.add_message("Jarvis", value, "assistant")
        elif event.kind == "error":
            self.set_status("Error")
            self.add_message("System", value, "system")
        elif event.kind == "warning":
            self.add_message("System", value, "system")
        elif event.kind == "speaking":
            if value == "start":
                self.wave_mode = "assistant"
                self.set_status("Speaking")
            else:
                self.wave_mode = "idle"
                self.audio_strength = 0.0
                self.set_status("Listening" if self.assistant.is_listening_enabled() else "Paused")
        elif event.kind == "audio_level":
            level = float(event.value)
            self.audio_strength = max(self.audio_strength * 0.6, level)
            if level > 0.02 and self.wave_mode != "assistant":
                self.wave_mode = "user"
                self.last_user_audio = time.monotonic()
                if self.status not in {"Processing", "Speaking", "Error"}:
                    self.set_status("Listening")
        elif event.kind == "listening_enabled":
            self.update_listen_button()
            if not bool(event.value):
                self.wave_mode = "idle"
                self.audio_strength = 0.0
                self.wave.update()

    def set_status(self, status: str) -> None:
        self.status = status
        self.status_label.setText(status)
        self.update_listen_button()

    def toggle_listening(self) -> None:
        self.assistant.toggle_listening()

    def update_listen_button(self) -> None:
        if self.assistant.is_listening_enabled():
            self.listen_button.setText("Stop listening")
            self.listen_button.setToolTip("Pause microphone recognition")
        else:
            self.listen_button.setText("Start listening")
            self.listen_button.setToolTip("Resume microphone recognition")

    def add_message(self, speaker: str, text: str, role: str) -> None:
        QtCore = self.QtCore
        QtWidgets = self.QtWidgets
        colors = {
            "user": ("#2563eb", "#ffffff", QtCore.Qt.AlignmentFlag.AlignRight),
            "assistant": ("#ffffff", "#111827", QtCore.Qt.AlignmentFlag.AlignLeft),
            "system": ("#fee2e2", "#991b1b", QtCore.Qt.AlignmentFlag.AlignLeft),
        }
        bg, fg, alignment = colors[role]

        row = QtWidgets.QWidget()
        row_layout = QtWidgets.QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)

        bubble = QtWidgets.QFrame()
        bubble.setMaximumWidth(520)
        bubble.setStyleSheet(f"background: {bg}; border: 1px solid #e5e7eb;")
        bubble_layout = QtWidgets.QVBoxLayout(bubble)
        bubble_layout.setContentsMargins(14, 10, 14, 10)
        bubble_layout.setSpacing(4)

        name = QtWidgets.QLabel(speaker)
        name.setStyleSheet(f"background: {bg}; color: {fg}; font-size: 10px; font-weight: 700;")
        bubble_layout.addWidget(name)

        body = QtWidgets.QLabel(text)
        body.setWordWrap(True)
        body.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        body.setStyleSheet(f"background: {bg}; color: {fg}; font-size: 13px;")
        bubble_layout.addWidget(body)

        if alignment == QtCore.Qt.AlignmentFlag.AlignRight:
            row_layout.addStretch(1)
            row_layout.addWidget(bubble)
        else:
            row_layout.addWidget(bubble)
            row_layout.addStretch(1)

        insert_at = max(0, self.messages_layout.count() - 1)
        self.messages_layout.insertWidget(insert_at, row)
        QtCore.QTimer.singleShot(0, self.scroll_to_bottom)

    def animate_wave(self) -> None:
        if self.wave_mode == "user" and time.monotonic() - self.last_user_audio > 1.0:
            self.wave_mode = "idle"
            self.audio_strength = 0.0
        if self.wave_mode == "idle":
            self.audio_strength *= 0.85
        self.phase += 0.23
        self.wave.update()

    def paint_wave(self, widget) -> None:  # noqa: ANN001
        QtCore = self.QtCore
        QtGui = self.QtGui
        width = max(1, widget.width())
        height = max(1, widget.height())

        painter = QtGui.QPainter(widget)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.fillRect(widget.rect(), QtGui.QColor("#111827"))

        center = height / 2
        amplitude = self.current_amplitude(height)
        color = "#60a5fa" if self.wave_mode == "user" else "#34d399"
        if self.wave_mode == "idle":
            color = "#475569"

        path = QtGui.QPainterPath()
        step = max(4, int(width / 50))
        points: list[tuple[float, float]] = []
        if self.wave_mode == "user" and self.pcm_buffer:
            samples = list(self.pcm_buffer)
            if len(samples) > 1:
                for x in range(0, width + step, step):
                    idx = min(int((x / width) * (len(samples) - 1)), len(samples) - 1)
                    y = center - (samples[idx] / 32768) * amplitude
                    points.append((float(x), y))
        if not points:
            for x in range(0, width + step, step):
                progress = x / max(width, 1)
                wave = math.sin((progress * math.pi * 4) + self.phase)
                shimmer = math.sin((progress * math.pi * 11) - (self.phase * 0.7)) * 0.35
                y = center + (wave * 0.6 + shimmer * 0.3) * amplitude
                points.append((float(x), y))

        if points:
            path.moveTo(points[0][0], points[0][1])
            for x, y in points[1:]:
                path.lineTo(x, y)
            pen = QtGui.QPen(QtGui.QColor(color), 4)
            pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawPath(path)

        painter.setPen(QtGui.QColor("#cbd5e1"))
        painter.setFont(QtGui.QFont("Helvetica", 11, QtGui.QFont.Weight.Bold))
        painter.drawText(18, 26, self.wave_label())
        painter.end()

    def current_amplitude(self, height: int) -> float:
        if self.wave_mode == "assistant":
            pulse = (math.sin(self.phase * 0.9) + 1) / 2
            return height * (0.14 + pulse * 0.18)
        if self.wave_mode == "user":
            return height * min(0.34, 0.08 + self.audio_strength * 0.7)
        return height * max(0.02, min(0.06, self.audio_strength * 0.35))

    def wave_label(self) -> str:
        if self.wave_mode == "assistant":
            return "Jarvis speaking"
        if self.wave_mode == "user":
            return "Listening"
        return self.status

    def scroll_to_bottom(self) -> None:
        scrollbar = self.scroll_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.poll_timer.stop()
        self.wave_timer.stop()
        self.assistant.stop()

def normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def audio_level(data: bytes) -> float:
    if len(data) < 2:
        return 0.0

    samples = array.array("h")
    samples.frombytes(data[: len(data) - (len(data) % 2)])
    if not samples:
        return 0.0

    total = sum(sample * sample for sample in samples)
    rms = math.sqrt(total / len(samples))
    return min(1.0, rms / 32768)


def resolve_vosk_model_path(model_path: str) -> Path:
    path = Path(model_path).expanduser().resolve()
    if is_vosk_model_dir(path):
        return path

    if path.is_dir():
        nested_models = [child for child in path.iterdir() if is_vosk_model_dir(child)]
        if len(nested_models) == 1:
            return nested_models[0]

        children = ", ".join(sorted(child.name for child in path.iterdir())[:8])
        detail = f" Found: {children}" if children else " The directory is empty."
        raise SystemExit(
            f"Vosk model files were not found in {path}.{detail}\n"
            "Download and unzip a Vosk model, then pass the directory that contains "
            "'am', 'conf', and 'graph'."
        )

    raise SystemExit(
        f"Vosk model path does not exist: {path}\n"
        "Download and unzip a Vosk model, then pass it with --vosk-model."
    )


def is_vosk_model_dir(path: Path) -> bool:
    return path.is_dir() and all((path / name).exists() for name in ("am", "conf", "graph"))


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Listen continuously, send speech to a local LLM, and speak the response."
    )
    parser.add_argument(
        "--llm-provider",
        choices=("ollama", "lmstudio"),
        default="ollama",
        help="Local LLM backend to use. Default: ollama",
    )
    parser.add_argument(
        "--llm-url",
        required=True,
        help=(
            "Base URL for the local LLM server, for example http://localhost:11434 "
            "for Ollama or http://localhost:1234 for LM Studio."
        ),
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Model name to use. Defaults to llama3.2 for Ollama. Required for LM Studio.",
    )
    parser.add_argument(
        "--ollama-model",
        default=None,
        help="Deprecated alias for --llm-model when using Ollama.",
    )
    parser.add_argument(
        "--vosk-model",
        required=True,
        help="Path to a local Vosk speech-recognition model directory.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Microphone sample rate. Default: 16000",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Optional sounddevice input device index.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="LLM request timeout in seconds. Default: 120",
    )
    parser.add_argument(
        "--console",
        action="store_true",
        help="Use the original terminal transcript instead of the desktop UI.",
    )
    args = parser.parse_args()
    llm_model = args.llm_model or args.ollama_model
    if not llm_model:
        if args.llm_provider == "lmstudio":
            raise SystemExit("--llm-model is required when --llm-provider is lmstudio.")
        llm_model = "llama3.2"

    return Config(
        llm_provider=args.llm_provider,
        llm_url=args.llm_url,
        llm_model=llm_model,
        vosk_model_path=args.vosk_model,
        sample_rate=args.sample_rate,
        device=args.device,
        timeout=args.timeout,
        console=args.console,
    )


def main() -> int:
    config = parse_args()
    events: queue.Queue[AssistantEvent] | None = None if config.console else queue.Queue()
    assistant = JarvisAssistant(config, events)

    def handle_shutdown(signum, frame) -> None:  # noqa: ANN001
        assistant.stop()

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    if config.console:
        try:
            assistant.run()
        except KeyboardInterrupt:
            pass
        finally:
            assistant.stop()
    else:
        if events is None:
            raise RuntimeError("UI mode requires an event queue.")
        try:
            ui = JarvisUI(assistant, events)
        except ModuleNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            assistant.stop()
            return 1

        try:
            ui.start()
        except KeyboardInterrupt:
            assistant.stop()
        finally:
            assistant.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
