#!/usr/bin/env python3
"""Voice assistant that listens continuously and sends prompts to a local LLM."""

from __future__ import annotations

import argparse
import array
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
from typing import Iterable

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
    value: str | float


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

    def audio_callback(self, indata: bytes, frames: int, time, status) -> None:  # noqa: ANN001
        if status:
            self.emit("warning", f"Audio warning: {status}")
        if self.speaking.is_set():
            return
        data = bytes(indata)
        self.emit("audio_level", audio_level(data))
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
            self.emit("status", "Listening")
            while self.running:
                data = self.audio_queue.get()
                if not self.running:
                    break
                if self.speaking.is_set():
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
        self.emit("status", "Listening")

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

    def stop(self) -> None:
        self.running = False
        self.audio_queue.put(b"")

    def discard_pending_audio(self) -> None:
        while True:
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

    def emit(self, kind: str, value: str | float) -> None:
        if self.events is not None:
            self.events.put(AssistantEvent(kind, value))
            return

        if kind == "status" and value == "Listening":
            print("Listening. Speak to ask a question.")
        elif kind == "user_message":
            print(f"Heard: {value}")
        elif kind == "assistant_message":
            print(f"Jarvis: {value}")
        elif kind in {"error", "warning"}:
            print(value, file=sys.stderr)


class JarvisUI:
    def __init__(self, assistant: JarvisAssistant, events: queue.Queue[AssistantEvent]) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.assistant = assistant
        self.events = events
        self.worker: threading.Thread | None = None
        self.closed = False
        self.status = "Starting"
        self.wave_mode = "idle"
        self.audio_strength = 0.0
        self.last_user_audio = 0.0
        self.phase = 0.0

        self.root = tk.Tk()
        self.root.title("Max Voice Assistant")
        self.root.geometry("760x620")
        self.root.minsize(520, 460)
        self.root.configure(bg="#f5f7fb")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        style = ttk.Style()
        style.configure("Shell.TFrame", background="#f5f7fb")
        style.configure("Header.TFrame", background="#111827")
        style.configure("Header.TLabel", background="#111827", foreground="#f9fafb")
        style.configure("Status.TLabel", background="#111827", foreground="#a7f3d0")

        shell = ttk.Frame(self.root, style="Shell.TFrame", padding=18)
        shell.pack(fill="both", expand=True)

        header = ttk.Frame(shell, style="Header.TFrame", padding=(18, 14))
        header.pack(fill="x")

        title = ttk.Label(header, text="Max", style="Header.TLabel", font=("Helvetica", 22, "bold"))
        title.pack(side="left")

        self.status_label = ttk.Label(
            header,
            text=self.status,
            style="Status.TLabel",
            font=("Helvetica", 12, "bold"),
        )
        self.status_label.pack(side="right")

        self.wave = tk.Canvas(
            shell,
            height=110,
            bg="#111827",
            highlightthickness=0,
            bd=0,
        )
        self.wave.pack(fill="x", pady=(0, 14))

        self.transcript_canvas = tk.Canvas(
            shell,
            bg="#f5f7fb",
            highlightthickness=0,
            bd=0,
        )
        self.scrollbar = ttk.Scrollbar(shell, orient="vertical", command=self.transcript_canvas.yview)
        self.messages = ttk.Frame(self.transcript_canvas, style="Shell.TFrame")
        self.messages_window = self.transcript_canvas.create_window(
            (0, 0),
            window=self.messages,
            anchor="nw",
        )
        self.transcript_canvas.configure(yscrollcommand=self.scrollbar.set)
        self.transcript_canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self.messages.bind("<Configure>", self.on_messages_configure)
        self.transcript_canvas.bind("<Configure>", self.on_canvas_configure)

        self.add_message("Jarvis", "Listening. Speak to ask a question.", "assistant")

    def start(self) -> None:
        self.worker = threading.Thread(target=self.run_assistant, daemon=True)
        self.worker.start()
        self.root.after(50, self.poll_events)
        self.root.after(40, self.animate_wave)
        self.root.mainloop()

    def run_assistant(self) -> None:
        try:
            self.assistant.run()
        except Exception as exc:  # noqa: BLE001
            self.events.put(AssistantEvent("error", f"Assistant stopped: {exc}"))

    def poll_events(self) -> None:
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            self.handle_event(event)

        if not self.closed:
            self.root.after(50, self.poll_events)

    def handle_event(self, event: AssistantEvent) -> None:
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
                self.set_status("Listening")
        elif event.kind == "audio_level":
            level = float(event.value)
            self.audio_strength = max(self.audio_strength * 0.6, level)
            if level > 0.02 and self.wave_mode != "assistant":
                self.wave_mode = "user"
                self.last_user_audio = time.monotonic()
                if self.status not in {"Processing", "Speaking", "Error"}:
                    self.set_status("Listening")

    def set_status(self, status: str) -> None:
        self.status = status
        self.status_label.configure(text=status)

    def add_message(self, speaker: str, text: str, role: str) -> None:
        colors = {
            "user": ("#2563eb", "#ffffff", "e"),
            "assistant": ("#ffffff", "#111827", "w"),
            "system": ("#fee2e2", "#991b1b", "w"),
        }
        bg, fg, anchor = colors[role]
        row = self.tk.Frame(self.messages, bg="#f5f7fb")
        row.pack(fill="x", pady=6)

        bubble = self.tk.Frame(row, bg=bg, padx=14, pady=10)
        bubble.pack(anchor=anchor, padx=8)

        name = self.tk.Label(
            bubble,
            text=speaker,
            bg=bg,
            fg=fg,
            font=("Helvetica", 10, "bold"),
            anchor="w",
        )
        name.pack(fill="x")

        body = self.tk.Label(
            bubble,
            text=text,
            bg=bg,
            fg=fg,
            justify="left",
            wraplength=max(280, self.transcript_canvas.winfo_width() - 190),
            font=("Helvetica", 13),
        )
        body.pack(fill="x")
        self.root.after_idle(self.scroll_to_bottom)

    def animate_wave(self) -> None:
        width = max(1, self.wave.winfo_width())
        height = max(1, self.wave.winfo_height())
        self.wave.delete("all")

        if self.wave_mode == "user" and time.monotonic() - self.last_user_audio > 0.7:
            self.wave_mode = "idle"
        if self.wave_mode == "idle":
            self.audio_strength *= 0.82

        self.phase += 0.23
        center = height / 2
        amplitude = self.current_amplitude(height)
        color = "#60a5fa" if self.wave_mode == "user" else "#34d399"
        if self.wave_mode == "idle":
            color = "#475569"

        points: list[float] = []
        step = 8
        for x in range(0, width + step, step):
            progress = x / max(width, 1)
            wave = math.sin((progress * math.pi * 4) + self.phase)
            shimmer = math.sin((progress * math.pi * 11) - (self.phase * 0.7)) * 0.35
            y = center + (wave + shimmer) * amplitude
            points.extend([x, y])

        self.wave.create_line(*points, fill=color, width=4, smooth=True, capstyle="round")
        self.wave.create_text(
            18,
            18,
            anchor="nw",
            text=self.wave_label(),
            fill="#cbd5e1",
            font=("Helvetica", 11, "bold"),
        )

        if not self.closed:
            self.root.after(40, self.animate_wave)

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

    def on_messages_configure(self, event) -> None:  # noqa: ANN001
        self.transcript_canvas.configure(scrollregion=self.transcript_canvas.bbox("all"))

    def on_canvas_configure(self, event) -> None:  # noqa: ANN001
        self.transcript_canvas.itemconfigure(self.messages_window, width=event.width)

    def scroll_to_bottom(self) -> None:
        self.transcript_canvas.configure(scrollregion=self.transcript_canvas.bbox("all"))
        self.transcript_canvas.yview_moveto(1.0)

    def close(self) -> None:
        self.closed = True
        self.assistant.stop()
        self.root.after(100, self.root.destroy)


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
        help="Use the original terminal transcript instead of the Tkinter UI.",
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
            missing_package = exc.name or "tkinter"
            print(f"Missing UI dependency: {missing_package}", file=sys.stderr)
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
