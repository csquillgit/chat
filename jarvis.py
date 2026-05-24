#!/usr/bin/env python3
"""Voice assistant that wakes on "Hey Max" and sends prompts to a local LLM."""

from __future__ import annotations

import argparse
import json
import queue
import shutil
import signal
import subprocess
import sys
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


WAKE_WORDS = "hey max"


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


class JarvisAssistant:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.audio_queue: queue.Queue[bytes] = queue.Queue()
        self.model = Model(str(resolve_vosk_model_path(config.vosk_model_path)))
        self.recognizer = KaldiRecognizer(self.model, config.sample_rate)
        self.speaker = TextSpeaker()
        self.running = True

    def audio_callback(self, indata: bytes, frames: int, time, status) -> None:  # noqa: ANN001
        if status:
            print(f"Audio warning: {status}", file=sys.stderr)
        self.audio_queue.put(bytes(indata))

    def listen(self) -> Iterable[str]:
        with sd.RawInputStream(
            samplerate=self.config.sample_rate,
            blocksize=8000,
            dtype="int16",
            channels=1,
            callback=self.audio_callback,
            device=self.config.device,
        ):
            print('Listening. Say "Hey Max" to start.')
            while self.running:
                data = self.audio_queue.get()
                if not self.running:
                    break
                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    text = normalize(result.get("text", ""))
                    if text:
                        yield text

    def run(self) -> None:
        waiting_for_prompt = False

        for text in self.listen():
            print(f"Heard: {text}")

            if not waiting_for_prompt:
                if WAKE_WORDS in text:
                    prompt = remove_wake_words(text)
                    if prompt:
                        self.respond_to(prompt)
                    else:
                        waiting_for_prompt = True
                        self.say("Yes?")
                    continue
                continue

            prompt = remove_wake_words(text)
            if not prompt:
                self.say("I'm listening.")
                continue

            waiting_for_prompt = False
            self.respond_to(prompt)

    def respond_to(self, prompt: str) -> None:
        try:
            response = self.ask_llm(prompt)
        except requests.RequestException as exc:
            print(f"LLM request failed: {exc}", file=sys.stderr)
            self.say("I could not reach the local language model.")
            return

        print(f"Jarvis: {response}")
        self.say(response)

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
        self.speaker.say(text)

    def stop(self) -> None:
        self.running = False
        self.audio_queue.put(b"")


def normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def remove_wake_words(text: str) -> str:
    return normalize(text.replace(WAKE_WORDS, "", 1))


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
        description='Listen for "Hey Max", send speech to a local LLM, and speak the response.'
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
    )


def main() -> int:
    assistant = JarvisAssistant(parse_args())

    def handle_shutdown(signum, frame) -> None:  # noqa: ANN001
        assistant.stop()

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    try:
        assistant.run()
    except KeyboardInterrupt:
        pass
    finally:
        assistant.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
