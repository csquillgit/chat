# Max Local Voice Assistant

This Python app listens for the wake words `Hey Max`, records the next spoken phrase, sends it to a local Ollama server, and speaks the response back.

The first version targets Ollama as the local LLM backend. On macOS, spoken responses use the built-in `say` command. On other platforms, the app falls back to `pyttsx3`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Install and run Ollama, then pull a model:

```bash
ollama pull llama3.2
ollama serve
```

Download a Vosk speech-recognition model from https://alphacephei.com/vosk/models and unzip it locally. A small English model is enough to start. For example:

```bash
curl -L -o vosk-model-small-en-us-0.15.zip https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip
```

The model folder passed to `--vosk-model` should contain directories named `am`, `conf`, and `graph`.

## Run

```bash
python3 jarvis.py \
  --llm-url http://localhost:11434 \
  --ollama-model llama3.2 \
  --vosk-model ./vosk-model-small-en-us-0.15
```

Say `Hey Max`, wait for `Yes?`, then ask your question. You can also say the prompt in one phrase, such as `Hey Max what time is it`.

## Useful Options

List audio devices:

```bash
python3 -m sounddevice
```

Then pass the input device index:

```bash
python3 jarvis.py --llm-url http://localhost:11434 --vosk-model ./vosk-model-small-en-us-0.15 --device 1
```

Change the Ollama model:

```bash
python3 jarvis.py \
  --llm-url http://localhost:11434 \
  --ollama-model mistral \
  --vosk-model ./vosk-model-small-en-us-0.15
```

## Troubleshooting

If Python reports a missing package, install dependencies into the same environment used to run the app:

```bash
python3 -m pip install -r requirements.txt
```

If Vosk says the model folder does not contain model files, make sure `--vosk-model` points to the unzipped folder that contains `am`, `conf`, and `graph`.

If text-to-speech only works once, update to the latest code. The app now uses macOS `say` first, which avoids the common `pyttsx3` event-loop issue on macOS.
