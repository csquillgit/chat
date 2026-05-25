# Max Local Voice Assistant

This Python app listens continuously while it is running, sends recognized speech to a local LLM server, and speaks the response back.

It supports Ollama and LM Studio as local LLM backends. On macOS, spoken responses use the built-in `say` command. On other platforms, the app falls back to `pyttsx3`.

## Get The Code

If you do not already have the project on your computer, clone it with Git and move into the project folder:

```bash
git clone https://github.com/csquillgit/chat.git
cd chat
```

## Setup

Create a Python environment and install the app dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Download a Vosk speech-recognition model from https://alphacephei.com/vosk/models and unzip it locally. A small English model is enough to start. For example:

```bash
curl -L -o vosk-model-small-en-us-0.15.zip https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip
```

The model folder passed to `--vosk-model` should contain directories named `am`, `conf`, and `graph`.

## Run With Ollama

In one terminal, pull a model and start Ollama:

```bash
ollama your-loaded-model-id
ollama serve
```

In another terminal, start Max:

```bash
python3 jarvis.py \
  --llm-provider ollama \
  --llm-url http://localhost:11434 \
  --llm-model your-loaded-model-id \
  --vosk-model ./vosk-model-small-en-us-0.15
```

## Run With LM Studio

In LM Studio:

1. Download a chat model.
2. Start the local server from the Developer tab.
3. Copy the model identifier shown by LM Studio.

Then start Max:

```bash
python3 jarvis.py \
  --llm-provider lmstudio \
  --llm-url http://localhost:1234 \
  --llm-model your-loaded-model-id \
  --vosk-model ./vosk-model-small-en-us-0.15
```

Once the app is running, speak your prompt directly.

## Useful Options

List audio devices:

```bash
python3 -m sounddevice
```

Then pass the input device index:

```bash
python3 jarvis.py \
  --llm-provider ollama \
  --llm-url http://localhost:11434 \
  --llm-model llama3.2 \
  --vosk-model ./vosk-model-small-en-us-0.15 \
  --device 1
```

Change the Ollama model:

```bash
python3 jarvis.py \
  --llm-provider ollama \
  --llm-url http://localhost:11434 \
  --llm-model mistral \
  --vosk-model ./vosk-model-small-en-us-0.15
```

For LM Studio, change `--llm-model` to the model identifier shown in LM Studio.

The old `--ollama-model` flag still works for Ollama, but `--llm-model` is preferred.

## Troubleshooting

If Python reports a missing package, install dependencies into the same environment used to run the app:

```bash
python3 -m pip install -r requirements.txt
```

If Vosk says the model folder does not contain model files, make sure `--vosk-model` points to the unzipped folder that contains `am`, `conf`, and `graph`.

If text-to-speech only works once, update to the latest code. The app now uses macOS `say` first, which avoids the common `pyttsx3` event-loop issue on macOS.
