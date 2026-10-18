# LiveKit Agents plugin for Freedom Speech

Speech-to-text and text-to-speech for Kazakh, Russian and English with [Freedom Speech](https://freedomspeech.kz) in the [LiveKit Agents](https://github.com/livekit/agents) framework. API reference: <https://freedomspeech.kz/docs>.

Распознавание и синтез казахской, русской и английской речи для LiveKit Agents.

## Installation

```bash
pip install livekit-plugins-freedomspeech
```

Until the package is on PyPI: `pip install -e livekit-plugins-freedomspeech` from this folder.

## Pre-requisites

An API key from <https://freedomspeech.kz>, passed as `api_key` or set as `FREEDOMSPEECH_API_KEY` (or `FS_API_KEY`).

Keys from the console.freedomspeech.kz dashboard work on https://api.freedomspeech.kz, the default `base_url` since 2026-10-18 (Ключ из кабинета console.freedomspeech.kz работает на https://api.freedomspeech.kz, это адрес по умолчанию с 18.10.2026). The plugin uses only `WS /ws/asr` and `WS /ws/tts`, both published on that host.

## Usage

```python
from livekit.agents import AgentSession
from livekit.plugins import freedomspeech

session = AgentSession(
    stt=freedomspeech.STT(),                       # language is detected by the server; language="kk" to hint
    tts=freedomspeech.TTS(voice="tomiris", language="kk"),
    # llm=..., vad=...
)
```

### STT

- Streaming over `WS /ws/asr`: 50 ms chunks of 16 kHz mono PCM16 (the stream resamples other input rates), `partial` becomes `INTERIM_TRANSCRIPT`, `final` becomes `FINAL_TRANSCRIPT` with the language reported by the server (`kk`, `ru`, `en`). A flush sends `{"type": "stop"}` so the server closes the current phrase. The server drops a socket after 15 s without audio, so 100 ms of silence is sent after 8 s of idle input.
- Non-streaming `recognize()` posts a WAV to `POST /v1/audio/transcriptions`.
- `prettify=False` returns raw recognizer output without formatting of numbers, dates and named entities.

### TTS

- `WS /ws/tts`, model `freedomspeech-flash` by default (`freedomspeech` is the slower model with emotions). Audio is 24 kHz mono PCM16.
- `synthesize(text)` is one request; `stream()` splits text into sentences and sends one request per sentence on a pooled connection, so the second and later sentences start without a handshake.
- `update_options(voice=..., model=..., language=..., emotion=...)` applies to the next request. Emotions and prosody tags are not connected for `freedomspeech-flash` yet.

## Layout

```
livekit-plugins-freedomspeech/
  pyproject.toml
  README.md
  livekit/plugins/freedomspeech/{__init__,stt,tts,log,version}.py, py.typed
  tests/test_freedomspeech.py
```

Same layout as the plugins in `livekit/agents/livekit-plugins/`; to contribute upstream, copy the folder there and add it to the workspace in the root `pyproject.toml`.

## Tests

```bash
pip install livekit-agents pytest pytest-asyncio
python -m pytest -o asyncio_mode=auto tests
```

12 tests on a scripted websocket, no network. Checked with livekit-agents 1.8.8, Python 3.13, plus one live check of each path against the API on 18.10.2026 (streaming STT, `recognize`, `synthesize`, streaming TTS).
