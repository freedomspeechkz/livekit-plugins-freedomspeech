"""Unit tests on a fake aiohttp websocket; no network. Run: pytest tests"""

from __future__ import annotations

import asyncio
import base64
import json
from unittest.mock import MagicMock

import aiohttp
import pytest

from livekit import rtc
from livekit.agents import APIConnectOptions, stt as lkstt, tts as lktts
from livekit.plugins.freedomspeech import STT, TTS
from livekit.plugins.freedomspeech.stt import _ws_base
from livekit.plugins.freedomspeech.tts import _build_request


class _FakeWS:
    """Scripted websocket: ``script`` maps a sent message predicate to replies."""

    def __init__(self, replies: list[str | bytes | None] | None = None):
        self.sent: list[str | bytes] = []
        self._queue: asyncio.Queue = asyncio.Queue()
        self.close_code = None
        self.closed = False
        for r in replies or []:
            self.reply(r)

    def reply(self, data):
        if data is None:
            self._queue.put_nowait(aiohttp.WSMessage(aiohttp.WSMsgType.CLOSED, None, None))
        elif isinstance(data, bytes):
            self._queue.put_nowait(aiohttp.WSMessage(aiohttp.WSMsgType.BINARY, data, None))
        else:
            self._queue.put_nowait(aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, data, None))

    async def send_str(self, data: str):
        self.sent.append(data)
        self.on_send(data)

    async def send_bytes(self, data: bytes):
        self.sent.append(data)

    def on_send(self, data: str):
        pass

    async def receive(self, timeout=None):
        return await self._queue.get()

    def exception(self):
        return None

    async def close(self):
        self.closed = True
        self.close_code = 1000
        self._queue.put_nowait(aiohttp.WSMessage(aiohttp.WSMsgType.CLOSED, None, None))


class _FakeSession:
    def __init__(self, ws: _FakeWS):
        self.ws = ws
        self.closed = False
        self.connect_calls: list[tuple[str, dict]] = []

    async def ws_connect(self, url, **kwargs):
        self.connect_calls.append((url, kwargs))
        return self.ws


def _frame(ms: int, sr: int = 16000) -> rtc.AudioFrame:
    n = sr * ms // 1000
    return rtc.AudioFrame(data=b"\x01\x00" * n, sample_rate=sr, num_channels=1, samples_per_channel=n)


# ---------------------------------------------------------------- STT


def test_ws_base_url():
    assert _ws_base("https://freedomspeech.kz/") == "wss://freedomspeech.kz"
    assert _ws_base("http://localhost:8000") == "ws://localhost:8000"


def test_stt_requires_api_key(monkeypatch):
    monkeypatch.delenv("FREEDOMSPEECH_API_KEY", raising=False)
    monkeypatch.delenv("FS_API_KEY", raising=False)
    with pytest.raises(ValueError):
        STT()


def test_stt_key_from_env(monkeypatch):
    monkeypatch.setenv("FS_API_KEY", "env-key")
    s = STT(http_session=MagicMock())
    assert s._api_key == "env-key"
    assert s.provider == "Freedom Speech"
    assert s.capabilities.streaming and s.capabilities.interim_results


def test_stt_rejects_other_sample_rate():
    with pytest.raises(ValueError):
        STT(api_key="k", sample_rate=48000, http_session=MagicMock())


@pytest.mark.asyncio
async def test_stt_stream_protocol_and_events():
    ws = _FakeWS([json.dumps({"type": "ready"})])

    def on_send(data):
        if data == json.dumps({"type": "stop"}):
            ws.reply(json.dumps({"type": "final", "text": "сәлеметсіз бе", "language": "kk"}))

    ws.on_send = on_send
    session = _FakeSession(ws)
    s = STT(api_key="k", prettify=False, http_session=session)
    stream = s.stream(conn_options=APIConnectOptions(max_retry=0, timeout=5))

    stream.push_frame(_frame(100))
    stream.push_frame(_frame(100))
    await asyncio.sleep(0.05)
    ws.reply(json.dumps({"type": "partial", "text": "сәлем"}))
    stream.end_input()

    events = []
    async for ev in stream:
        events.append(ev)
    await s.aclose()

    url, kwargs = session.connect_calls[0]
    assert url == "wss://freedomspeech.kz/ws/asr?apiKey=k&prettify=false"
    assert kwargs["headers"]["User-Agent"].startswith("livekit-plugins-freedomspeech")
    audio = [m for m in ws.sent if isinstance(m, bytes)]
    assert audio and all(len(m) == 1600 for m in audio)  # 50 ms chunks at 16 kHz
    assert json.dumps({"type": "stop"}) in ws.sent
    assert ws.closed

    types = [e.type for e in events]
    assert lkstt.SpeechEventType.START_OF_SPEECH in types
    assert lkstt.SpeechEventType.INTERIM_TRANSCRIPT in types
    final = next(e for e in events if e.type == lkstt.SpeechEventType.FINAL_TRANSCRIPT)
    assert final.alternatives[0].text == "сәлеметсіз бе"
    assert final.alternatives[0].language == "kk"
    assert types[-1] == lkstt.SpeechEventType.END_OF_SPEECH or lkstt.SpeechEventType.RECOGNITION_USAGE in types
    usage = [e for e in events if e.type == lkstt.SpeechEventType.RECOGNITION_USAGE]
    assert usage and abs(sum(e.recognition_usage.audio_duration for e in usage) - 0.2) < 0.01


@pytest.mark.asyncio
async def test_stt_stream_fails_without_ready():
    ws = _FakeWS([json.dumps({"type": "error", "message": "unauthorized"})])
    s = STT(api_key="bad", http_session=_FakeSession(ws))
    stream = s.stream(conn_options=APIConnectOptions(max_retry=0, timeout=5))
    stream.end_input()
    with pytest.raises(Exception):
        async for _ in stream:
            pass
    await s.aclose()


def test_stt_update_options_marks_reconnect_only_for_prettify():
    s = STT(api_key="k", http_session=MagicMock())
    s._opts.language = None
    s.update_options(language="ru")
    assert s._opts.language == "ru"
    assert s._sanitize_options(language="en").language == "en"


# ---------------------------------------------------------------- TTS


def test_tts_build_request_fields():
    t = TTS(api_key="k", voice="taymas", model="freedomspeech", language="kk", emotion="happy", http_session=MagicMock())
    req = json.loads(_build_request(t._opts, "Сәлем", "r1"))
    assert req == {
        "type": "synthesize",
        "text": "Сәлем",
        "requestId": "r1",
        "voice": "taymas",
        "model": "freedomspeech",
        "language": "kk",
        "emotion": "happy",
    }
    t.update_options(voice="tomiris", language=None, emotion=None)
    req = json.loads(_build_request(t._opts, "x", "r2"))
    assert req["voice"] == "tomiris" and "language" not in req and "emotion" not in req


def test_tts_defaults_and_validation():
    t = TTS(api_key="k", http_session=MagicMock())
    assert t.model == "freedomspeech-flash" and t.sample_rate == 24000 and t.num_channels == 1
    assert t._ws_url() == "wss://freedomspeech.kz/ws/tts?apiKey=k"
    with pytest.raises(ValueError):
        TTS(api_key="k", sample_rate=16000, http_session=MagicMock())


@pytest.mark.asyncio
async def test_tts_synthesize_collects_chunks():
    pcm = b"\x01\x00" * 2400  # 100 ms at 24 kHz
    ws = _FakeWS()

    def on_send(data):
        req = json.loads(data)
        if req.get("type") == "synthesize":
            rid = req["requestId"]
            ws.reply(json.dumps({"type": "audio_chunk", "requestId": "other", "audio": base64.b64encode(b"\xff\xff").decode()}))
            ws.reply(json.dumps({"type": "audio_chunk", "requestId": rid, "audio": base64.b64encode(pcm).decode(), "sampleRate": 24000}))
            ws.reply(json.dumps({"type": "audio_chunk", "requestId": rid, "audio": base64.b64encode(pcm).decode(), "sampleRate": 24000}))
            ws.reply(json.dumps({"type": "audio_end", "requestId": rid, "totalChunks": 2, "sampleRate": 24000}))

    ws.on_send = on_send
    t = TTS(api_key="k", http_session=_FakeSession(ws))
    frame = await t.synthesize("Сәлем", conn_options=APIConnectOptions(max_retry=0, timeout=5)).collect()
    await t.aclose()

    assert frame.sample_rate == 24000 and frame.num_channels == 1
    # two 100 ms chunks (the foreign request is ignored); the emitter may pad the tail frame
    assert 4800 <= frame.samples_per_channel <= 4800 + 480
    assert frame.data.tobytes()[:4] == b"\x01\x00\x01\x00"
    assert json.loads(ws.sent[0])["text"] == "Сәлем"


@pytest.mark.asyncio
async def test_tts_stream_one_request_per_sentence():
    pcm = b"\x01\x00" * 240
    ws = _FakeWS()
    requests = []

    def on_send(data):
        req = json.loads(data)
        if req.get("type") == "synthesize":
            requests.append(req["text"])
            rid = req["requestId"]
            ws.reply(json.dumps({"type": "audio_chunk", "requestId": rid, "audio": base64.b64encode(pcm).decode()}))
            ws.reply(json.dumps({"type": "audio_end", "requestId": rid}))

    ws.on_send = on_send
    t = TTS(api_key="k", http_session=_FakeSession(ws))
    stream = t.stream(conn_options=APIConnectOptions(max_retry=0, timeout=5))
    # the basic sentence tokenizer merges very short sentences, so these are long enough
    stream.push_text("Бірінші сөйлем осы жерде тұрады. ")
    stream.push_text("Екінші сөйлем де осы жерде тұрады.")
    stream.flush()
    stream.end_input()
    frames = [a async for a in stream]
    await t.aclose()

    assert requests == ["Бірінші сөйлем осы жерде тұрады.", "Екінші сөйлем де осы жерде тұрады."]
    assert frames and frames[-1].is_final
    assert all(isinstance(a, lktts.SynthesizedAudio) for a in frames)


@pytest.mark.asyncio
async def test_tts_error_message_raises():
    ws = _FakeWS()
    ws.on_send = lambda data: ws.reply(json.dumps({"type": "error", "code": "queue_full", "message": "busy"}))
    t = TTS(api_key="k", http_session=_FakeSession(ws))
    with pytest.raises(Exception):
        await t.synthesize("x", conn_options=APIConnectOptions(max_retry=0, timeout=5)).collect()
    await t.aclose()
