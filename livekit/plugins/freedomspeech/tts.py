# Copyright 2025 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Freedom Speech text-to-speech for LiveKit Agents.

``WS /ws/tts``: one ``synthesize`` request per sentence, audio comes back as base64
PCM16 chunks at 24 kHz followed by ``audio_end``. API reference: https://freedomspeech.kz/docs
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import weakref
from dataclasses import dataclass, replace
from urllib.parse import urlencode

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .log import logger

BASE_URL = "https://api.freedomspeech.kz"
SAMPLE_RATE = 24000
NUM_CHANNELS = 1
USER_AGENT = "livekit-plugins-freedomspeech"
DEFAULT_MODEL = "freedomspeech-flash"
DEFAULT_VOICE = "tomiris"


@dataclass
class _TTSOptions:
    model: str
    voice: str
    language: str | None
    emotion: str | None
    sample_rate: int
    base_url: str
    api_key: str
    sentence_tokenizer: tokenize.SentenceTokenizer


def _env_api_key() -> str | None:
    return os.environ.get("FREEDOMSPEECH_API_KEY") or os.environ.get("FS_API_KEY")


def _ws_base(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.startswith("https://"):
        return "wss://" + base[len("https://") :]
    if base.startswith("http://"):
        return "ws://" + base[len("http://") :]
    return base


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        voice: str = DEFAULT_VOICE,
        language: NotGivenOr[str] = NOT_GIVEN,
        emotion: NotGivenOr[str] = NOT_GIVEN,
        sample_rate: int = SAMPLE_RATE,
        api_key: str | None = None,
        base_url: str = BASE_URL,
        sentence_tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Create a new instance of Freedom Speech TTS.

        Args:
            model: ``freedomspeech-flash`` (default, streams audio as it is generated) or
                ``freedomspeech`` (synthesizes the sentence, then streams it).
            voice: ``tomiris``, ``taymas`` or a voice id from ``GET /api/voices``.
            language: Language hint, ``kk``, ``ru`` or ``en``. Defaults to server detection.
            emotion: Emotion for voices that support it (``neutral``, ``happy``, ``sad``,
                ``surprised``, ``angry``, ``fearful``). ``freedomspeech-flash`` ignores it.
            sample_rate: Output sample rate. The server produces 24000 Hz.
            api_key: Freedom Speech API key. If not provided, ``FREEDOMSPEECH_API_KEY`` or
                ``FS_API_KEY`` from the environment is used.
            base_url: API base URL. Defaults to ``https://api.freedomspeech.kz``.
            sentence_tokenizer: Tokenizer that splits streamed text into requests.
            http_session: Optional aiohttp ClientSession to use for requests.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )
        key = api_key or _env_api_key()
        if not key:
            raise ValueError(
                "Freedom Speech API key required. Set FREEDOMSPEECH_API_KEY or provide api_key."
            )
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"Freedom Speech TTS produces {SAMPLE_RATE} Hz audio only")
        if not is_given(sentence_tokenizer):
            sentence_tokenizer = tokenize.basic.SentenceTokenizer()

        self._opts = _TTSOptions(
            model=model,
            voice=voice,
            language=language if is_given(language) and language else None,
            emotion=emotion if is_given(emotion) and emotion else None,
            sample_rate=sample_rate,
            base_url=base_url.rstrip("/"),
            api_key=key,
            sentence_tokenizer=sentence_tokenizer,
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SynthesizeStream]()
        self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
            max_session_duration=3600,
            mark_refreshed_on_get=False,
        )

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Freedom Speech"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def _ws_url(self) -> str:
        return f"{_ws_base(self._opts.base_url)}/ws/tts?{urlencode({'apiKey': self._opts.api_key})}"

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        session = self._ensure_session()
        try:
            ws = await asyncio.wait_for(
                session.ws_connect(
                    self._ws_url(), headers={"User-Agent": USER_AGENT}, max_msg_size=0
                ),
                timeout,
            )
        except aiohttp.ClientResponseError as e:
            # the request info carries the URL with the API key; do not chain it
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        logger.debug("Established new Freedom Speech TTS WebSocket connection")
        return ws

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        try:
            await ws.send_str(json.dumps({"type": "cancel"}))
        except Exception:
            pass
        finally:
            await ws.close()

    def update_options(
        self,
        *,
        model: NotGivenOr[str] = NOT_GIVEN,
        voice: NotGivenOr[str] = NOT_GIVEN,
        language: NotGivenOr[str | None] = NOT_GIVEN,
        emotion: NotGivenOr[str | None] = NOT_GIVEN,
    ) -> None:
        """Change synthesis parameters; they are sent per request, no reconnect needed."""
        if is_given(model):
            self._opts.model = model
        if is_given(voice):
            self._opts.voice = voice
        if is_given(language):
            self._opts.language = language or None
        if is_given(emotion):
            self._opts.emotion = emotion or None

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    def prewarm(self) -> None:
        self._pool.prewarm()

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()
        await self._pool.aclose()


def _build_request(opts: _TTSOptions, text: str, request_id: str) -> str:
    req: dict = {
        "type": "synthesize",
        "text": text,
        "requestId": request_id,
        "voice": opts.voice,
        "model": opts.model,
    }
    if opts.language:
        req["language"] = opts.language
    if opts.emotion:
        req["emotion"] = opts.emotion
    return json.dumps(req, ensure_ascii=False)


async def _synthesize_one(
    ws: aiohttp.ClientWebSocketResponse,
    opts: _TTSOptions,
    text: str,
    request_id: str,
    output_emitter: tts.AudioEmitter,
    timeout: float,
) -> None:
    """Send one request and push its chunks until ``audio_end`` for that request."""
    await ws.send_str(_build_request(opts, text, request_id))
    while True:
        # a request can wait in the server queue up to 60 s before audio starts
        msg = await ws.receive(timeout=max(timeout, 90.0))
        if msg.type in (
            aiohttp.WSMsgType.CLOSE,
            aiohttp.WSMsgType.CLOSED,
            aiohttp.WSMsgType.CLOSING,
        ):
            raise APIStatusError(
                "freedomspeech websocket connection closed unexpectedly",
                status_code=ws.close_code or -1,
                body=f"{msg.data=} {msg.extra=}",
            )
        if msg.type == aiohttp.WSMsgType.ERROR:
            raise APIConnectionError("freedomspeech connection lost") from ws.exception()
        if msg.type != aiohttp.WSMsgType.TEXT:
            continue
        resp = json.loads(msg.data)
        rid = resp.get("requestId")
        if rid is not None and rid != request_id:
            continue
        mtype = resp.get("type")
        if mtype == "audio_chunk":
            audio = resp.get("audio")
            if audio:
                output_emitter.push(base64.b64decode(audio))
        elif mtype in ("audio_end", "cancelled"):
            return
        elif mtype == "error":
            raise APIError(message="freedomspeech TTS returned error", body=resp)
        else:
            logger.debug("unknown freedomspeech message type %s", mtype)


class ChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
        )
        try:
            async with self._tts._pool.connection(timeout=self._conn_options.timeout) as ws:
                await _synthesize_one(
                    ws,
                    self._opts,
                    self._input_text,
                    request_id,
                    output_emitter,
                    self._conn_options.timeout,
                )
            output_emitter.flush()
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except APIError:
            raise
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=request_id, body=None
            ) from None
        except Exception as e:
            raise APIConnectionError() from e


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        segments_ch = utils.aio.Chan[tokenize.SentenceStream]()
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
            stream=True,
        )

        async def _tokenize_input() -> None:
            # text between flushes is one segment; sentences inside it are separate requests
            sent_stream = None
            async for input in self._input_ch:
                if isinstance(input, str):
                    if sent_stream is None:
                        sent_stream = self._opts.sentence_tokenizer.stream()
                        segments_ch.send_nowait(sent_stream)
                    sent_stream.push_text(input)
                elif isinstance(input, self._FlushSentinel):
                    if sent_stream:
                        sent_stream.end_input()
                    sent_stream = None
            segments_ch.close()

        async def _run_segments() -> None:
            async for sent_stream in segments_ch:
                await self._run_ws(sent_stream, output_emitter)

        tasks = [
            asyncio.create_task(_tokenize_input()),
            asyncio.create_task(_run_segments()),
        ]
        try:
            await asyncio.gather(*tasks)
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except APIError:
            raise
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=request_id, body=None
            ) from None
        except Exception as e:
            raise APIConnectionError() from e
        finally:
            await utils.aio.gracefully_cancel(*tasks)

    async def _run_ws(
        self, sent_stream: tokenize.SentenceStream, output_emitter: tts.AudioEmitter
    ) -> None:
        segment_id = utils.shortuuid()
        output_emitter.start_segment(segment_id=segment_id)
        async with self._tts._pool.connection(timeout=self._conn_options.timeout) as ws:
            self._acquire_time = self._tts._pool.last_acquire_time
            self._connection_reused = self._tts._pool.last_connection_reused
            async for sentence in sent_stream:
                text = sentence.token.strip()
                if not text:
                    continue
                self._mark_started()
                await _synthesize_one(
                    ws,
                    self._opts,
                    text,
                    utils.shortuuid(),
                    output_emitter,
                    self._conn_options.timeout,
                )
        output_emitter.end_segment()
