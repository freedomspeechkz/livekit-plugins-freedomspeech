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

"""Freedom Speech speech-to-text for LiveKit Agents.

Streaming: ``WS /ws/asr`` takes raw 16 kHz mono PCM16 and returns ``partial`` and
``final`` messages. Non-streaming: ``POST /v1/audio/transcriptions``.
API reference: https://freedomspeech.kz/docs
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import time
import weakref
from dataclasses import dataclass
from urllib.parse import urlencode

import aiohttp

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    LanguageCode,
    stt,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given

from .log import logger

BASE_URL = "https://api.freedomspeech.kz"
SAMPLE_RATE = 16000  # the server accepts only 16 kHz mono PCM16 and does not resample
NUM_CHANNELS = 1
# the server drops a socket after 15 s without audio; silence is sent before that
KEEPALIVE_IDLE_S = 8.0
KEEPALIVE_SILENCE_MS = 100
USAGE_REPORT_S = 5.0
# after the input ends, how long to wait for the last final transcript before closing
FINAL_AFTER_STOP_S = 5.0
USER_AGENT = "livekit-plugins-freedomspeech"


@dataclass
class STTOptions:
    language: LanguageCode | None
    prettify: bool
    sample_rate: int
    num_channels: int
    base_url: str


def _env_api_key() -> str | None:
    return os.environ.get("FREEDOMSPEECH_API_KEY") or os.environ.get("FS_API_KEY")


def _ws_base(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.startswith("https://"):
        return "wss://" + base[len("https://") :]
    if base.startswith("http://"):
        return "ws://" + base[len("http://") :]
    return base


class STT(stt.STT):
    def __init__(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        prettify: bool = True,
        sample_rate: int = SAMPLE_RATE,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: str = BASE_URL,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Create a new instance of Freedom Speech STT.

        Args:
            language: Language hint, ``kk``, ``ru`` or ``en``. The server detects the
                language on its own; the hint only labels events until the server reports
                the detected language with a final transcript. Defaults to auto-detect.
            prettify: Whether the server formats named entities, numbers and dates.
                ``False`` returns the raw recognizer output. Defaults to True.
            sample_rate: Sample rate of the audio sent to the server. The server accepts
                16000 only; other input rates are resampled by the stream.
            api_key: Freedom Speech API key. If not provided, ``FREEDOMSPEECH_API_KEY`` or
                ``FS_API_KEY`` from the environment is used.
            base_url: API base URL. Defaults to ``https://api.freedomspeech.kz``.
            http_session: Optional aiohttp ClientSession to use for requests.

        Raises:
            ValueError: If no API key is provided or found in environment variables.
        """
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                diarization=False,
                aligned_transcript=False,
            )
        )
        key = api_key if is_given(api_key) else _env_api_key()
        if not key:
            raise ValueError(
                "Freedom Speech API key is required, either as argument or set"
                " FREEDOMSPEECH_API_KEY environment variable"
            )
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"Freedom Speech streaming accepts only {SAMPLE_RATE} Hz audio")
        self._api_key = key
        self._opts = STTOptions(
            language=LanguageCode(language) if is_given(language) and language else None,
            prettify=prettify,
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
            base_url=base_url.rstrip("/"),
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SpeechStream]()

    @property
    def model(self) -> str:
        return "freedomspeech"

    @property
    def provider(self) -> str:
        return "Freedom Speech"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        config = self._sanitize_options(language=language)
        form = aiohttp.FormData()
        form.add_field("prettify", "true" if config.prettify else "false")
        if config.language:
            form.add_field("language", str(config.language))
        form.add_field(
            "file",
            rtc.combine_audio_frames(buffer).to_wav_bytes(),
            filename="audio.wav",
            content_type="audio/wav",
        )
        try:
            async with self._ensure_session().post(
                url=f"{config.base_url}/v1/audio/transcriptions",
                data=form,
                headers={"X-API-Key": self._api_key, "User-Agent": USER_AGENT},
                timeout=aiohttp.ClientTimeout(total=120, sock_connect=conn_options.timeout),
            ) as res:
                res.raise_for_status()
                data = await res.json()
        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except Exception as e:
            raise APIConnectionError() from e

        text = str(data.get("text", ""))
        lang = data.get("language") or (str(config.language) if config.language else "")
        duration = data.get("duration")
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            request_id=utils.shortuuid(),
            alternatives=[
                stt.SpeechData(
                    language=LanguageCode(lang),
                    text=text,
                    start_time=0.0,
                    end_time=float(duration) if isinstance(duration, (int, float)) else 0.0,
                )
            ],
        )

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        config = self._sanitize_options(language=language)
        stream = SpeechStream(
            stt=self,
            conn_options=conn_options,
            opts=config,
            api_key=self._api_key,
            http_session=self._ensure_session(),
        )
        self._streams.add(stream)
        return stream

    def update_options(
        self,
        *,
        language: NotGivenOr[str | None] = NOT_GIVEN,
        prettify: NotGivenOr[bool] = NOT_GIVEN,
    ) -> None:
        if is_given(language):
            self._opts.language = LanguageCode(language) if language else None
        if is_given(prettify):
            self._opts.prettify = prettify
        for stream in self._streams:
            stream.update_options(language=language, prettify=prettify)

    def _sanitize_options(self, *, language: NotGivenOr[str] = NOT_GIVEN) -> STTOptions:
        config = dataclasses.replace(self._opts)
        if is_given(language) and language:
            config.language = LanguageCode(language)
        return config


class SpeechStream(stt.SpeechStream):
    _STOP_MSG: str = json.dumps({"type": "stop"})

    def __init__(
        self,
        *,
        stt: STT,
        opts: STTOptions,
        conn_options: APIConnectOptions,
        api_key: str,
        http_session: aiohttp.ClientSession,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)
        self._opts = opts
        self._api_key = api_key
        self._session = http_session
        self._speaking = False
        self._request_id = utils.shortuuid()
        self._reconnect_event = asyncio.Event()
        self._final_after_stop = asyncio.Event()
        self._usage_pending = 0.0
        self._usage_last_report = time.monotonic()
        self._last_audio_sent = time.monotonic()

    def update_options(
        self,
        *,
        language: NotGivenOr[str | None] = NOT_GIVEN,
        prettify: NotGivenOr[bool] = NOT_GIVEN,
    ) -> None:
        if is_given(language):
            self._opts.language = LanguageCode(language) if language else None
        if is_given(prettify) and prettify != self._opts.prettify:
            self._opts.prettify = prettify
            # prettify lives in the socket query, so the connection is rebuilt
            self._reconnect_event.set()

    def _ws_url(self) -> str:
        query: dict[str, str] = {"apiKey": self._api_key}
        if not self._opts.prettify:
            query["prettify"] = "false"
        return f"{_ws_base(self._opts.base_url)}/ws/asr?{urlencode(query)}"

    async def _run(self) -> None:
        closing_ws = False

        async def keepalive_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            # the server closes the socket after 15 s without audio; when the input is
            # quiet, short silence keeps it open
            nonlocal closing_ws
            silence = b"\x00" * (self._opts.sample_rate * KEEPALIVE_SILENCE_MS // 1000 * 2)
            try:
                while True:
                    await asyncio.sleep(1.0)
                    if closing_ws or ws.closed:
                        return
                    if time.monotonic() - self._last_audio_sent >= KEEPALIVE_IDLE_S:
                        await ws.send_bytes(silence)
                        self._last_audio_sent = time.monotonic()
            except (aiohttp.ClientError, ConnectionError) as e:
                if closing_ws or self._session.closed:
                    return
                raise APIConnectionError("freedomspeech connection closed unexpectedly") from e
            except Exception as e:
                logger.warning(f"Freedom Speech keepalive task exited: {e}")
                return

        @utils.log_exceptions(logger=logger)
        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws
            # forward audio in chunks of 50 ms
            samples_50ms = self._opts.sample_rate // 20
            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=self._opts.sample_rate,
                num_channels=self._opts.num_channels,
                samples_per_channel=samples_50ms,
            )
            has_ended = False
            try:
                async for data in self._input_ch:
                    frames: list[rtc.AudioFrame] = []
                    if isinstance(data, rtc.AudioFrame):
                        frames.extend(audio_bstream.write(data.data.tobytes()))
                    elif isinstance(data, self._FlushSentinel):
                        frames.extend(audio_bstream.flush())
                        has_ended = True

                    for frame in frames:
                        self._collect_usage(frame.duration)
                        await ws.send_bytes(frame.data.tobytes())
                        self._last_audio_sent = time.monotonic()

                    if has_ended:
                        self._flush_usage()
                        await ws.send_str(SpeechStream._STOP_MSG)
                        has_ended = False

                # input ended: ask for the last final, wait for it briefly, then close the
                # socket ourselves (the server keeps it open until 15 s of silence)
                self._final_after_stop.clear()
                await ws.send_str(SpeechStream._STOP_MSG)
                try:
                    await asyncio.wait_for(self._final_after_stop.wait(), FINAL_AFTER_STOP_S)
                except asyncio.TimeoutError:
                    pass
                closing_ws = True
                await ws.close()
            except (aiohttp.ClientError, ConnectionError) as e:
                if closing_ws or self._session.closed:
                    return
                raise APIConnectionError("freedomspeech connection closed unexpectedly") from e

        @utils.log_exceptions(logger=logger)
        async def recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws
            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if closing_ws or self._session.closed:
                        return
                    raise APIStatusError(
                        message="freedomspeech connection closed unexpectedly",
                        status_code=ws.close_code or -1,
                        body=f"{msg.data=} {msg.extra=}",
                    )
                if msg.type == aiohttp.WSMsgType.ERROR:
                    if closing_ws or self._session.closed:
                        return
                    raise APIConnectionError("freedomspeech connection lost") from ws.exception()
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue
                try:
                    self._process_stream_event(json.loads(msg.data))
                except Exception:
                    logger.exception("failed to process freedomspeech message")

        ws: aiohttp.ClientWebSocketResponse | None = None
        while True:
            try:
                ws = await self._connect_ws()
                tasks = [
                    asyncio.create_task(send_task(ws)),
                    asyncio.create_task(recv_task(ws)),
                    asyncio.create_task(keepalive_task(ws)),
                ]
                tasks_group = asyncio.gather(*tasks)
                wait_reconnect_task = asyncio.create_task(self._reconnect_event.wait())
                try:
                    done, _ = await asyncio.wait(
                        (tasks_group, wait_reconnect_task),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in done:
                        if task != wait_reconnect_task:
                            task.result()
                    if wait_reconnect_task not in done:
                        break
                    self._reconnect_event.clear()
                finally:
                    await utils.aio.gracefully_cancel(*tasks, wait_reconnect_task)
                    tasks_group.cancel()
                    tasks_group.exception()
            finally:
                self._flush_usage()
                if ws is not None:
                    await ws.close()

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        t0 = time.perf_counter()
        try:
            ws = await asyncio.wait_for(
                self._session.ws_connect(
                    self._ws_url(),
                    headers={"User-Agent": USER_AGENT},
                    heartbeat=30.0,
                    max_msg_size=0,
                ),
                self._conn_options.timeout,
            )
        except asyncio.TimeoutError:
            raise APIConnectionError("failed to connect to freedomspeech") from None
        except aiohttp.ClientResponseError as e:
            # the request info carries the URL with the API key; do not chain it
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except Exception as e:
            raise APIConnectionError(
                f"failed to connect to freedomspeech ({type(e).__name__})"
            ) from None

        # the server sends {"type": "ready"} before it accepts audio
        try:
            msg = await asyncio.wait_for(ws.receive(), self._conn_options.timeout)
            ready = json.loads(msg.data) if msg.type == aiohttp.WSMsgType.TEXT else {}
        except Exception as e:
            await ws.close()
            raise APIConnectionError("no ready frame from freedomspeech") from e
        if ready.get("type") != "ready":
            await ws.close()
            raise APIStatusError(
                message=f"unexpected first message from freedomspeech: {ready.get('type')}",
                status_code=-1,
                request_id=None,
                body=ready.get("message"),
            )

        self._report_connection_acquired(time.perf_counter() - t0, False)
        logger.debug("Established new Freedom Speech STT WebSocket connection")
        return ws

    def _collect_usage(self, duration: float) -> None:
        self._usage_pending += duration
        if time.monotonic() - self._usage_last_report >= USAGE_REPORT_S:
            self._flush_usage()

    def _flush_usage(self) -> None:
        if self._usage_pending <= 0:
            return
        duration, self._usage_pending = self._usage_pending, 0.0
        self._usage_last_report = time.monotonic()
        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.RECOGNITION_USAGE,
                request_id=self._request_id,
                alternatives=[],
                recognition_usage=stt.RecognitionUsage(audio_duration=duration),
            )
        )

    def _speech_data(self, text: str, language: str | None) -> stt.SpeechData:
        lang = language or (str(self._opts.language) if self._opts.language else "")
        return stt.SpeechData(language=LanguageCode(lang), text=text)

    def _process_stream_event(self, data: dict) -> None:
        type_ = data.get("type")
        if type_ == "partial":
            text = data.get("text") or ""
            if not text:
                return
            if not self._speaking:
                self._speaking = True
                self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH))
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                    request_id=self._request_id,
                    alternatives=[self._speech_data(text, None)],
                )
            )
        elif type_ == "final":
            text = data.get("text") or ""
            if text:
                if not self._speaking:
                    self._speaking = True
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
                    )
                self._event_ch.send_nowait(
                    stt.SpeechEvent(
                        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                        request_id=self._request_id,
                        alternatives=[self._speech_data(text, data.get("language"))],
                    )
                )
            if self._speaking:
                self._speaking = False
                self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH))
            self._request_id = utils.shortuuid()
            self._final_after_stop.set()
        elif type_ == "ready":
            pass
        elif type_ == "error":
            raise APIStatusError(
                message=str(data.get("message", "freedomspeech error")),
                status_code=-1,
                request_id=self._request_id,
                body=data,
            )
        else:
            logger.debug("unexpected freedomspeech message type %s", type_)
