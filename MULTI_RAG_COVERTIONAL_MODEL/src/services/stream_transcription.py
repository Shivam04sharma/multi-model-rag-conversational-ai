"""Live audio-stream transcription service.

Two modes:
  StreamTranscriber  — fetches audio from a URL and pipes it to Deepgram (server-side).
  BrowserSTTSession  — accepts raw audio chunks pushed by a browser WebSocket client
                       and returns transcript events via an asyncio Queue.
"""
import asyncio
import queue as stdlib_queue
import threading
from collections.abc import AsyncGenerator
from typing import TypedDict

import httpx
import structlog
from deepgram import DeepgramClient
from deepgram.core.events import EventType

logger = structlog.get_logger(__name__)

_STT_KEYWORDS: list[str] = []


class TranscriptEvent(TypedDict):
    type: str       # always "transcript"
    text: str
    is_final: bool


class StreamTranscriber:
    """Transcribe a live audio stream URL in real time."""

    def __init__(self, api_key: str) -> None:
        self._client = DeepgramClient(api_key=api_key)

    async def transcribe(
        self,
        stream_url: str,
        *,
        language: str = "en",
        model: str = "nova-3",
        smart_format: str = "true",
        interim_results: str = "true",
        endpointing: int = 10,
    ) -> AsyncGenerator[TranscriptEvent, None]:
        """
        Yield :class:`TranscriptEvent` dicts as they arrive from the stream.
        Stops when the HTTP audio feed ends or the caller breaks iteration.
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[TranscriptEvent | None] = asyncio.Queue()
        stop_flag = threading.Event()

        def _run() -> None:
            """Background thread: Deepgram WS + httpx audio feed."""
            try:
                with self._client.listen.v1.connect(
                    model=model,
                    language=language,
                    smart_format=smart_format,
                    interim_results=interim_results,
                    endpointing=endpointing,
                    keywords=_STT_KEYWORDS,
                ) as conn:
                    ready = threading.Event()

                    def on_open(_result) -> None:
                        ready.set()

                    def on_message(result) -> None:
                        channel = getattr(result, "channel", None)
                        if not (channel and hasattr(channel, "alternatives")):
                            return
                        text: str = channel.alternatives[0].transcript
                        is_final: bool = getattr(result, "is_final", True)
                        if text:
                            asyncio.run_coroutine_threadsafe(
                                queue.put(
                                    TranscriptEvent(
                                        type="transcript",
                                        text=text,
                                        is_final=is_final,
                                    )
                                ),
                                loop,
                            )

                    def on_error(error) -> None:
                        logger.error("stream_transcriber_ws_error", error=str(error))

                    conn.on(EventType.OPEN, on_open)
                    conn.on(EventType.MESSAGE, on_message)
                    conn.on(EventType.ERROR, on_error)

                    def _feed_audio() -> None:
                        ready.wait(timeout=10)
                        try:
                            with httpx.stream(
                                "GET", stream_url, follow_redirects=True, timeout=None
                            ) as response:
                                logger.info(
                                    "stream_transcriber_audio_connected",
                                    url=stream_url,
                                    status=response.status_code,
                                )
                                for chunk in response.iter_bytes():
                                    if stop_flag.is_set():
                                        break
                                    conn.send_media(chunk)
                        except Exception as exc:
                            logger.error("stream_transcriber_fetch_error", error=str(exc))

                    audio_thread = threading.Thread(target=_feed_audio, daemon=True)
                    audio_thread.start()
                    conn.start_listening()
                    audio_thread.join(timeout=2)

            except Exception as exc:
                logger.error("stream_transcriber_error", error=str(exc))
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        threading.Thread(target=_run, daemon=True).start()

        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item
        except GeneratorExit:
            pass
        finally:
            stop_flag.set()


# ─── Browser WebSocket STT ─────────────────────────────────────────────────────

class BrowserSTTSession:
    """
    Accepts raw audio bytes pushed by a browser WebSocket client
    and forwards them to Deepgram live transcription.

    Usage:
        session = BrowserSTTSession(api_key, loop)
        session.start()

        # From WebSocket receive loop:
        session.send_audio(chunk_bytes)

        # Consume transcript events:
        async for event in session.events():
            await ws.send_json(event)

        # On disconnect:
        session.stop()
    """

    def __init__(self, api_key: str, loop: asyncio.AbstractEventLoop) -> None:
        self._api_key = api_key
        self._loop = loop
        self._audio_q: stdlib_queue.Queue[bytes | None] = stdlib_queue.Queue()
        self._event_q: asyncio.Queue[TranscriptEvent | None] = asyncio.Queue()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run_deepgram, daemon=True)
        self._thread.start()

    def send_audio(self, chunk: bytes) -> None:
        """Push a raw audio chunk from the browser (webm/opus or any Deepgram-supported format)."""
        self._audio_q.put(chunk)

    def stop(self) -> None:
        self._audio_q.put(None)  # sentinel → stops feeder + deepgram session
        if self._thread:
            self._thread.join(timeout=4)

    async def events(self) -> AsyncGenerator[TranscriptEvent, None]:
        """Async generator that yields transcript events until the session ends."""
        while True:
            item = await self._event_q.get()
            if item is None:
                break
            yield item

    # ── internal ──────────────────────────────────────────────────────────────

    def _run_deepgram(self) -> None:
        client = DeepgramClient(api_key=self._api_key)
        try:
            with client.listen.v1.connect(
                model="nova-3",
                language="multi",
                # deepgram-sdk v6 encodes query params as strings — must pass 'true'/'false'
                smart_format="true",
                interim_results="true",
                endpointing=300,
            ) as conn:

                def on_message(result) -> None:
                    channel = getattr(result, "channel", None)
                    if not (channel and hasattr(channel, "alternatives")):
                        return
                    alts = channel.alternatives
                    if not alts:
                        return
                    text: str = alts[0].transcript
                    # is_final defaults to False so interim results propagate as partial
                    is_final: bool = getattr(result, "is_final", False) or False
                    if text:
                        asyncio.run_coroutine_threadsafe(
                            self._event_q.put(
                                TranscriptEvent(type="transcript", text=text, is_final=is_final)
                            ),
                            self._loop,
                        )

                def on_error(error) -> None:
                    logger.error("browser_stt_ws_error", error=str(error))

                conn.on(EventType.MESSAGE, on_message)
                conn.on(EventType.ERROR, on_error)

                def _feed_audio() -> None:
                    while True:
                        chunk = self._audio_q.get()
                        if chunk is None:
                            # Tell Deepgram we're done → closes the WS → start_listening() returns
                            try:
                                conn.send_close_stream()
                            except Exception:
                                pass
                            break
                        try:
                            conn.send_media(chunk)
                        except Exception as exc:
                            logger.error("browser_stt_send_error", error=str(exc))
                            break

                feeder = threading.Thread(target=_feed_audio, daemon=True)
                feeder.start()
                # start_listening() BLOCKS until Deepgram closes the connection.
                # The feeder calls send_close_stream() on stop() which triggers that close.
                conn.start_listening()

        except Exception as exc:
            logger.error("browser_stt_session_error", error=str(exc))
        finally:
            asyncio.run_coroutine_threadsafe(self._event_q.put(None), self._loop)
