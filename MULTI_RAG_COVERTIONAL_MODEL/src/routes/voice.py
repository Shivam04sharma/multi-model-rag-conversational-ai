"""Voice agent routes — WebSocket conversation + REST TTS + live-stream STT endpoints."""
import asyncio
import base64
import json
import os
import pathlib
import re

import httpx
import structlog
from config import settings
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, StreamingResponse
from google.cloud import speech, texttospeech
from google.oauth2 import service_account
from pydantic import BaseModel
from services.stream_transcription import BrowserSTTSession, StreamTranscriber

logger = structlog.get_logger(__name__)

router = APIRouter()

# ─── Config ───────────────────────────────────────────────────────────────────

GOOGLE_CREDENTIALS_PATH = settings.google_application_credentials or os.getenv(
    "GOOGLE_CREDENTIALS_PATH", "vertex_credentials.json"
)
REASONING_ENDPOINT = os.getenv(
    "REASONING_ENDPOINT",
    f"{settings.internal_rag_url}/api/v1/rag/query",
)
REASONING_ENDPOINT_BEARER_TOKEN: str | None = (
    os.getenv("REASONING_ENDPOINT_BEARER_TOKEN") or ""
).strip() or None


def _resolve_google_credentials():
    """
    Resolve Google credentials in priority order:
    1. File path (GOOGLE_CREDENTIALS_PATH or GOOGLE_APPLICATION_CREDENTIALS)
    2. Individual k8s Secret fields (beta/prod — no file mounted)
    """
    # Priority 1: credentials file path
    creds_path = GOOGLE_CREDENTIALS_PATH
    if creds_path and not os.path.isabs(creds_path):
        candidate = pathlib.Path(__file__).parent / creds_path
        if candidate.exists():
            creds_path = str(candidate)
    if creds_path and os.path.exists(creds_path):
        return service_account.Credentials.from_service_account_file(creds_path)

    # Priority 2: individual fields injected via k8s Secret
    private_key = getattr(settings, "gcp_private_key", "")
    if private_key:
        info = {
            "type": "service_account",
            "project_id": getattr(settings, "gcp_project_id", ""),
            "private_key_id": getattr(settings, "gcp_private_key_id", ""),
            "private_key": private_key.replace("\\n", "\n"),
            "client_email": getattr(settings, "gcp_client_email", ""),
            "client_id": getattr(settings, "gcp_client_id", ""),
            "auth_uri": getattr(
                settings, "gcp_auth_uri", "https://accounts.google.com/o/oauth2/auth"
            ),
            "token_uri": getattr(settings, "gcp_token_uri", "https://oauth2.googleapis.com/token"),
        }
        return service_account.Credentials.from_service_account_info(info)

    raise RuntimeError(
        "Google credentials not found. Set GOOGLE_APPLICATION_CREDENTIALS "
        "or inject GCP Secret fields (gcp_private_key, gcp_client_email, etc.)."
    )


# ─── Brand-name pronunciation + recognition tables ────────────────────────────
#
# STT boost phrases:  Google Speech SpeechContext + Deepgram keyterms
#   — tells the model to weight these words higher when decoding audio.
#
# TTS SSML aliases:   <sub alias="..."> replaces the written form with a
#   phonetically correct spoken form that the TTS engine can read naturally.
#   "Oh-nuh-fied" is how en-IN Journey-D renders the correct 3-syllable sound.

_STT_BOOST_PHRASES: list[str] = []

_TTS_PRONUNCIATIONS: list[tuple[str, str]] = []


def _to_ssml(text: str) -> str:
    """
    Wrap brand names in SSML <sub alias> so Google TTS pronounces them correctly.
    Falls back to plain <speak>text</speak> if no brand name appears.
    """
    import html

    # Escape HTML entities first, then apply substitutions
    escaped = html.escape(text)
    for written, spoken in _TTS_PRONUNCIATIONS:
        escaped_written = html.escape(written)
        escaped = escaped.replace(
            escaped_written,
            f'<sub alias="{html.escape(spoken)}">{escaped_written}</sub>',
        )
    return f"<speak>{escaped}</speak>"


# ─── Language detection ────────────────────────────────────────────────────────

_HINGLISH_MARKERS = {
    "kya",
    "hai",
    "hain",
    "kaise",
    "kaisa",
    "kahan",
    "kab",
    "kyun",
    "kyunki",
    "mujhe",
    "mera",
    "meri",
    "mere",
    "aap",
    "tum",
    "main",
    "hum",
    "yeh",
    "woh",
    "bata",
    "batao",
    "chahiye",
    "nahi",
    "nahin",
    "achha",
    "theek",
    "shukriya",
    "dhanyawad",
    "namaste",
    "kaun",
    "kuch",
    "bahut",
    "zyada",
    "hoga",
    "kar",
    "karo",
    "karna",
    "tha",
    "thi",
    "the",
    "kiddan",
    "tussi",
    "tusi",
    "haal",
    "sat",
    "waheguru",
    "alvida",
    "phir",
    "kon",
    "kasa",
}

_MARATHI_MARKERS = {
    "आहे",
    "आहेत",
    "होते",
    "नाही",
    "नको",
    "तुम्ही",
    "आम्ही",
    "सांगा",
    "आणि",
    "मला",
    "माझे",
    "काय",
    "कसे",
    "केव्हा",
    "कोण",
    "कुठे",
    "कारण",
}


def _detect_language(text: str) -> str:
    if any("\u0a00" <= c <= "\u0a7f" for c in text):
        return "punjabi"
    if any("\u0900" <= c <= "\u097f" for c in text):
        return "marathi" if any(w in text for w in _MARATHI_MARKERS) else "hinglish"
    words = set(re.sub(r"[^a-zA-Z\s]", "", text).lower().split())
    return "hinglish" if words & _HINGLISH_MARKERS else "en"


# ─── Behavioural responses ─────────────────────────────────────────────────────

_BEHAVIORAL_RESPONSES: dict[str, dict[str, str]] = {
    "greeting": {
        "en": "Hello! I'm your AI voice assistant. How can I help you today?",
        "hinglish": (
            "Namaste! Main aapka AI voice assistant hun. Batao, main aapki kya madad kar sakta hun?"
        ),
        "punjabi": ("ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ! ਮੈਂ ਤੁਹਾਡਾ AI ਵੌਇਸ ਅਸਿਸਟੈਂਟ ਹਾਂ। ਮੈਂ ਤੁਹਾਡੀ ਕਿਵੇਂ ਮਦਦ ਕਰ ਸਕਦਾ ਹਾਂ?"),
        "marathi": ("नमस्कार! मी तुमचा AI व्हॉइस असिस्टंट आहे. मी तुम्हाला कशी मदत करू शकतो?"),
    },
    "how_are_you": {
        "en": "I'm doing great, thanks for asking! What can I help you with?",
        "hinglish": "Main bilkul theek hun, shukriya! Batao, aapki kya madad kar sakta hun?",
        "punjabi": ("ਮੈਂ ਬਿਲਕੁਲ ਠੀਕ ਹਾਂ, ਧੰਨਵਾਦ! ਦੱਸੋ, ਮੈਂ ਤੁਹਾਡੀ ਕੀ ਮਦਦ ਕਰ ਸਕਦਾ ਹਾਂ?"),
        "marathi": "मी अगदी ठीक आहे, धन्यवाद! सांगा, मी तुम्हाला कशी मदत करू शकतो?",
    },
    "who_are_you": {
        "en": "I'm an AI voice assistant. I can answer questions from the knowledge base.",
        "hinglish": "Main ek AI voice assistant hun. Main knowledge base se sawaalon ke jawaab de sakta hun.",
        "punjabi": "ਮੈਂ ਇੱਕ AI ਵੌਇਸ ਅਸਿਸਟੈਂਟ ਹਾਂ। ਮੈਂ knowledge base ਤੋਂ ਸਵਾਲਾਂ ਦੇ ਜਵਾਬ ਦੇ ਸਕਦਾ ਹਾਂ।",
        "marathi": "मी एक AI व्हॉइस असिस्टंट आहे. मी knowledge base मधून प्रश्नांची उत्तरे देऊ शकतो.",
    },
    "thanks": {
        "en": "You're welcome! Is there anything else I can help you with?",
        "hinglish": "Koi baat nahi! Kya aur kuch poochna chahte hain?",
        "punjabi": "ਕੋਈ ਗੱਲ ਨਹੀਂ! ਕੀ ਤੁਸੀਂ ਹੋਰ ਕੁਝ ਪੁੱਛਣਾ ਚਾਹੁੰਦੇ ਹੋ?",
        "marathi": "काही हरकत नाही! आणखी काही विचारायचे आहे का?",
    },
    "bye": {
        "en": "Goodbye! Have a great day!",
        "hinglish": "Alvida! Aapka din achha ho!",
        "punjabi": "ਅਲਵਿਦਾ! ਤੁਹਾਡਾ ਦਿਨ ਵਧੀਆ ਰਹੇ!",
        "marathi": "निरोप! तुमचा दिवस छान जाओ!",
    },
    "what_can_you_do": {
        "en": "I can answer questions from the knowledge base. Just ask me anything!",
        "hinglish": "Main knowledge base se sawaalon ke jawaab de sakta hun. Kuch bhi poochho!",
        "punjabi": "ਮੈਂ knowledge base ਤੋਂ ਸਵਾਲਾਂ ਦੇ ਜਵਾਬ ਦੇ ਸਕਦਾ ਹਾਂ।",
        "marathi": "मी knowledge base मधून प्रश्नांची उत्तरे देऊ शकतो.",
    },
}

_BEHAVIORAL_PATTERNS = [
    (
        [
            "hi",
            "hello",
            "hey",
            "hii",
            "namaste",
            "namaskar",
            "sat sri akal",
            "waheguru",
            "jai hind",
            "assalamu alaikum",
        ],
        "greeting",
    ),
    (
        [
            "how are you",
            "kaise ho",
            "kaise hain",
            "kaisa hai",
            "kasa ahat",
            "kasa ahes",
            "tussi kaise ho",
            "aap kaise hain",
            "kiddan",
            "ki haal hai",
            "kya haal hai",
        ],
        "how_are_you",
    ),
    (
        [
            "who are you",
            "what are you",
            "kaun ho",
            "kaun hain",
            "aap kaun ho",
            "tusi kaun ho",
            "tu kon ahes",
            "tumhi kon ahat",
            "apan kaun",
        ],
        "who_are_you",
    ),
    (
        [
            "thank you",
            "thanks",
            "shukriya",
            "dhanyawad",
            "dhanyavaad",
            "bahut shukriya",
            "thanks a lot",
            "aabhar",
        ],
        "thanks",
    ),
    (
        [
            "bye",
            "goodbye",
            "tata",
            "alvida",
            "phir milenge",
            "see you",
            "take care",
            "phir milte hain",
        ],
        "bye",
    ),
    (
        [
            "what can you do",
            "kya kar sakte ho",
            "kya karte ho",
            "help me",
            "help kar",
            "madad karo",
            "help karo",
        ],
        "what_can_you_do",
    ),
]


def _get_behavioral_response(text: str, lang: str) -> str | None:
    lowered = text.lower().strip()
    for patterns, intent in _BEHAVIORAL_PATTERNS:
        for p in patterns:
            if (
                lowered == p
                or lowered.startswith(p + " ")
                or lowered.endswith(" " + p)
                or f" {p} " in lowered
            ):
                return _BEHAVIORAL_RESPONSES[intent].get(lang, _BEHAVIORAL_RESPONSES[intent]["en"])
    return None


# ─── Voice Agent ───────────────────────────────────────────────────────────────

_TIMING_KEYS = ("stt_start", "stt_end", "rag_start", "rag_end", "tts_start", "tts_end")


class VoiceAgent:
    def __init__(self) -> None:
        credentials = _resolve_google_credentials()
        self.speech_client = speech.SpeechClient(credentials=credentials)
        self.tts_client = texttospeech.TextToSpeechClient(credentials=credentials)
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5.0, read=60.0, write=5.0, pool=5.0)
        )
        self.client_ws: WebSocket | None = None
        self.is_cancelled = False
        self.current_task: asyncio.Task | None = None
        self.bearer_token: str | None = REASONING_ENDPOINT_BEARER_TOKEN
        self.timing: dict = {k: None for k in _TIMING_KEYS}
        logger.info("VoiceAgent initialized", rag_endpoint=REASONING_ENDPOINT)

    # ─── STT ──────────────────────────────────────────────────────────────────

    async def transcribe_audio(self, audio_data: bytes) -> str:
        import time

        self.timing["stt_start"] = time.time()
        try:
            audio = speech.RecognitionAudio(content=audio_data)
            # SpeechContext boosts brand-name phrases so Google STT recognizes them
            
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code="en-IN",
                alternative_language_codes=["hi-IN", "pa-IN", "mr-IN"],
                enable_automatic_punctuation=True,
            )
            response = await asyncio.to_thread(
                self.speech_client.recognize, config=config, audio=audio
            )
            self.timing["stt_end"] = time.time()
            if response.results:
                transcript = response.results[0].alternatives[0].transcript
                logger.info(
                    "stt_done",
                    duration=round(self.timing["stt_end"] - self.timing["stt_start"], 2),
                    text=transcript[:80],
                )
                return transcript
            return ""
        except Exception as exc:
            logger.error("stt_error", error=str(exc))
            return ""

    # ─── TTS ──────────────────────────────────────────────────────────────────

    def _tts_voice_for_text(self, text: str) -> tuple[str, str]:
        """Return (language_code, voice_name) based on script detection."""
        if any("\u0a00" <= c <= "\u0a7f" for c in text):
            return "pa-IN", "pa-IN-Wavenet-B"
        if any("\u0900" <= c <= "\u097f" for c in text):
            if any(w in text for w in _MARATHI_MARKERS):
                return "mr-IN", "mr-IN-Wavenet-C"
            return "hi-IN", "hi-IN-Neural2-B"
        return "en-IN", "en-IN-Journey-D"

    async def synthesize(self, text: str) -> bytes:
        """Synthesize text → raw PCM bytes (LINEAR16, 24 kHz)."""
        if not text.strip():
            return b""
        language_code, voice_name = self._tts_voice_for_text(text)
        # Journey voices do NOT support SSML — they reject it with a 400 error.
        # Use plain text for Journey voices; SSML <sub alias> for Wavenet/Neural2.
        if "Journey" in voice_name:
            synthesis_input = texttospeech.SynthesisInput(text=text)
        else:
            ssml_text = _to_ssml(text)
            synthesis_input = texttospeech.SynthesisInput(ssml=ssml_text)
        voice = texttospeech.VoiceSelectionParams(language_code=language_code, name=voice_name)
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=24000,
        )
        try:
            resp = await asyncio.to_thread(
                self.tts_client.synthesize_speech,
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config,
            )
            return resp.audio_content
        except Exception as grpc_err:
            if any(k in str(grpc_err) for k in ("503", "handshaker", "UNAVAILABLE")):
                logger.warning("grpc_stale_channel_recreating_tts")
                self.tts_client = texttospeech.TextToSpeechClient(
                    credentials=_resolve_google_credentials()
                )
                resp = await asyncio.to_thread(
                    self.tts_client.synthesize_speech,
                    input=synthesis_input,
                    voice=voice,
                    audio_config=audio_config,
                )
                return resp.audio_content
            logger.error("tts_error", error=str(grpc_err))
            return b""

    # ─── RAG streaming pipeline ────────────────────────────────────────────────

    def _make_voice_friendly(self, text: str) -> str:
        text = re.sub(r"\s*\[\d+\]", "", text).strip()
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return " ".join(sentences[:3]) if len(sentences) > 3 else text

    async def _run_streaming_pipeline(self, question: str) -> None:  # noqa: C901
        import time

        self.timing.update(
            {"rag_start": time.time(), "rag_end": None, "tts_start": None, "tts_end": None}
        )

        lang = _detect_language(question)
        behavioral = _get_behavioral_response(question, lang)
        if behavioral:
            if self.client_ws:
                await self.client_ws.send_text(json.dumps({"type": "response", "text": behavioral}))
            audio = await self.synthesize(behavioral)
            if audio and self.client_ws and not self.is_cancelled:
                await self.client_ws.send_text(
                    json.dumps(
                        {
                            "type": "audio_chunk",
                            "data": base64.b64encode(audio).decode(),
                            "chunk_index": 0,
                            "total_chunks": 1,
                            "is_final": True,
                        }
                    )
                )
            return

        headers = {"Authorization": f"Bearer {self.bearer_token}"} if self.bearer_token else {}
        buffer = ""
        tts_tasks: list[asyncio.Task] = []
        full_parts: list[str] = []
        rag_lang = lang if lang != "en" else "auto"

        try:
            async with self.http_client.stream(
                "POST",
                REASONING_ENDPOINT,
                headers=headers,
                json={"query": question, "top_k": 5, "stream": True, "language": rag_lang},
            ) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    logger.error(
                        "rag_http_error",
                        status=resp.status_code,
                        body=body.decode(errors="replace")[:300],
                    )
                    if self.client_ws:
                        await self.client_ws.send_text(
                            json.dumps(
                                {
                                    "type": "response",
                                    "text": "Sorry, I couldn't retrieve an answer right now.",
                                }
                            )
                        )
                    return

                async for line in resp.aiter_lines():
                    if self.is_cancelled:
                        for t in tts_tasks:
                            t.cancel()
                        return
                    if not line.startswith("data:"):
                        continue
                    try:
                        payload = json.loads(line[5:].strip())
                    except (json.JSONDecodeError, ValueError):
                        continue

                    event = payload.get("event")
                    data = payload.get("data", "")

                    if event == "meta":
                        self.timing["rag_end"] = self.timing["tts_start"] = time.time()
                    elif event == "token":
                        token = data if isinstance(data, str) else ""
                        if not token:
                            continue
                        buffer += token
                        full_parts.append(token)
                        while True:
                            m = re.search(r"(?<=[.!?])\s+", buffer)
                            if not m:
                                break
                            sentence = buffer[: m.start() + 1].strip()
                            buffer = buffer[m.end() :]
                            if sentence:
                                tts_tasks.append(asyncio.create_task(self.synthesize(sentence)))
                    elif event in ("done", "error"):
                        if event == "error":
                            logger.error("rag_stream_error", data=data)
                        break

        except Exception as exc:
            logger.error("rag_streaming_exception", error=str(exc))
            if self.timing["rag_end"] is None:
                self.timing["rag_end"] = time.time()
            if self.client_ws and not self.is_cancelled:
                await self.client_ws.send_text(
                    json.dumps(
                        {
                            "type": "response",
                            "text": "Sorry, an error occurred retrieving the answer.",
                        }
                    )
                )
            return

        if self.is_cancelled:
            return
        if buffer.strip():
            tts_tasks.append(asyncio.create_task(self.synthesize(buffer.strip())))
        if not tts_tasks:
            return

        full_answer = self._make_voice_friendly("".join(full_parts))
        if self.client_ws and not self.is_cancelled:
            await self.client_ws.send_text(json.dumps({"type": "response", "text": full_answer}))

        total = len(tts_tasks)
        for i, task in enumerate(tts_tasks):
            if self.is_cancelled:
                task.cancel()
                continue
            try:
                audio = await task
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.error("tts_task_error", index=i, error=str(exc))
                continue
            if audio and self.client_ws and not self.is_cancelled:
                await self.client_ws.send_text(
                    json.dumps(
                        {
                            "type": "audio_chunk",
                            "data": base64.b64encode(audio).decode(),
                            "chunk_index": i,
                            "total_chunks": total,
                            "is_final": i == total - 1,
                        }
                    )
                )

        self.timing["tts_end"] = time.time()
        if self.client_ws and not self.is_cancelled:
            stt_s = self.timing["stt_start"]
            stt_e = self.timing["stt_end"]
            rag_s = self.timing["rag_start"]
            rag_e = self.timing["rag_end"] or self.timing["tts_start"] or time.time()
            tts_s = self.timing["tts_start"] or rag_e
            tts_e = self.timing["tts_end"]
            stt_t = (stt_e - stt_s) if (stt_s and stt_e) else 0
            rag_t = rag_e - rag_s
            tts_t = (tts_e - tts_s) if tts_e else 0
            await self.client_ws.send_text(
                json.dumps(
                    {
                        "type": "timing",
                        "timing": {
                            "stt": round(stt_t, 2),
                            "rag": round(rag_t, 2),
                            "tts": round(tts_t, 2),
                            "total": round(stt_t + rag_t + tts_t, 2),
                        },
                    }
                )
            )

    def _cancel(self) -> None:
        self.is_cancelled = True
        if self.current_task and not self.current_task.done():
            self.current_task.cancel()

    async def process_audio(self, audio_base64: str) -> None:
        self.is_cancelled = False
        try:
            audio_bytes = base64.b64decode(audio_base64)
            transcript = await self.transcribe_audio(audio_bytes)
            if not transcript:
                if self.client_ws:
                    await self.client_ws.send_text(
                        json.dumps({"type": "error", "message": "Could not transcribe audio"})
                    )
                return
            if self.client_ws:
                await self.client_ws.send_text(
                    json.dumps({"type": "transcript", "text": transcript})
                )
            await self._run_streaming_pipeline(transcript)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error("process_audio_error", error=str(exc))
            if self.client_ws:
                await self.client_ws.send_text(json.dumps({"type": "error", "message": str(exc)}))

    async def process_text(self, text: str) -> None:
        self.is_cancelled = False
        try:
            if self.client_ws:
                await self.client_ws.send_text(json.dumps({"type": "transcript", "text": text}))
            await self._run_streaming_pipeline(text)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error("process_text_error", error=str(exc))
            if self.client_ws:
                await self.client_ws.send_text(json.dumps({"type": "error", "message": str(exc)}))

    async def cleanup(self) -> None:
        await self.http_client.aclose()


# Singleton agent per process
_agent: VoiceAgent | None = None


def get_agent() -> VoiceAgent:
    global _agent
    if _agent is None:
        _agent = VoiceAgent()
    return _agent


# ─── REST TTS endpoint ─────────────────────────────────────────────────────────


class TTSRequest(BaseModel):
    text: str
    language: str = "auto"  # auto | en | hinglish | punjabi | marathi


class TTSResponse(BaseModel):
    audio_base64: str
    language_detected: str
    voice_used: str


@router.post(
    "/tts",
    summary="Text-to-Speech — returns base64-encoded LINEAR16 PCM audio at 24 kHz",
)
async def tts_endpoint(body: TTSRequest) -> TTSResponse:
    agent = get_agent()
    lang = _detect_language(body.text) if body.language == "auto" else body.language
    _, voice_name = agent._tts_voice_for_text(body.text)
    audio = await agent.synthesize(body.text)
    return TTSResponse(
        audio_base64=base64.b64encode(audio).decode(),
        language_detected=lang,
        voice_used=voice_name,
    )


@router.post(
    "/tts/raw",
    summary="Text-to-Speech — returns raw LINEAR16 PCM audio (audio/l16)",
    response_class=Response,
)
async def tts_raw_endpoint(body: TTSRequest) -> Response:
    agent = get_agent()
    audio = await agent.synthesize(body.text)
    return Response(content=audio, media_type="audio/l16; rate=24000")


# ─── Live-stream transcription endpoint ───────────────────────────────────────

_DEEPGRAM_API_KEY: str = settings.deepgram_api_key or os.getenv("DEEPGRAM_API_KEY", "")


class TranscribeStreamRequest(BaseModel):
    url: str
    language: str = "en"
    model: str = "nova-3"
    smart_format: bool = True
    interim_results: bool = True
    endpointing: int = 10


@router.post(
    "/transcribe-stream",
    summary="Transcribe a live audio stream URL — returns SSE transcript events",
    response_class=StreamingResponse,
)
async def transcribe_stream(body: TranscribeStreamRequest) -> StreamingResponse:
    """
    Stream audio from *body.url* through a live transcription service and
    return an SSE feed of transcript events:

    ```
    data: {"type":"transcript","text":"Hello world","is_final":true}
    ```
    """
    transcriber = StreamTranscriber(api_key=_DEEPGRAM_API_KEY)

    async def _sse_generator():
        async for event in transcriber.transcribe(
            body.url,
            language=body.language,
            model=body.model,
            smart_format=body.smart_format,
            interim_results=body.interim_results,
            endpointing=body.endpointing,
        ):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(_sse_generator(), media_type="text/event-stream")


# ─── Browser STT WebSocket ────────────────────────────────────────────────────

@router.websocket("/stt-ws")
async def stt_browser_websocket(websocket: WebSocket) -> None:
    """
    Accept raw audio chunks from the browser (webm/opus via MediaRecorder),
    pipe them to Deepgram live transcription, and stream transcript events back.

    Browser sends:  binary frames (audio/webm;codecs=opus chunks)
    Server returns: JSON text frames
        {"type": "transcript", "text": "...", "is_final": true/false}
        {"type": "error",      "message": "..."}
        {"type": "ready"}  — sent once Deepgram session is up
    """
    await websocket.accept()

    if not _DEEPGRAM_API_KEY:
        await websocket.send_json({
            "type": "error",
            "message": "Deepgram API key not configured (set DEEPGRAM_API_KEY)",
        })
        await websocket.close()
        return

    loop = asyncio.get_running_loop()
    session = BrowserSTTSession(api_key=_DEEPGRAM_API_KEY, loop=loop)
    session.start()
    await websocket.send_json({"type": "ready"})
    logger.info("stt_browser_ws_connected")

    # ── consumer: read transcript events from Deepgram → forward to browser ──
    async def _forward_transcripts() -> None:
        async for event in session.events():
            try:
                await websocket.send_json(event)
            except Exception:
                break

    forward_task = asyncio.create_task(_forward_transcripts())

    try:
        while True:
            chunk = await websocket.receive_bytes()
            session.send_audio(chunk)
    except WebSocketDisconnect:
        logger.info("stt_browser_ws_disconnected")
    except Exception as exc:
        logger.error("stt_browser_ws_error", error=str(exc))
    finally:
        session.stop()
        forward_task.cancel()


# ─── WebSocket voice agent endpoint ───────────────────────────────────────────


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    agent = get_agent()
    agent.client_ws = websocket
    logger.info("voice_ws_connected")

    try:
        while True:
            message = await websocket.receive_text()
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                continue

            msg_type = data.get("type")

            if msg_type == "set_token":
                agent.bearer_token = (
                    data.get("token") or ""
                ).strip() or REASONING_ENDPOINT_BEARER_TOKEN
            elif msg_type == "cancel":
                agent._cancel()
            elif msg_type in ("audio", "text"):
                agent._cancel()
                if agent.current_task and not agent.current_task.done():
                    try:
                        await asyncio.wait_for(asyncio.shield(agent.current_task), timeout=0.15)
                    except (TimeoutError, asyncio.CancelledError):
                        pass
                agent.is_cancelled = False
                if msg_type == "audio":
                    agent.current_task = asyncio.create_task(
                        agent.process_audio(data.get("data", ""))
                    )
                else:
                    agent.current_task = asyncio.create_task(
                        agent.process_text(data.get("data", ""))
                    )

    except WebSocketDisconnect:
        logger.info("voice_ws_disconnected")
    except Exception as exc:
        logger.error("voice_ws_error", error=str(exc))
    finally:
        agent._cancel()
        agent.client_ws = None
