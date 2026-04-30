<div align="center">

# 🧠 Multi RAG Conversational AI Model

**A production-ready RAG system with Voice AI — semantic vector search, real-time STT, and TTS in a single conversational backend**

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat&logo=fastapi&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-4169E1?style=flat&logo=postgresql&logoColor=white)
![Vertex AI](https://img.shields.io/badge/Vertex_AI-Embeddings-4285F4?style=flat&logo=googlecloud&logoColor=white)
![Deepgram](https://img.shields.io/badge/Deepgram-Nova--3_STT-101010?style=flat&logo=deepgram&logoColor=white)
![Google TTS](https://img.shields.io/badge/Google-TTS-DB4437?style=flat&logo=google&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat&logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

*Ingest any text document, query it with natural language, and talk to it — full voice loop with STT → RAG → TTS.*

</div>

---

## 📌 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [How Multi-RAG Works](#how-multi-rag-works)
- [STT — Two Approaches](#stt--two-approaches)
- [TTS — Google Cloud Text-to-Speech](#tts--google-cloud-text-to-speech)
- [Voice Agent Pipeline](#voice-agent-pipeline)
- [API Endpoints](#api-endpoints)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Local Setup](#local-setup)
- [Environment Variables](#environment-variables)

---

## Overview

This system allows you to:

1. **Ingest** any text document into a vector database
2. **Query** it using natural language — results are ranked by semantic similarity
3. **Talk to it** — speak a question, get a spoken answer back (full voice loop)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        CLIENT                               │
│         Browser / Mobile / WebSocket Client                 │
└────────────┬──────────────────────────┬────────────────────┘
             │ REST                     │ WebSocket
             ▼                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   FastAPI Backend (port number)               │
│                                                             │
│   ┌──────────────┐   ┌──────────────┐   ┌───────────────┐  │
│   │  RAG Routes  │   │ Voice Routes │   │  Health/Docs  │  │
│   │  /api/v1/rag │   │/api/v1/voice │   │  /health      │  │
│   └──────┬───────┘   └──────┬───────┘   └───────────────┘  │
│          │                  │                               │
│   ┌──────▼───────┐   ┌──────▼──────────────────────────┐   │
│   │  RAG Pipeline│   │        Voice Agent               │   │
│   │  - Chunker   │   │  STT → RAG Query → TTS           │   │
│   │  - Embedder  │   │                                  │   │
│   │  - VectorDB  │   └──────────────────────────────────┘   │
│   └──────┬───────┘                                          │
└──────────┼──────────────────────────────────────────────────┘
           │
    ┌──────▼──────────────────────────┐
    │         PostgreSQL + pgvector   │
    │   rag_documents | rag_chunks    │
    │   HNSW index (cosine distance)  │
    └─────────────────────────────────┘
```

---

## How Multi-RAG Works

RAG = **Retrieval** + **Augmented** + **Generation**

Instead of relying purely on a language model's training data, RAG first retrieves relevant context from your own documents, then uses that context to answer the question.

### Step 1 — Ingestion Pipeline

```
Raw Text
   │
   ▼
┌──────────────────────────────────────────┐
│  chunker.py — Sliding Window Chunking    │
│                                          │
│  chunk_size    = 512 tokens              │
│  chunk_overlap = 64 tokens               │
│                                          │
│  "The quick brown fox..." →              │
│  [chunk_0: "The quick brown..."]         │
│  [chunk_1: "...brown fox jumps..."]  ←── overlap
│  [chunk_2: "...jumps over the..."]       │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│  embedder.py — Vertex AI Embeddings      │
│                                          │
│  Model: text-embedding-004 (768-dim)     │
│                                          │
│  "The quick brown fox" →                 │
│  [0.023, -0.412, 0.891, ...]  (768 floats)
│                                          │
│  LRU Cache (500 entries) for speed       │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│  PostgreSQL + pgvector                   │
│                                          │
│  Table: con.rag_chunks                │
│  Column: embedding vector(768)           │
│  Index:  HNSW (cosine ops)               │
│          m=16, ef_construction=64        │
└──────────────────────────────────────────┘
```

### Step 2 — Query Pipeline

```
User Question: "What is machine learning?"
   │
   ▼
┌──────────────────────────────────────────┐
│  embed_query() — same embedding model    │
│  "What is machine learning?" →           │
│  [0.041, -0.389, 0.762, ...]             │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│  vector_store.py — pgvector Search       │
│                                          │
│  SELECT ... FROM rag_chunks              │
│  ORDER BY embedding <=> query_vector     │
│  WHERE similarity >= 0.30                │
│  LIMIT 5                                 │
│                                          │
│  Returns top-K chunks ranked by          │
│  cosine similarity score                 │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│  Response                                │
│  {                                       │
│    chunk_id: "...",                      │
│    content:  "Machine learning is...",   │
│    score:    0.87,                       │
│    filename: "ml-intro.txt"              │
│  }                                       │
└──────────────────────────────────────────┘
```

### Why "Multi" RAG?

This system supports **multiple retrieval strategies** that can be combined:

| Strategy | How | When |
|---|---|---|
| **Vector Search** | pgvector cosine similarity on embeddings | Semantic / meaning-based queries |
| **Full-Text Search** | PostgreSQL `tsvector` + GIN index (BM25) | Exact keyword matching |
| **Hybrid** | Both combined via Reciprocal Rank Fusion (RRF) | Best of both worlds |

---

## STT — Two Approaches

The system implements Speech-to-Text in **two completely different ways**:

---

### Approach 1 — Google Cloud Speech API (Server-side batch STT)

Used in the **Voice Agent WebSocket** (`/api/v1/voice/ws`).

```
Browser
  │
  │  WebSocket message: { type: "audio", data: "<base64 PCM>" }
  │
  ▼
VoiceAgent.transcribe_audio()
  │
  │  audio bytes → Google Speech-to-Text API
  │  - Encoding: LINEAR16
  │  - Sample rate: 16000 Hz
  │  - Primary language: en-IN
  │  - Fallback languages: hi-IN, pa-IN, mr-IN
  │
  ▼
transcript: "What is machine learning?"
```

**How it works:**
- Browser records audio and sends it as a base64-encoded blob over WebSocket
- Server decodes it and sends the raw bytes to Google Speech API in one batch call
- Google returns the full transcript
- Supports multilingual detection (English, Hindi, Punjabi, Marathi)

**Code:** `services/voice.py → VoiceAgent.transcribe_audio()`

---

### Approach 2 — Deepgram Live Streaming STT (Real-time browser STT)

Used in **two sub-modes**:

#### 2a — Browser WebSocket STT (`/api/v1/voice/stt-ws`)

```
Browser (MediaRecorder API)
  │
  │  Binary WebSocket frames (webm/opus chunks, ~250ms each)
  │
  ▼
BrowserSTTSession (services/stream_transcription.py)
  │
  ├── Feeder Thread: reads chunks from queue → conn.send_media()
  │
  └── Deepgram WebSocket Connection
        - Model: nova-3
        - Language: multi (auto-detect)
        - interim_results: true  ← partial transcripts while speaking
        - endpointing: 300ms
        │
        ▼
      Transcript events streamed back to browser:
      { type: "transcript", text: "What is...", is_final: false }
      { type: "transcript", text: "What is machine learning?", is_final: true }
```

**How it works:**
- Browser uses `MediaRecorder` API to capture microphone audio
- Audio chunks are sent as binary WebSocket frames to the server
- Server pipes them in real-time to Deepgram's live transcription WebSocket
- Deepgram sends back partial + final transcripts as they are recognized
- Server forwards these events back to the browser

**Code:** `services/stream_transcription.py → BrowserSTTSession`

#### 2b — Server-side Stream URL Transcription (`POST /api/v1/voice/transcribe-stream`)

```
POST /api/v1/voice/transcribe-stream
{ "url": "http://radio-stream.example.com/audio" }
  │
  ▼
StreamTranscriber
  │
  ├── Background Thread 1: httpx streams audio bytes from URL
  │                        → conn.send_media(chunk)
  │
  └── Deepgram WebSocket Connection
        │
        ▼
      SSE response stream to caller:
      data: {"type":"transcript","text":"Hello world","is_final":true}
      data: {"type":"transcript","text":"How are you","is_final":true}
```

**How it works:**
- Caller provides a URL of a live audio stream (e.g. radio, RTMP, HLS)
- Server fetches the audio stream using `httpx` and pipes bytes to Deepgram
- Transcripts are returned as Server-Sent Events (SSE) to the caller

**Code:** `services/stream_transcription.py → StreamTranscriber`

---

### STT Comparison

| Feature | Google Speech API | Deepgram (Browser WS) | Deepgram (Stream URL) |
|---|---|---|---|
| Input | Base64 audio blob | Binary WS frames | HTTP audio stream URL |
| Mode | Batch (full audio) | Real-time streaming | Real-time streaming |
| Interim results | ❌ | ✅ | ✅ |
| Multilingual | ✅ (en/hi/pa/mr) | ✅ (auto-detect) | ✅ |
| Use case | Voice agent loop | Browser mic input | Server-side streams |

---

## TTS — Google Cloud Text-to-Speech

**Endpoint:** `POST /api/v1/voice/tts`

```
Text: "Machine learning is a subset of AI"
  │
  ▼
_detect_language()
  │  Detects: English / Hinglish / Punjabi / Marathi
  │  based on Unicode script + keyword markers
  │
  ▼
_tts_voice_for_text()
  │
  ├── Punjabi script (ਪੰਜਾਬੀ)  → pa-IN-Wavenet-B
  ├── Hindi/Marathi (देवनागरी) → hi-IN-Neural2-B / mr-IN-Wavenet-C
  └── English / Hinglish       → en-IN-Journey-D
  │
  ▼
Google Cloud TTS API
  │  - Audio encoding: LINEAR16
  │  - Sample rate: 24000 Hz
  │  - SSML support for Wavenet/Neural2 voices
  │
  ▼
Response: base64-encoded PCM audio bytes
```

**Available endpoints:**

| Endpoint | Response | Use case |
|---|---|---|
| `POST /api/v1/voice/tts` | JSON with `audio_base64` | Browser playback |
| `POST /api/v1/voice/tts/raw` | Raw PCM bytes (`audio/l16`) | Direct audio pipe |

---

## Voice Agent Pipeline

Full end-to-end voice conversation loop via WebSocket `/api/v1/voice/ws`:

```
Browser
  │
  │  1. { type: "audio", data: "<base64 PCM>" }
  │
  ▼
VoiceAgent.process_audio()
  │
  │  2. Google STT → transcript
  │     "What is deep learning?"
  │
  ▼
  │  3. Send transcript back to browser
  │     { type: "transcript", text: "What is deep learning?" }
  │
  ▼
_run_streaming_pipeline()
  │
  │  4. Check behavioral patterns (greetings, thanks, bye)
  │     → instant response without RAG
  │
  │  5. POST /api/v1/rag/query
  │     Stream response tokens as they arrive
  │
  │  6. Sentence-level TTS — synthesize each sentence
  │     as soon as it's complete (parallel tasks)
  │
  ▼
  │  7. Stream audio chunks back to browser
  │     { type: "audio_chunk", data: "<base64>", chunk_index: 0, is_final: false }
  │     { type: "audio_chunk", data: "<base64>", chunk_index: 1, is_final: true  }
  │
  │  8. Send timing stats
  │     { type: "timing", stt: 0.8, rag: 1.2, tts: 0.4, total: 2.4 }
  │
  ▼
Browser plays audio chunks in order
```

---

## API Endpoints

### RAG

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/rag/ingest` | Ingest text into knowledge base |
| `POST` | `/api/v1/rag/query` | Semantic search query |
| `GET` | `/api/v1/rag/documents` | List all documents |
| `GET` | `/api/v1/rag/documents/{id}` | Get document details |
| `DELETE` | `/api/v1/rag/documents/{id}` | Delete document + chunks |

### Voice

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/voice/tts` | Text → base64 audio |
| `POST` | `/api/v1/voice/tts/raw` | Text → raw PCM audio |
| `POST` | `/api/v1/voice/transcribe-stream` | Stream URL → SSE transcripts (Deepgram) |
| `WS` | `/api/v1/voice/stt-ws` | Browser mic → live transcripts (Deepgram) |
| `WS` | `/api/v1/voice/ws` | Full voice agent loop |

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Framework** | FastAPI + Uvicorn |
| **Database** | PostgreSQL 16 + pgvector extension |
| **Embeddings** | Google Vertex AI `text-embedding-004` (768-dim) |
| **Vector Index** | HNSW (Hierarchical Navigable Small World) |
| **STT (batch)** | Google Cloud Speech-to-Text API |
| **STT (stream)** | Deepgram Nova-3 (live WebSocket) |
| **TTS** | Google Cloud Text-to-Speech (Journey / Neural2 / Wavenet) |
| **Cache** | Redis + in-process LRU (embeddings) |
| **Async** | Python asyncio + asyncpg |
| **Logging** | structlog |

---

## Project Structure

```
src/
├── main.py                    # FastAPI app entry point
├── deps.py                    # Shared dependencies
├── config/
│   └── config_local.py        # Local environment settings
├── db/
│   ├── models.py              # SQLAlchemy models (RagDocument, RagChunk)
│   └── session.py             # DB engine, session factory, schema init
├── routes/
│   ├── rag.py                 # RAG REST endpoints
│   └── voice.py               # TTS / STT / Voice agent endpoints
├── services/
│   ├── chunker.py             # Sliding window text chunker
│   ├── embedder.py            # Vertex AI embeddings + LRU cache
│   ├── ingestion.py           # Full ingest pipeline
│   ├── vector_store.py        # pgvector search + upsert
│   └── stream_transcription.py # Deepgram STT (browser WS + stream URL)

```

---

## Local Setup

### Prerequisites

- Python 3.11+
- Docker Desktop
- Google Cloud service account with Vertex AI + Speech + TTS APIs enabled
- Deepgram API key

### 1. Start PostgreSQL with pgvector

```bash
docker run -d --name postgres-local \
  -e POSTGRES_USER=your_user \
  -e POSTGRES_PASSWORD=your_pass \
  -e POSTGRES_DB=your_schema \
  -p your_port:your_port \
  pgvector/pgvector:pg16
```


### 2. Install dependencies

```bash
cd src
pip install -r requirements.txt
```

### 3. Configure environment

Copy `.env` and fill in your credentials (see below).

### 4. Run the server

```bash
cd src
uvicorn main:app --host 0.0.0.0 --port your_port --reload
```
---

## License

MIT
