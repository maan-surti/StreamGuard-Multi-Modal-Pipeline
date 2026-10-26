# 🛡️ StreamGuard: Multi-Modal Deepfake Audio Detection Pipeline

![React](https://img.shields.io/badge/React-18.x-blue?style=flat-square&logo=react)
![Next.js](https://img.shields.io/badge/Next.js-14.x-black?style=flat-square&logo=next.js)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi)
![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-FF9D00?style=flat-square)

StreamGuard is a real-time, multi-modal streaming data pipeline and cybersecurity dashboard designed to detect AI-generated job candidates during live technical interviews. Built for high-throughput stream mining and big data analysis, it intercepts raw audio, decodes it into text, and evaluates acoustic and semantic signatures simultaneously to generate a real-time Trust Score.

## 🧠 Core Architecture & Algorithms

This pipeline implements foundational Big Data stream-mining algorithms entirely in Python to process infinite data streams within constrained memory limits:

*   **Acoustic Deepfake Detection (Hugging Face):** Evaluates 16kHz audio arrays against the `MelodyMachine/Deepfake-audio-detection-V2` model to detect synthetic voice artifacts.
*   **Semantic Speech-to-Text (OpenAI Whisper):** Transcribes real-time audio chunks with a finely-tuned silence threshold (0.005) for optimized GPU/CPU inference.
*   **Bloom Filters ($O(1)$):** Space-efficient probabilistic data structure used to blacklist AI-generated colloquialisms (e.g., "as an AI", "delve into").
*   **DGIM Velocity Tracking ($O(\log^2 N)$):** Sliding window anomaly detection that counts the frequency of suspicious phrases in the data stream using exponentially sized buckets.
*   **Flajolet-Martin Sketch ($O(1)$):** Probabilistic algorithm to estimate the number of distinct IP addresses (callers) hitting the WebSocket server by tracking trailing binary zeros.
*   **TF-IDF Cosine Similarity:** Vectorizes the live transcript and calculates spatial similarity against known AI interview "cheat sheets."

## 💻 Cyberpunk Dashboard
The frontend is a `Next.js` single-page application utilizing a `WebSocket` connection to the Python backend. It features:
*   Real-time intercept log with JSON payload inspection.
*   Live `Recharts` Threat Timeline graphing Trust Scores vs. Acoustic Signatures.
*   CSS-based acoustic hardware sensor.
*   Recruiter Batch Analysis portal for analyzing static `.wav` and `.mp3` files via a FastAPI `POST` endpoint.

## 🚀 Local Deployment

### 1. Backend (FastAPI & ML Models)
Ensure you have `ffmpeg` installed on your OS for Whisper audio processing (Windows: `winget install ffmpeg`).
```bash
# Clone the repository
git clone [https://github.com/maan-surti/StreamGuard-Multi-Modal-Pipeline.git](https://github.com/maan-surti/StreamGuard-Multi-Modal-Pipeline.git)
cd StreamGuard-Multi-Modal-Pipeline

# Install Python dependencies
pip install fastapi uvicorn whisper transformers torch numpy python-multipart

# Start the ML Server (Downloads HF/Whisper models to RAM on first boot)
python server.py
