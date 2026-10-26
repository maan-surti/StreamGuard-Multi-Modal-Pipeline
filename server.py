import os
import json
import tempfile
import uvicorn
import numpy as np
import whisper
import hashlib
import math
from collections import Counter
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline

# ==========================================
# 1. STREAM-MINING ALGORITHMS
# ==========================================

class BloomFilter:
    def __init__(self, size=1024, hash_count=3):
        self.size = size
        self.hash_count = hash_count
        self.bit_array = [0] * size

    def _hashes(self, item: str):
        item_bytes = item.encode('utf-8')
        return [(int(hashlib.sha256(item_bytes + str(i).encode()).hexdigest(), 16) % self.size) for i in range(self.hash_count)]

    def add(self, item: str):
        for idx in self._hashes(item):
            self.bit_array[idx] = 1

    def might_contain(self, item: str) -> bool:
        return all(self.bit_array[idx] for idx in self._hashes(item))


class DGIMVelocityTracker:
    def __init__(self, window_size=15):
        self.window_size = window_size
        self.buckets = []
        self.clock = 0

    def add_event(self, is_suspicious: bool):
        self.clock += 1
        self.buckets = [b for b in self.buckets if self.clock - b[0] < self.window_size]
        if is_suspicious:
            self.buckets.append([self.clock, 1])
            self._merge_buckets()

    def _merge_buckets(self):
        changed = True
        while changed:
            changed = False
            size_map = {}
            for i, b in enumerate(self.buckets):
                size_map.setdefault(b[1], []).append(i)
            for size, indices in size_map.items():
                if len(indices) >= 3:
                    idx_oldest, idx_second = sorted(indices)[:2]
                    merged_bucket = [self.buckets[idx_second][0], size * 2]
                    for idx in sorted([idx_oldest, idx_second], reverse=True):
                        del self.buckets[idx]
                    self.buckets.append(merged_bucket)
                    changed = True
                    break
        self.buckets.sort(key=lambda b: b[0])

    def estimate_count(self) -> int:
        if not self.buckets: return 0
        sorted_buckets = sorted(self.buckets, key=lambda b: b[0])
        return sum(b[1] for b in sorted_buckets[1:]) + (sorted_buckets[0][1] // 2)


class FlajoletMartin:
    def __init__(self):
        self.max_trailing_zeros = 0
        self.phi = 0.77351

    def add(self, ip_address: str):
        hash_int = int(hashlib.sha256(ip_address.encode('utf-8')).hexdigest(), 16)
        trailing_zeros = 0
        while (hash_int & 1) == 0 and hash_int > 0:
            trailing_zeros += 1
            hash_int >>= 1
        self.max_trailing_zeros = max(self.max_trailing_zeros, trailing_zeros)

    def estimate(self) -> int:
        if self.max_trailing_zeros == 0:
            return 0
        return int((2 ** self.max_trailing_zeros) / self.phi)


class TFIDFMatcher:
    def __init__(self, reference_corpus):
        self.reference_corpus = reference_corpus
        self.vocab = set()
        self.idf = {}
        self.reference_vectors = []
        self._build_idf()
        self._build_reference_vectors()

    def _get_words(self, text):
        return [w.strip(".,!?\"'").lower() for w in text.split() if w.strip(".,!?\"'")]

    def _build_idf(self):
        N = len(self.reference_corpus)
        doc_freqs = Counter()
        for doc in self.reference_corpus:
            words = set(self._get_words(doc))
            self.vocab.update(words)
            for w in words:
                doc_freqs[w] += 1
        for w in self.vocab:
            self.idf[w] = math.log((1 + N) / (1 + doc_freqs[w])) + 1

    def _build_reference_vectors(self):
        for doc in self.reference_corpus:
            self.reference_vectors.append(self._get_vector(doc))

    def _get_vector(self, text):
        words = self._get_words(text)
        tf = Counter(words)
        vec = {}
        total_words = len(words)
        if total_words == 0: return vec
        for w, count in tf.items():
            if w in self.vocab:
                vec[w] = (count / total_words) * self.idf[w]
        return vec

    def _cosine_similarity(self, vec1, vec2):
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum([vec1[x] * vec2[x] for x in intersection])
        sum1 = sum([vec1[x]**2 for x in vec1.keys()])
        sum2 = sum([vec2[x]**2 for x in vec2.keys()])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)
        if not denominator: return 0.0
        return float(numerator) / denominator

    def get_max_similarity(self, input_text):
        input_vec = self._get_vector(input_text)
        if not input_vec: return 0.0
        max_sim = 0.0
        for ref_vec in self.reference_vectors:
            sim = self._cosine_similarity(input_vec, ref_vec)
            max_sim = max(max_sim, sim)
        return max_sim * 100

# ==========================================
# 2. FASTAPI SETUP & BROADCAST MANAGER
# ==========================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception:
                pass

manager = ConnectionManager()

print("Loading Whisper ML Model into RAM...")
ml_model = whisper.load_model("small")

print("Loading Acoustic Deepfake Model into RAM...")
acoustic_model = pipeline("audio-classification", model="MelodyMachine/Deepfake-audio-detection-V2")

ai_blacklist = BloomFilter()
for phrase in ["as an ai", "language model", "delve into", "best practices", "comprehensive overview"]:
    ai_blacklist.add(phrase)

velocity_tracker = DGIMVelocityTracker(window_size=15)
ip_tracker = FlajoletMartin()

KNOWN_CHEATSHEET = [
    "Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data.",
    "A binary search tree is a data structure where each node has at most two children, referred to as the left and right child.",
    "Object-oriented programming is a programming paradigm based on the concept of objects, which can contain data and code."
]
text_matcher = TFIDFMatcher(KNOWN_CHEATSHEET)

# ------------------------------------------
# ENDPOINTS
# ------------------------------------------

@app.post("/analyze-file")
async def analyze_file(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(await file.read())
        temp_path = temp_audio.name

    try:
        audio_np = whisper.load_audio(temp_path)
        sample_length = min(len(audio_np), 16000 * 30)
        acoustic_input = {"raw": audio_np[:sample_length], "sampling_rate": 16000}
        acoustic_results = acoustic_model(acoustic_input, top_k=2)
        fake_score = next((r['score'] * 100 for r in acoustic_results if r['label'].lower() == 'fake'), 0.0)

        result = ml_model.transcribe(audio_np, fp16=False, language="en")
        transcript = result['text'].strip().lower()
        
        words = transcript.split()
        blacklisted_words = [w for w in words if ai_blacklist.might_contain(w)]
        similarity_percent = text_matcher.get_max_similarity(transcript)

        is_suspicious = (fake_score > 80.0) or (similarity_percent > 50.0) or (len(blacklisted_words) > 3)

        return {
            "filename": file.filename,
            "acoustic_fake_percent": round(fake_score, 1),
            "text_similarity": round(similarity_percent, 1),
            "blacklisted_hits": len(blacklisted_words),
            "transcript_preview": transcript[:200] + "...",
            "verdict": "⚠️ AI GENERATED" if is_suspicious else "✅ LIKELY HUMAN"
        }
    finally:
        os.remove(temp_path)

@app.websocket("/dashboard-stream")
async def dashboard_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.websocket("/audio-stream")
async def audio_stream_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_ip = websocket.client.host if websocket.client else "127.0.0.1"
    ip_tracker.add(client_ip)
    
    audio_buffer = []
    SAMPLE_RATE = 16000
    WINDOW_SECONDS = 5

    try:
        while True:
            data = await websocket.receive_bytes()
            chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            audio_buffer.extend(chunk)

            if len(audio_buffer) >= SAMPLE_RATE * WINDOW_SECONDS:
                audio_np = np.array(audio_buffer, dtype=np.float32)
                
                if np.max(np.abs(audio_np)) < 0.005: 
                    audio_buffer.clear()
                    continue
                
                acoustic_input = {"raw": audio_np, "sampling_rate": SAMPLE_RATE}
                acoustic_results = acoustic_model(acoustic_input, top_k=2)
                fake_score = next((r['score'] * 100 for r in acoustic_results if r['label'].lower() == 'fake'), 0.0)

                result = ml_model.transcribe(audio_np, fp16=False, language="en", initial_prompt="Technical AIML job interview.")
                transcript = result['text'].strip().lower()

                if transcript:
                    words = transcript.split()
                    is_blacklisted = any(ai_blacklist.might_contain(word) for word in words)
                    velocity_tracker.add_event(is_suspicious=is_blacklisted)
                    
                    suspicious_count = velocity_tracker.estimate_count()
                    estimated_distinct_ips = ip_tracker.estimate()
                    similarity_percent = text_matcher.get_max_similarity(transcript)

                    base_score = 100
                    if fake_score > 96.0: base_score -= 50
                    if is_blacklisted: base_score -= 30
                    if similarity_percent > 60.0: base_score -= 20
                    base_score -= (suspicious_count * 10)
                    
                    authenticity_score = max(0, min(100, int(base_score)))

                    payload = {
                        "transcript": transcript,
                        "acoustic_fake_percent": round(fake_score, 1),
                        "text_similarity": round(similarity_percent, 1),
                        "is_blacklisted": is_blacklisted,
                        "suspicious_count": suspicious_count,
                        "estimated_ips": estimated_distinct_ips,
                        "authenticity_score": authenticity_score,
                        "status": "Needs Review" if authenticity_score < 60 else "Genuine"
                    }
                    await manager.broadcast(payload)

                audio_buffer.clear()
    except WebSocketDisconnect:
        pass

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)