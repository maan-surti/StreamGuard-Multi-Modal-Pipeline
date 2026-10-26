"use client";

/* ============================================================
   STREAMGUARD — Real-Time Deepfake Detection Dashboard
   Single-file Next.js page (App Router, "use client")
   Stack: React 18/19 hooks + Tailwind + recharts
   ============================================================ */

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
  type DragEvent,
} from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

/* ---------------- Backend endpoints ---------------- */

const WS_URL = "ws://127.0.0.1:8000/dashboard-stream";
const ANALYZE_URL = "http://127.0.0.1:8000/analyze-file";

/* ---------------- Types ---------------- */

type Severity = "CRITICAL" | "WARNING" | "INFO";
type LinkStatus = "CONNECTING" | "ONLINE" | "OFFLINE";
type UploadStatus = "IDLE" | "UPLOADING" | "SUCCESS" | "ERROR";

interface InterceptMessage {
  id: number;
  timestamp: string;
  severity: Severity;
  source: string;
  trust: number | null;
  acoustic: number | null;
  tfidf: number | null;
  ips: number | null;
  raw: unknown;
}

interface MetricState {
  trust: number | null;
  acoustic: number | null;
  tfidf: number | null;
  ips: number | null;
}

interface FileUploadResult {
  filename: string;
  acoustic_fake_percent: number;
  text_similarity: number;
  blacklisted_hits: number;
  transcript_preview: string;
  verdict: string;
}

/* ---------------- Payload extraction helpers ---------------- */

function pickNumber(payload: Record<string, unknown>, keys: string[]): number | null {
  for (const key of keys) {
    const value = payload[key];
    if (typeof value === "number" && Number.isFinite(value)) return value;
    if (typeof value === "string" && value.trim() !== "" && !Number.isNaN(Number(value))) {
      return Number(value);
    }
  }
  return null;
}

function extractMetrics(payload: Record<string, unknown>): MetricState {
  return {
    trust: pickNumber(payload, ["trust_score", "trustScore", "trust", "confidence", "authenticity_score"]),
    acoustic: pickNumber(payload, [
      "acoustic_ai_signature",
      "acoustic_signature",
      "acousticSignature",
      "acoustic",
      "acoustic_fake_percent"
    ]),
    tfidf: pickNumber(payload, [
      "tfidf_similarity",
      "tf_idf_similarity",
      "tfidfSimilarity",
      "similarity",
      "text_similarity",
      "tfidf",
    ]),
    ips: pickNumber(payload, [
      "flajolet_martin_ips",
      "flajolet_martin_unique_ips",
      "unique_ips",
      "uniqueIps",
      "ip_count",
      "estimated_ips",
      "ips",
    ]),
  };
}

function deriveSeverity(trust: number | null): Severity {
  if (trust === null) return "INFO";
  if (trust < 55) return "CRITICAL";
  if (trust < 80) return "WARNING";
  return "INFO";
}

/* ---------------- Small presentational atoms ---------------- */

function Corners() {
  const base =
    "pointer-events-none absolute h-3 w-3 border-cyan-400/70 transition-all duration-300";
  return (
    <>
      <span className={`${base} left-0 top-0 border-l-2 border-t-2`} />
      <span className={`${base} right-0 top-0 border-r-2 border-t-2`} />
      <span className={`${base} bottom-0 left-0 border-b-2 border-l-2`} />
      <span className={`${base} bottom-0 right-0 border-b-2 border-r-2`} />
    </>
  );
}

const SEVERITY_STYLES: Record<Severity, string> = {
  CRITICAL: "text-[#ff2d6f] border-[#ff2d6f]/40 bg-[#ff2d6f]/10",
  WARNING: "text-[#ffb020] border-[#ffb020]/40 bg-[#ffb020]/10",
  INFO: "text-[#00e5ff] border-[#00e5ff]/40 bg-[#00e5ff]/10",
};

/* ============================================================
   Main component
   ============================================================ */

export default function StreamGuardDashboard() {
  const [messages, setMessages] = useState<InterceptMessage[]>([]);
  const [linkStatus, setLinkStatus] = useState<LinkStatus>("CONNECTING");
  const idCounter = useRef(0);
  const logEndRef = useRef<HTMLDivElement | null>(null);

  const [metrics, setMetrics] = useState<MetricState>({
    trust: null,
    acoustic: null,
    tfidf: null,
    ips: null,
  });
  const [deltas, setDeltas] = useState({ trust: 0, acoustic: 0, tfidf: 0, ips: 0 });
  const [clock, setClock] = useState<string>("--:--:-- UTC");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploadStatus, setUploadStatus] = useState<UploadStatus>("IDLE");
  const [uploadResult, setUploadResult] = useState<FileUploadResult | null>(null);
  const [uploadError, setUploadError] = useState<string>("");
  const [uploadLatency, setUploadLatency] = useState<number | null>(null);
  const [dragging, setDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const ingest = useCallback((payload: unknown) => {
    if (typeof payload !== "object" || payload === null) return;
    const data = payload as Record<string, unknown>;
    const extracted = extractMetrics(data);
    const id = ++idCounter.current;

    const entry: InterceptMessage = {
      id,
      timestamp: new Date().toISOString().slice(11, 19),
      severity: deriveSeverity(extracted.trust),
      source:
        (typeof data.source === "string" && data.source) ||
        (typeof data.channel === "string" && data.channel) ||
        "INTERCEPT-NODE",
      trust: extracted.trust,
      acoustic: extracted.acoustic,
      tfidf: extracted.tfidf,
      ips: extracted.ips,
      raw: data,
    };

    setMessages((prev) => [...prev.slice(-99), entry]);

    setMetrics((prev) => {
      setDeltas({
        trust: extracted.trust !== null && prev.trust !== null ? extracted.trust - prev.trust : 0,
        acoustic: extracted.acoustic !== null && prev.acoustic !== null ? extracted.acoustic - prev.acoustic : 0,
        tfidf: extracted.tfidf !== null && prev.tfidf !== null ? extracted.tfidf - prev.tfidf : 0,
        ips: extracted.ips !== null && prev.ips !== null ? extracted.ips - prev.ips : 0,
      });
      return {
        trust: extracted.trust ?? prev.trust,
        acoustic: extracted.acoustic ?? prev.acoustic,
        tfidf: extracted.tfidf ?? prev.tfidf,
        ips: extracted.ips ?? prev.ips,
      };
    });
  }, []);

  useEffect(() => {
    let socket: WebSocket | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    let attempts = 0;
    let cancelled = false;

    const connect = () => {
      if (cancelled) return;
      setLinkStatus("CONNECTING");
      try {
        socket = new WebSocket(WS_URL);
      } catch {
        setLinkStatus("OFFLINE");
        attempts += 1;
        reconnectTimer = setTimeout(connect, Math.min(1000 * 2 ** attempts, 10000));
        return;
      }

      socket.onopen = () => {
        attempts = 0;
        setLinkStatus("ONLINE");
      };

      socket.onmessage = (event) => {
        try {
          ingest(JSON.parse(event.data as string));
        } catch {
          ingest({ source: "RAW", payload: String(event.data) });
        }
      };

      socket.onerror = () => setLinkStatus("OFFLINE");
      socket.onclose = () => {
        if (cancelled) return;
        setLinkStatus("OFFLINE");
        attempts += 1;
        reconnectTimer = setTimeout(connect, Math.min(1000 * 2 ** attempts, 10000));
      };
    };

    connect();

    return () => {
      cancelled = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      if (socket) {
        socket.onclose = null;
        socket.close();
      }
    };
  }, [ingest]);

  useEffect(() => {
    const tick = () =>
      setClock(
        new Date().toLocaleTimeString("en-GB", {
          timeZone: "UTC",
          hour12: false,
        }) + " UTC",
      );
    tick();
    const timer = setInterval(tick, 1000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  const handleUpload = useCallback(async () => {
    if (!selectedFile || uploadStatus === "UPLOADING") return;

    setUploadStatus("UPLOADING");
    setUploadResult(null);
    setUploadError("");
    setUploadLatency(null);
    const startedAt = performance.now();

    try {
      const formData = new FormData();
      formData.append("file", selectedFile, selectedFile.name);

      const response = await fetch(ANALYZE_URL, {
        method: "POST",
        body: formData,
      });

      setUploadLatency(Math.round(performance.now() - startedAt));

      if (!response.ok) {
        throw new Error(`Server responded ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      setUploadResult(data as FileUploadResult);
      setUploadStatus("SUCCESS");
    } catch (error) {
      setUploadError(
        error instanceof Error
          ? `TRANSMISSION FAULT :: ${error.message}\nCheck that the analysis service is reachable at ${ANALYZE_URL}`
          : "TRANSMISSION FAULT :: unknown error",
      );
      setUploadStatus("ERROR");
    }
  }, [selectedFile, uploadStatus]);

  const onFileSelect = (event: ChangeEvent<HTMLInputElement>) => {
    setSelectedFile(event.target.files?.[0] ?? null);
    setUploadStatus("IDLE");
    setUploadResult(null);
    setUploadError("");
  };

  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);
    const file = event.dataTransfer.files?.[0];
    if (file) {
      setSelectedFile(file);
      setUploadStatus("IDLE");
      setUploadResult(null);
      setUploadError("");
    }
  };

  const metricCards = useMemo(
    () => [
      {
        label: "TRUST SCORE",
        unit: "%",
        value: metrics.trust,
        delta: deltas.trust,
        accent: "#00e5ff",
        sub: "Composite authenticity index",
      },
      {
        label: "ACOUSTIC AI SIGNATURE",
        unit: "%",
        value: metrics.acoustic,
        delta: deltas.acoustic,
        accent: "#a855f7",
        sub: "Spectral synthesis detection",
      },
      {
        label: "TF-IDF SIMILARITY",
        unit: "%",
        value: metrics.tfidf,
        delta: deltas.tfidf,
        accent: "#39ff88",
        sub: "Script vector divergence",
      },
      {
        label: "FLAJOLET-MARTIN IPs",
        unit: "",
        value: metrics.ips,
        delta: deltas.ips,
        accent: "#ffb020",
        sub: "Distinct source estimates",
      },
    ],
    [metrics, deltas],
  );

  const statusPill =
    linkStatus === "ONLINE"
      ? { text: "LINK ONLINE", color: "#39ff88" }
      : linkStatus === "CONNECTING"
        ? { text: "HANDSHAKE…", color: "#ffb020" }
        : { text: "LINK OFFLINE", color: "#ff2d6f" };

  const criticalCount = messages.filter((m) => m.severity === "CRITICAL").length;

  return (
    <main className="sg-root min-h-screen w-full overflow-x-hidden px-4 py-6 text-[#c9f2ff] md:px-8">
      <style jsx global>{`
        @import url("https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Share+Tech+Mono&display=swap");
        .sg-root {
          background:
            radial-gradient(1200px 600px at 20% -10%, rgba(0, 229, 255, 0.08), transparent 60%),
            radial-gradient(900px 500px at 90% 110%, rgba(168, 85, 247, 0.09), transparent 60%),
            linear-gradient(rgba(0, 229, 255, 0.035) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 229, 255, 0.035) 1px, transparent 1px),
            #03060b;
          background-size: auto, auto, 42px 42px, 42px 42px, auto;
          font-family: "Share Tech Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
        }
        .sg-display { font-family: "Orbitron", "Share Tech Mono", monospace; }
        .sg-glow-cyan { text-shadow: 0 0 8px rgba(0, 229, 255, 0.75), 0 0 26px rgba(0, 229, 255, 0.35); }
        .sg-blink { animation: sg-blink 1.1s steps(2, start) infinite; }
        @keyframes sg-blink { to { visibility: hidden; } }
        .sg-pulse { animation: sg-pulse 1.8s ease-in-out infinite; }
        @keyframes sg-pulse { 0%, 100% { opacity: 1; box-shadow: 0 0 10px currentColor; } 50% { opacity: 0.55; box-shadow: 0 0 22px currentColor; } }
        
        .waveform-bar {
          width: 4px;
          background: #00e5ff;
          border-radius: 2px;
          animation: wave 1s ease-in-out infinite alternate;
        }
        .waveform-bar:nth-child(1) { animation-delay: 0.0s; }
        .waveform-bar:nth-child(2) { animation-delay: 0.2s; }
        .waveform-bar:nth-child(3) { animation-delay: 0.4s; }
        .waveform-bar:nth-child(4) { animation-delay: 0.1s; }
        .waveform-bar:nth-child(5) { animation-delay: 0.5s; }
        .waveform-bar:nth-child(6) { animation-delay: 0.3s; }
        .waveform-bar:nth-child(7) { animation-delay: 0.6s; }
        .waveform-bar:nth-child(8) { animation-delay: 0.2s; }
        @keyframes wave {
          0% { height: 10px; opacity: 0.4; }
          100% { height: 40px; opacity: 1; box-shadow: 0 0 10px #00e5ff; }
        }
        .waveform-offline .waveform-bar {
          animation: none;
          height: 10px;
          background: #ff2d6f;
          opacity: 0.3;
        }
      `}</style>

      {/* ---------- Header ---------- */}
      <header className="mb-6 flex flex-wrap items-center justify-between gap-4 border-b border-cyan-400/20 pb-4">
        <div className="flex items-center gap-4">
          <div className="relative grid h-11 w-11 place-items-center border border-cyan-400/50 bg-cyan-400/5">
            <Corners />
            <span className="sg-display sg-glow-cyan text-lg font-black text-cyan-300">SG</span>
          </div>
          <div>
            <h1 className="sg-display sg-glow-cyan text-2xl font-black tracking-[0.18em] text-cyan-100 md:text-3xl">
              STREAMGUARD
            </h1>
            <p className="text-[11px] tracking-[0.3em] text-cyan-500/80">
              DEEPFAKE INTERCEPTION GRID // NODE v2.4.1
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3 text-xs">
          <span className="hidden text-cyan-500/80 md:inline">{clock}</span>
          <span
            className="sg-pulse inline-block h-2.5 w-2.5 rounded-full"
            style={{ color: statusPill.color, backgroundColor: statusPill.color }}
          />
          <span
            className="border px-3 py-1.5 tracking-[0.2em]"
            style={{
              color: statusPill.color,
              borderColor: `${statusPill.color}55`,
              backgroundColor: `${statusPill.color}0d`,
            }}
          >
            {statusPill.text}
          </span>
        </div>
      </header>

      {/* ---------- Metric grid ---------- */}
      <section className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {metricCards.map((card) => {
          const display = card.value === null ? "──" : String(card.value);
          const deltaText =
            card.value === null || card.delta === 0
              ? "STABLE"
              : `${card.delta > 0 ? "▲" : "▼"} ${Math.abs(Math.round(card.delta * 10) / 10)}${card.unit}`;
          return (
            <div key={card.label} className="relative overflow-hidden border border-cyan-400/20 bg-[#060b14]/85 p-5 backdrop-blur-sm transition-colors duration-300 hover:border-cyan-400/50">
              <Corners />
              <div className="absolute inset-x-0 top-0 h-px" style={{ background: `linear-gradient(90deg, transparent, ${card.accent}, transparent)` }} />
              <p className="text-[10px] tracking-[0.28em]" style={{ color: card.accent }}>{card.label}</p>
              <div className="mt-3 flex items-baseline gap-2">
                <span className="sg-display text-4xl font-bold leading-none md:text-5xl" style={{ color: card.accent, textShadow: `0 0 14px ${card.accent}88` }}>
                  {display}
                  {card.value !== null && card.unit && <span className="ml-1 text-lg opacity-70">{card.unit}</span>}
                </span>
              </div>
              <div className="mt-3 flex items-center justify-between text-[10px]">
                <span className="text-cyan-600/70">{card.sub}</span>
                <span className={card.delta > 0 ? "text-[#39ff88]" : card.delta < 0 ? "text-[#ff2d6f]" : "text-cyan-700"}>{deltaText}</span>
              </div>
              <div className="mt-3 h-1 w-full overflow-hidden rounded-full bg-cyan-400/10">
                <div className="h-full rounded-full transition-all duration-700" style={{ width: card.value === null ? "6%" : `${Math.min(Math.max(card.value, 2), 100)}%`, background: `linear-gradient(90deg, ${card.accent}44, ${card.accent})` }} />
              </div>
            </div>
          );
        })}
      </section>

      {/* ---------- Visualizers: Chart & Waveform ---------- */}
      <section className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-3">
        <div className="relative flex h-[18rem] flex-col overflow-hidden border border-cyan-400/20 bg-[#060b14]/85 backdrop-blur-sm lg:col-span-2">
          <Corners />
          <div className="border-b border-cyan-400/20 px-4 py-2">
            <h2 className="sg-display text-sm font-bold tracking-[0.25em] text-cyan-300">▸ THREAT TIMELINE (REAL-TIME)</h2>
          </div>
          <div className="flex-1 p-4 pb-0">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={messages} margin={{ top: 5, right: 0, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#00e5ff15" vertical={false} />
                <XAxis dataKey="timestamp" stroke="#00e5ff55" fontSize={10} tickMargin={10} />
                <YAxis stroke="#00e5ff55" fontSize={10} domain={[0, 100]} />
                <Tooltip 
                  contentStyle={{ backgroundColor: "#03060b", border: "1px solid #00e5ff55", borderRadius: 0, color: "#00e5ff" }}
                  itemStyle={{ fontSize: "12px", fontFamily: "monospace" }}
                  labelStyle={{ display: "none" }}
                />
                <Line type="monotone" dataKey="trust" name="Trust Score" stroke="#00e5ff" strokeWidth={2} dot={false} isAnimationActive={false} />
                <Line type="monotone" dataKey="acoustic" name="AI Signature" stroke="#a855f7" strokeWidth={2} dot={false} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="relative flex h-[18rem] flex-col overflow-hidden border border-cyan-400/20 bg-[#060b14]/85 backdrop-blur-sm">
          <Corners />
          <div className="border-b border-cyan-400/20 px-4 py-2">
            <h2 className="sg-display text-sm font-bold tracking-[0.25em] text-cyan-300">▸ ACOUSTIC SENSOR</h2>
          </div>
          <div className="flex flex-1 items-center justify-center p-4">
            <div className={`flex h-[60px] items-center gap-2 ${linkStatus === "ONLINE" ? "" : "waveform-offline"}`}>
              {[...Array(8)].map((_, i) => <div key={i} className="waveform-bar"></div>)}
            </div>
          </div>
          <div className="border-t border-cyan-400/10 p-3 text-center text-[10px] tracking-widest text-cyan-500/70">
            {linkStatus === "ONLINE" ? "CAPTURING HARDWARE MIC STREAM" : "SENSOR DISCONNECTED"}
          </div>
        </div>
      </section>

      {/* ---------- Live log + upload portal ---------- */}
      <section className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-3">
        <div className="relative flex h-[24rem] flex-col overflow-hidden border border-cyan-400/20 bg-[#060b14]/85 backdrop-blur-sm lg:col-span-2">
          <Corners />
          <div className="flex items-center justify-between border-b border-cyan-400/20 px-4 py-3">
            <h2 className="sg-display text-sm font-bold tracking-[0.25em] text-cyan-300">▸ LIVE INTERCEPT LOG</h2>
            <div className="flex items-center gap-4 text-[10px] tracking-widest">
              <span className="text-[#ff2d6f]">CRITICAL: {criticalCount}</span>
              <span className="text-cyan-500/80">FRAMES: {messages.length}</span>
            </div>
          </div>
          <div className="relative flex-1 overflow-y-auto p-4 text-xs leading-relaxed" style={{ scrollbarWidth: "thin", scrollbarColor: "rgba(0, 229, 255, 0.45) transparent" }}>
            {messages.length === 0 ? (
              <p className="text-cyan-700"><span className="sg-blink">█</span> awaiting intercept stream @ {WS_URL} …</p>
            ) : (
              <ul className="space-y-2">
                {messages.map((m) => (
                  <li key={m.id} className="border-l-2 pl-3" style={{ borderColor: m.severity === "CRITICAL" ? "#ff2d6f" : m.severity === "WARNING" ? "#ffb020" : "#00e5ff" }}>
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="text-cyan-600">[{m.timestamp}]</span>
                      <span className={`border px-1.5 py-px text-[9px] tracking-widest ${SEVERITY_STYLES[m.severity]}`}>{m.severity}</span>
                      <span className="text-cyan-400">{m.source}</span>
                      <span className="text-cyan-800">
                        trust={m.trust ?? "–"} acoustic={m.acoustic ?? "–"} tfidf={m.tfidf ?? "–"} ips={m.ips ?? "–"}
                      </span>
                    </div>
                    <pre className="mt-0.5 max-w-full truncate text-[10px] text-cyan-800/80">{JSON.stringify(m.raw)}</pre>
                  </li>
                ))}
              </ul>
            )}
            <div ref={logEndRef} />
          </div>
        </div>

        <div className="relative flex h-[24rem] flex-col overflow-hidden border border-fuchsia-400/20 bg-[#060b14]/85 backdrop-blur-sm">
          <Corners />
          <div className="border-b border-fuchsia-400/20 px-4 py-3">
            <h2 className="sg-display text-sm font-bold tracking-[0.25em] text-fuchsia-300">▸ RECRUITER BATCH ANALYSIS</h2>
          </div>
          <div className="flex flex-1 flex-col gap-4 p-4">
            <div
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`flex flex-1 cursor-pointer flex-col items-center justify-center gap-2 border border-dashed p-4 text-center transition-colors ${dragging ? "border-fuchsia-400 bg-fuchsia-400/10" : "border-fuchsia-400/30 hover:border-fuchsia-400/70 hover:bg-fuchsia-400/5"}`}
            >
              <input ref={fileInputRef} type="file" className="hidden" onChange={onFileSelect} />
              <span className="text-3xl text-fuchsia-400">⇪</span>
              <p className="text-xs tracking-widest text-fuchsia-200/90">{selectedFile ? selectedFile.name : "DROP CANDIDATE MEDIA FILE"}</p>
            </div>
            <button
              onClick={handleUpload}
              disabled={!selectedFile || uploadStatus === "UPLOADING"}
              className="sg-display w-full border border-fuchsia-400/60 bg-fuchsia-500/15 py-3 text-xs font-bold tracking-[0.3em] text-fuchsia-200 transition-all hover:bg-fuchsia-500/30 hover:shadow-[0_0_24px_rgba(217,70,239,0.45)] disabled:cursor-not-allowed disabled:opacity-40"
            >
              {uploadStatus === "UPLOADING" ? "TRANSMITTING…" : "INITIATE ANALYSIS"}
            </button>
            
            {/* Parsed Result Console */}
            <div className="flex-1 overflow-y-auto border border-fuchsia-400/20 bg-black/50 p-4 text-[10px] leading-relaxed" style={{ scrollbarWidth: "thin", scrollbarColor: "rgba(217, 70, 239, 0.45) transparent" }}>
              {uploadStatus === "IDLE" && !uploadResult && <p className="text-fuchsia-700"><span className="sg-blink">█</span> result console standby…</p>}
              {uploadStatus === "UPLOADING" && <p className="sg-blink text-fuchsia-300">▸ uploading FormData → {ANALYZE_URL}</p>}
              
              {uploadStatus === "SUCCESS" && uploadResult && (
                <div className="flex flex-col gap-3">
                  <div className="flex items-center justify-between border-b border-fuchsia-400/30 pb-2">
                    <span className="text-fuchsia-200 tracking-wider">TARGET: <br/><span className="font-bold">{uploadResult.filename}</span></span>
                    <span className={`px-2 py-1 font-bold ${uploadResult.verdict.includes("AI") ? "bg-[#ff2d6f]/20 text-[#ff2d6f] border border-[#ff2d6f]/50" : "bg-[#39ff88]/20 text-[#39ff88] border border-[#39ff88]/50"}`}>
                      {uploadResult.verdict}
                    </span>
                  </div>
                  
                  <div className="flex flex-col gap-1">
                    <div className="flex justify-between text-fuchsia-300"><span>ACOUSTIC RISK</span> <span>{uploadResult.acoustic_fake_percent}%</span></div>
                    <div className="h-1.5 w-full bg-fuchsia-900/40 rounded-full overflow-hidden">
                      <div className="h-full bg-gradient-to-r from-fuchsia-500 to-[#ff2d6f]" style={{ width: `${uploadResult.acoustic_fake_percent}%` }}></div>
                    </div>
                  </div>

                  <div className="flex flex-col gap-1">
                    <div className="flex justify-between text-fuchsia-300"><span>TEXT OVERLAP</span> <span>{uploadResult.text_similarity}%</span></div>
                    <div className="h-1.5 w-full bg-fuchsia-900/40 rounded-full overflow-hidden">
                      <div className="h-full bg-gradient-to-r from-fuchsia-500 to-[#ffb020]" style={{ width: `${uploadResult.text_similarity}%` }}></div>
                    </div>
                  </div>

                  <div className="mt-2 border-t border-fuchsia-400/20 pt-2 text-fuchsia-200/80">
                    <p className="mb-1 text-fuchsia-400">▸ BLACKLIST HITS: {uploadResult.blacklisted_hits}</p>
                    <p className="italic bg-fuchsia-400/5 p-2 border-l-2 border-fuchsia-400/30">"{uploadResult.transcript_preview}"</p>
                  </div>
                </div>
              )}
              {uploadStatus === "ERROR" && <pre className="whitespace-pre-wrap break-all text-[#ff2d6f]">{uploadError}</pre>}
            </div>
          </div>
        </div>
      </section>

      {/* ---------- Footer ---------- */}
      <footer className="mt-4 flex flex-wrap items-center justify-between gap-2 border-t border-cyan-400/15 pt-3 text-[10px] tracking-[0.25em] text-cyan-700">
        <span>STREAMGUARD © 2026 // UNIVERSITY RESEARCH BUILD</span>
        <span>WS: {WS_URL.replace("ws://", "")} · POST: {ANALYZE_URL.replace("http://", "")}</span>
      </footer>
    </main>
  );
}