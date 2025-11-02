import { useState, useEffect, useMemo } from "react";
import { Histogram, ChartModal } from "./Chart";
import PieChart from "./PieChart";
import MainUI from "./MainUI";

const BASE_URL = "http://127.0.0.1:5001";

// Helper to request JSON
async function fetchJson(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`HTTP ${res.status} — ${text || res.statusText}`);
  }
  return await res.json();
}

// Build histogram bins from toxicity scores
function makeHistogram(scores, bins = 10) {
  if (!scores?.length) return null;

  const counts = Array(bins).fill(0);
  const edges = Array.from({ length: bins + 1 }, (_, i) => i / bins);

  scores.forEach((s) => {
    const idx = Math.min(bins - 1, Math.floor(s * bins));
    counts[idx]++;
  });

  return { counts, edges };
}

export default function App() {
  // Time & fancy UI state
  const [now, setNow] = useState(() => new Date().toLocaleString());
  const [typed, setTyped] = useState("");
  const [tagIndex, setTagIndex] = useState(0);
  const [isDeleting, setIsDeleting] = useState(false);

  const taglines = useMemo(
    () => [
      "AI-powered early warning detection for toxic conversations.",
      "Predict toxicity before it spreads.",
      "Safer online spaces.",
    ],
    []
  );

  // File / model state
  const [file, setFile] = useState(null);
  const [uploadedName, setUploadedName] = useState("");
  const [multiFiles, setMultiFiles] = useState([]);
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState("");

  // UI state
  const [recent, setRecent] = useState([]);
  const [showRecent, setShowRecent] = useState(false);
  const [showMulti, setShowMulti] = useState(false);

  const [summary, setSummary] = useState(null);
  const [histogram, setHistogram] = useState(null);
  const [topFlagged, setTopFlagged] = useState([]);

  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [multiResults, setMultiResults] = useState(null);

  const [showChart, setShowChart] = useState(false);
  const [binDetail, setBinDetail] = useState(null);

  const showMultiModal = showMulti && multiResults;

  // Live clock updates
  useEffect(() => {
    if (showChart) return;
    const t = setInterval(() => setNow(new Date().toLocaleString()), 1000);
    return () => clearInterval(t);
  }, [showChart]);

  // Typing animation
  useEffect(() => {
    if (showChart) return;
    const full = taglines[tagIndex];
    const speed = isDeleting ? 35 : 55;
    const pause = 1000;

    const tick = setTimeout(() => {
      if (!isDeleting) {
        const next = full.slice(0, typed.length + 1);
        setTyped(next);
        if (next === full) setTimeout(() => setIsDeleting(true), pause);
      } else {
        const next = full.slice(0, typed.length - 1);
        setTyped(next);
        if (!next.length) {
          setIsDeleting(false);
          setTagIndex((i) => (i + 1) % taglines.length);
        }
      }
    }, speed);

    return () => clearTimeout(tick);
  }, [typed, isDeleting, tagIndex, taglines, showChart]);

  // Load available models
  useEffect(() => {
    fetchJson(`${BASE_URL}/models`)
      .then((data) => {
        if (data.available?.length) {
          setModels(data.available);
          const defaultModel = data.active || data.available[0];
          setSelectedModel(defaultModel);
          fetch(`${BASE_URL}/set_model`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ model: defaultModel }),
          });
        }
      })
      .catch((e) => alert(e.message));
  }, []);

  // Switch models
  async function handleModelChange(newModel) {
    setSelectedModel(newModel);
    await fetch(`${BASE_URL}/set_model`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: newModel }),
    });
  }

  // Upload
  function handleUpload(e) {
    const f = e.target.files[0];
    if (!f) return;
    setFile(f);
    setUploadedName(f.name);
    setMultiFiles([]);
    clearResults();
  }

  async function handleMultiUpload(e) {
    const files = Array.from(e.target.files);
    if (!files.length) return;

    const form = new FormData();
    files.forEach((f) => form.append("files", f));
    const res = await fetch(`${BASE_URL}/upload-multiple`, {
      method: "POST",
      body: form,
    });
    const json = await res.json();
    if (json.uploaded) {
      setFile(null);
      setUploadedName("");
      clearResults();
      setMultiFiles((prev) => [...prev, ...json.uploaded.map((u) => u.filename)]);
    }
  }

  function clearResults() {
    setSummary(null);
    setHistogram(null);
    setTopFlagged([]);
    setShowChart(false);
  }

  // Analyze single thread
  async function handleAnalyze() {
    if (!file) return;
    setIsAnalyzing(true);
    setProgress(0);

    const interval = setInterval(
      () => setProgress((p) => Math.min(90, Math.round(p + Math.random() * 18 + 7))),
      350
    );

    try {
      const form = new FormData();
      form.append("file", file);
      const upRes = await fetch(`${BASE_URL}/upload`, { method: "POST", body: form });
      const upJson = await upRes.json();
      if (!upRes.ok) throw new Error(upJson.error || "Upload failed");
      const filename = upJson.filename;

      setUploadedName(filename);
      setRecent((prev) => [
        { name: filename, time: new Date().toLocaleString() },
        ...prev.slice(0, 8),
      ]);

      await analyzeFilename(filename);
      setShowChart(true);
    } catch (err) {
      alert(err.message);
    } finally {
      clearInterval(interval);
      setTimeout(() => setIsAnalyzing(false), 350);
    }
  }

  // Core analyze call
  async function analyzeFilename(filename) {
    const res = await fetch(`${BASE_URL}/analyze-file`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename, text_col: "txt", model: selectedModel }),
    });

    const json = await res.json();
    if (!res.ok) throw new Error(json.error || "Analysis failed");

    setSummary({
      mean: json.mean,
      p90: json.p90,
      p95: json.p95,
      max: json.max
    });

    // Build histogram from backend scores
    setHistogram(makeHistogram(json.histogram));
    setTopFlagged(json.top || []);
    setProgress(100);
  }

  // Analyze multiple threads
  async function analyzeMultipleThreads() {
    setIsAnalyzing(true);
    setProgress(0);

    try {
      const res = await fetch(`${BASE_URL}/analyze-multiple`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filenames: multiFiles, model: selectedModel }),
      });

      const json = await res.json();
      setIsAnalyzing(false);

      setMultiResults(
        (json.results || []).map((r) => ({
          filename: r.file,
          summary: { mean: r.mean },
        }))
      );

      setShowMulti(true);
    } catch (err) {
      setIsAnalyzing(false);
      alert("Multi-analysis failed: " + err.message);
    }
  }

  return (
    <>
      <MainUI
        now={now}
        typed={typed}
        recent={recent}
        showRecent={showRecent}
        setShowRecent={setShowRecent}
        analyzeFilename={analyzeFilename}
        file={file}
        uploadedName={uploadedName}
        multiFiles={multiFiles}
        handleUpload={handleUpload}
        handleMultiUpload={handleMultiUpload}
        handleAnalyze={handleAnalyze}
        analyzeMultipleThreads={analyzeMultipleThreads}
        isAnalyzing={isAnalyzing}
        progress={progress}
        summary={summary}
        histogram={histogram}
        topFlagged={topFlagged}
        showChart={showChart}
        setShowChart={setShowChart}
        binDetail={binDetail}
        setBinDetail={setBinDetail}
        models={models}
        selectedModel={selectedModel}
        setSelectedModel={handleModelChange}
        modelName={selectedModel}
      />

      {showMultiModal && (
        <PieChart
          results={multiResults}
          onClose={() => {
            setShowMulti(false);
            setMultiResults(null);
          }}
          modelName={selectedModel}
        />
      )}
    </>
  );
}
