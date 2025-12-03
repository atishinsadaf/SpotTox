import { useState, useEffect, useMemo } from "react";
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
      "Safer online spaces start here.",
      "One click to know: Is it toxic?",
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

  // Message and thread counts
  const [messageCount, setMessageCount] = useState(null);
  const [threadCount, setThreadCount] = useState(null);

  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressMessage, setProgressMessage] = useState("");
  const [multiResults, setMultiResults] = useState(null);

  const [showChart, setShowChart] = useState(false);
  const [binDetail, setBinDetail] = useState(null);

  // Quick result state for "Is It Toxic?" button
  const [quickResult, setQuickResult] = useState(null);

  const showMultiModal = showMulti && multiResults;

  // Thread search state
  const [showThreadSearch, setShowThreadSearch] = useState(false);
  const [searchThread, setSearchThread] = useState("");
  const [threadResults, setThreadResults] = useState([]);

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

  // Upload single file
  function handleUpload(e) {
    const f = e.target.files[0];
    if (!f) return;
    setFile(f);
    setUploadedName(f.name);
    setMultiFiles([]);
    clearResults();
  }

  // Handle image/screenshot upload with OCR
  async function handleImageUpload(e) {
    const f = e.target.files[0];
    if (!f) return;
    
    setFile(f);
    setUploadedName(f.name);
    setMultiFiles([]);
    clearResults();
    setIsAnalyzing(true);
    setProgress(0);
    setProgressMessage("Uploading image...");

    try {
      // Upload the image
      const form = new FormData();
      form.append("file", f);
      setProgress(20);

      const upRes = await fetch(`${BASE_URL}/upload-image`, {
        method: "POST",
        body: form,
      });

      const upJson = await upRes.json();
      if (!upRes.ok) throw new Error(upJson.error || "Upload failed");

      setProgress(40);
      setProgressMessage("Extracting text from image (OCR)...");

      // Analyze the image
      const analyzeRes = await fetch(`${BASE_URL}/analyze-image`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          filename: upJson.filename,
          model: selectedModel,
        }),
      });

      const analyzeJson = await analyzeRes.json();
      
      if (!analyzeRes.ok) {
        // Check if it's an OCR not available error
        if (analyzeJson.install_instructions) {
          alert(`⚠️ ${analyzeJson.error}\n\n${analyzeJson.install_instructions}`);
        } else {
          alert(`Analysis failed: ${analyzeJson.error}`);
        }
        throw new Error(analyzeJson.error);
      }

      setProgress(80);
      setProgressMessage("Processing results...");

      // Add to recent
      setRecent((prev) => [
        { name: f.name, time: new Date().toLocaleString() },
        ...prev.slice(0, 8),
      ]);

      // Set the results
      setSummary({
        mean: analyzeJson.mean,
        p90: analyzeJson.p90,
        p95: analyzeJson.p95,
        max: analyzeJson.max,
      });
      setHistogram(makeHistogram(analyzeJson.histogram));
      setTopFlagged(analyzeJson.top || []);
      setMessageCount(analyzeJson.message_count);
      setThreadCount(analyzeJson.thread_count);

      setProgress(100);
      setProgressMessage("Analysis complete!");

      // Show quick result
      setQuickResult({
        mean: analyzeJson.mean,
        messageCount: analyzeJson.message_count,
        threadCount: 1,
      });

    } catch (err) {
      console.error("Image analysis error:", err);
      setProgressMessage("Analysis failed");
    } finally {
      setTimeout(() => {
        setIsAnalyzing(false);
        setProgressMessage("");
      }, 500);
    }
  }

  // Upload multiple files
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
    setQuickResult(null);
    setMessageCount(null);
    setThreadCount(null);
  }

  // Quick check handler, simple "Is It Toxic?" answer
  async function handleQuickCheck() {
    if (!file) return;

    try {
      // First upload the file
      const form = new FormData();
      form.append("file", file);

      const upRes = await fetch(`${BASE_URL}/upload`, {
        method: "POST",
        body: form,
      });

      const upJson = await upRes.json();
      if (!upRes.ok) throw new Error(upJson.error || "Upload failed");

      const filename = upJson.filename;
      setUploadedName(filename);

      // Add to recent
      setRecent((prev) => [
        { name: filename, time: new Date().toLocaleString() },
        ...prev.slice(0, 8),
      ]);

      // Analyze the file - try "text" first, then "txt"
      let textCol = "text";
      let res = await fetch(`${BASE_URL}/analyze-file`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          filename,
          text_col: textCol,
          model: selectedModel,
        }),
      });

      let json = await res.json();

      // If "text" column failed, try "txt"
      if (!res.ok && json.error?.includes("missing")) {
        textCol = "txt";
        res = await fetch(`${BASE_URL}/analyze-file`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            filename,
            text_col: textCol,
            model: selectedModel,
          }),
        });
        json = await res.json();
        if (!res.ok) throw new Error(json.error || "Analysis failed");
      }

      if (!res.ok) throw new Error(json.error || "Analysis failed");

      // Process results for quick check
      let mean, msgCount;
      
      if (Array.isArray(json.scores) && typeof json.scores[0] === "object") {
        // LSTM case
        const toxicityScores = json.scores.map((s) => s.toxicity ?? 0);
        mean = toxicityScores.reduce((a, b) => a + b, 0) / toxicityScores.length;
        msgCount = toxicityScores.length;
      } else {
        // BERT/RoBERTa case
        mean = json.mean;
        msgCount = json.histogram?.length || 0;
      }

      // Store full results for later if user wants to see charts
      processAnalysisResults(json);

      // Set quick result
      setQuickResult({
        mean: mean,
        messageCount: msgCount,
        threadCount: 1,
      });

    } catch (err) {
      alert("Analysis failed: " + err.message);
    }
  }

  // Analyze single thread (full analysis with charts)
  async function handleAnalyze() {
    if (!file) return;
    setIsAnalyzing(true);
    setProgress(0);
    setProgressMessage("Uploading file...");

    try {
      const form = new FormData();
      form.append("file", file);
      setProgress(10);

      const upRes = await fetch(`${BASE_URL}/upload`, {
        method: "POST",
        body: form,
      });
      const upJson = await upRes.json();
      if (!upRes.ok) throw new Error(upJson.error || "Upload failed");

      setProgress(20);
      setProgressMessage("File uploaded successfully!");
      const filename = upJson.filename;

      setUploadedName(filename);
      setRecent((prev) => [
        { name: filename, time: new Date().toLocaleString() },
        ...prev.slice(0, 8),
      ]);

      setProgressMessage("Tokenizing text data...");
      setProgress(40);
      await new Promise((r) => setTimeout(r, 500));

      setProgressMessage(`Running ${selectedModel} model...`);
      setProgress(60);
      await analyzeFilename(filename);

      setProgressMessage("Generating visualizations...");
      setProgress(90);
      await new Promise((r) => setTimeout(r, 400));

      setProgress(100);
      setProgressMessage("Analysis complete!");
      await new Promise((r) => setTimeout(r, 400));

      setShowChart(true);
    } catch (err) {
      alert(err.message);
      setProgressMessage("Analysis failed!");
    } finally {
      setTimeout(() => {
        setIsAnalyzing(false);
        setProgressMessage("");
      }, 500);
    }
  }

  // Core analyze call
  async function analyzeFilename(filename) {
    // Try "text" first, then "txt"
    let textCol = "text";
    
    const res = await fetch(`${BASE_URL}/analyze-file`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename, text_col: textCol, model: selectedModel }),
    });

    const json = await res.json();
    
    // If "text" column failed, try "txt"
    if (!res.ok && json.error?.includes("missing")) {
      textCol = "txt";
      const res2 = await fetch(`${BASE_URL}/analyze-file`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename, text_col: textCol, model: selectedModel }),
      });
      const json2 = await res2.json();
      if (!res2.ok) throw new Error(json2.error || "Analysis failed");
      
      processAnalysisResults(json2);
      return;
    }
    
    if (!res.ok) throw new Error(json.error || "Analysis failed");
    processAnalysisResults(json);
  }

  // Process analysis results (handles both LSTM and BERT/RoBERTa)
  function processAnalysisResults(json) {
    // LSTM returns full label dict - extract toxicity list
    if (Array.isArray(json.scores) && typeof json.scores[0] === "object") {
      const toxicityScores = json.scores.map((s) => s.toxicity ?? 0);

      const mean = toxicityScores.reduce((a, b) => a + b, 0) / toxicityScores.length;
      const sorted = [...toxicityScores].sort((a, b) => a - b);
      const p90 = sorted[Math.floor(sorted.length * 0.9)];
      const p95 = sorted[Math.floor(sorted.length * 0.95)];
      const max = Math.max(...toxicityScores);

      setSummary({ mean, p90, p95, max });
      setHistogram(makeHistogram(toxicityScores));
      setTopFlagged(json.top || []);
      setMessageCount(toxicityScores.length);
      setProgress(100);
      return;
    }

    // Normal BERT / RoBERTa case
    setSummary({
      mean: json.mean,
      p90: json.p90,
      p95: json.p95,
      max: json.max,
    });

    setHistogram(makeHistogram(json.histogram));
    setTopFlagged(json.top || []);
    
    // Set message count from histogram length
    if (json.histogram) {
      setMessageCount(json.histogram.length);
    }
    setProgress(100);
  }

  // Analyze multiple threads
  async function analyzeMultipleThreads() {
    setIsAnalyzing(true);
    setProgress(0);
    setProgressMessage("Starting batch analysis...");

    const total = multiFiles.length;
    let current = 0;

    function stepProgress() {
      const target = Math.round(((current + 0.5) / total) * 100);
      setProgress((p) => (p < target ? target : p));
    }

    try {
      const results = [];

      for (const file of multiFiles) {
        setProgressMessage(`Analyzing ${current + 1}/${total}: ${file}`);

        // Try "text" first, then "txt"
        let res = await fetch(`${BASE_URL}/analyze-file`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ filename: file, text_col: "text", model: selectedModel }),
        });

        let json = await res.json();
        
        if (!res.ok && json.error?.includes("missing")) {
          res = await fetch(`${BASE_URL}/analyze-file`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ filename: file, text_col: "txt", model: selectedModel }),
          });
          json = await res.json();
        }
        
        if (!res.ok) throw new Error(json.error || "Analysis failed");

        results.push({ filename: file, summary: { mean: json.mean } });

        current++;
        stepProgress();
        await new Promise((r) => setTimeout(r, 200));
      }

      setProgress(100);
      setProgressMessage("Batch analysis complete!");

      setMultiResults(results);
      setShowMulti(true);
      setThreadCount(results.length);
    } catch (err) {
      alert("Multi-analysis failed: " + err.message);
      setProgressMessage("Batch analysis failed!");
    } finally {
      setTimeout(() => {
        setIsAnalyzing(false);
        setProgressMessage("");
      }, 500);
    }
  }

  // Thread search handler
  async function handleThreadSearch() {
    if (!searchThread.trim()) {
      alert("Enter a thread ID.");
      return;
    }

    try {
      const res = await fetchJson(`${BASE_URL}/search_thread`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ thread_id: searchThread }),
      });

      if (res.found) {
        setThreadResults(res.rows);
      } else {
        setThreadResults([]);
        alert("Thread ID not found.");
      }
    } catch (err) {
      alert("Search failed: " + err.message);
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
        handleImageUpload={handleImageUpload}
        handleAnalyze={handleAnalyze}
        handleQuickCheck={handleQuickCheck}
        analyzeMultipleThreads={analyzeMultipleThreads}
        isAnalyzing={isAnalyzing}
        progress={progress}
        progressMessage={progressMessage}
        summary={summary}
        histogram={histogram}
        topFlagged={topFlagged}
        messageCount={messageCount}
        threadCount={threadCount}
        quickResult={quickResult}
        setQuickResult={setQuickResult}
        showChart={showChart}
        setShowChart={setShowChart}
        binDetail={binDetail}
        setBinDetail={setBinDetail}
        models={models}
        selectedModel={selectedModel}
        setSelectedModel={handleModelChange}
        modelName={selectedModel}

        // TEAMMATE'S: Thread search props
        searchThread={searchThread}
        setSearchThread={setSearchThread}
        handleThreadSearch={handleThreadSearch}
        threadResults={threadResults}

        // Reddit/histogram helpers
        makeHistogram={makeHistogram}
        setSummary={setSummary}
        setHistogram={setHistogram}
        setTopFlagged={setTopFlagged}
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