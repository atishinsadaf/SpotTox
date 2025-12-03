import React, { useState } from "react";
import { Histogram, ChartModal } from "./Chart";
import ChatUI from "./ChatUI";

export default function MainUI({
  now,
  typed,

  // recent uploads
  recent,
  showRecent,
  setShowRecent,
  analyzeFilename,

  // uploads
  file,
  uploadedName,
  multiFiles,
  handleUpload,
  handleMultiUpload,
  handleImageUpload,

  // analysis
  handleAnalyze,
  analyzeMultipleThreads,
  handleQuickCheck,

  // state
  isAnalyzing,
  progress,
  progressMessage,
  summary,
  histogram,
  topFlagged,
  messageCount,
  threadCount,

  // quick check result
  quickResult,
  setQuickResult,

  // chart
  showChart,
  setShowChart,
  binDetail,
  setBinDetail,

  // models
  models,
  selectedModel,
  setSelectedModel,
  modelName,

  // REDDIT NEW PROPS
  makeHistogram,
  setSummary,
  setHistogram,
  setTopFlagged
}) {

  const [showChat, setShowChat] = useState(false);

  // Search Thread State
  const [showSearch, setShowSearch] = useState(false);
  const [searchThread, setSearchThread] = useState("");

  // Reddit State
  const [showReddit, setShowReddit] = useState(false);
  const [redditURL, setRedditURL] = useState("");

  // Reddit progress bar
  const [redditLoading, setRedditLoading] = useState(false);
  const [redditProgress, setRedditProgress] = useState(0);

  // Quick check loading state
  const [quickLoading, setQuickLoading] = useState(false);

  const btn = (color) => ({
    padding: "12px 20px",
    background:
      color === "red"
        ? "linear-gradient(90deg,#dc2626,#ef4444)"
        : color === "green"
        ? "linear-gradient(90deg,#16a34a,#22c55e)"
        : color === "orange"
        ? "linear-gradient(90deg,#ea580c,#f97316)"
        : "linear-gradient(90deg,#4f46e5,#6366f1)",
    color: "white",
    borderRadius: "12px",
    cursor: "pointer",
    fontWeight: 600,
    boxShadow: "0 4px 10px rgba(0,0,0,0.5)",
    border: "none",
  });

  const analyzeBtn = {
    marginTop: 12,
    padding: "12px 20px",
    background: "linear-gradient(90deg,#16a34a,#22c55e)",
    color: "white",
    borderRadius: "12px",
    fontWeight: 700,
    cursor: "pointer",
    border: "none",
  };

  const spinner = {
    margin: "0 auto 10px",
    width: 28,
    height: 28,
    borderRadius: "50%",
    border: "3px solid rgba(255,255,255,0.2)",
    borderTopColor: "#22c55e",
    animation: "spin .9s linear infinite",
  };

  const progressBar = {
    width: 280,
    maxWidth: "90%",
    height: 10,
    background: "#0f172a",
    borderRadius: 999,
    overflow: "hidden",
    margin: "0 auto",
    boxShadow: "inset 0 0 6px rgba(0,0,0,0.6)",
  };

  // Helper function to get toxicity verdict
  const getToxicityVerdict = (score) => {
    if (score < 0.3) return { label: "SAFE", color: "#22c55e", emoji: "✅" };
    if (score < 0.5) return { label: "LOW RISK", color: "#84cc16", emoji: "⚠️" };
    if (score < 0.7) return { label: "MODERATE", color: "#f59e0b", emoji: "⚠️" };
    if (score < 0.85) return { label: "TOXIC", color: "#ef4444", emoji: "🚨" };
    return { label: "HIGHLY TOXIC", color: "#dc2626", emoji: "🚫" };
  };

  return (
    <div
      style={{
        position: "relative",
        display: "flex",
        justifyContent: "center",
        alignItems: "flex-start",
        minHeight: "100vh",
        overflow: "auto",
        background:
          "linear-gradient(135deg, #000000, #111827, #1f2937, #000000)",
        padding: "20px 0",
      }}
    >
      {/* Keyframes for spinner */}
      <style>{`
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
        .gradient-text {
          background: linear-gradient(90deg, #ef4444, #f97316, #eab308);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
        }
        .caret {
          animation: blink 1s step-end infinite;
        }
        @keyframes blink {
          50% { opacity: 0; }
        }
        .pulse-btn {
          animation: pulse 2s infinite;
        }
        @keyframes pulse {
          0%, 100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
          50% { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
        }
        
        /* Responsive styles */
        @media (max-width: 600px) {
          main {
            padding: 20px !important;
            margin: 10px !important;
          }
        }
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
          width: 8px;
        }
        ::-webkit-scrollbar-track {
          background: #1f2937;
          border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb {
          background: #4b5563;
          border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
          background: #6b7280;
        }
      `}</style>

      {/* ------------------------------------------------------
          TOP RIGHT: CHAT MODE + RECENT UPLOADS
      ------------------------------------------------------ */}
      <div
        style={{
          position: "fixed",
          top: 16,
          right: 16,
          zIndex: 9998,
          display: "flex",
          alignItems: "center",
          gap: 10,
        }}
      >

        <button
          onClick={() => setShowChat(true)}
          style={{
            padding: "10px 14px",
            background: "linear-gradient(90deg,#10b981,#22c55e)",
            color: "white",
            border: "none",
            borderRadius: 12,
            fontWeight: 700,
            cursor: "pointer",
            boxShadow: "0 4px 10px rgba(0,0,0,0.4)",
          }}
        >
          💬 Chat Mode
        </button>

        <button
          onClick={() => setShowRecent(!showRecent)}
          style={{
            padding: "10px 14px",
            background: "#374151",
            color: "white",
            border: "none",
            borderRadius: 12,
            fontWeight: 700,
            cursor: "pointer",
            boxShadow: "0 4px 10px rgba(0,0,0,0.4)",
          }}
        >
          📁 Recent ▾
        </button>

        {showRecent && (
          <div
            style={{
              position: "absolute",
              top: 50,
              right: 0,
              width: 320,
              maxHeight: 360,
              overflowY: "auto",
              background: "#1f2937",
              color: "white",
              borderRadius: 12,
              boxShadow: "0 12px 30px rgba(0,0,0,0.6)",
              padding: 10,
            }}
          >
            <div style={{ fontWeight: 800, color: "#f87171", marginBottom: 8 }}>
              Recent uploads
            </div>

            {recent.length === 0 ? (
              <div style={{ color: "#9ca3af" }}>No uploads yet.</div>
            ) : (
              <ul style={{ listStyle: "none", margin: 0, padding: 0 }}>
                {recent.map((r, i) => (
                  <li
                    key={r.name + i}
                    onClick={async () => {
                      setShowRecent(false);
                      await analyzeFilename(r.name);
                      setShowChart(true);
                    }}
                    style={{
                      padding: "8px 10px",
                      borderRadius: 8,
                      marginBottom: 6,
                      background: "#111827",
                      cursor: "pointer",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                    }}
                  >
                    <span>{r.name}</span>
                    <span style={{ color: "#9ca3af", fontSize: 12 }}>
                      {r.time}
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
      </div>

      {/* ------------------------------------------------------
          MAIN PANEL
      ------------------------------------------------------ */}
      <main
        style={{
          padding: "30px 40px",
          borderRadius: 16,
          background: "#111827",
          color: "white",
          textAlign: "center",
          maxWidth: 560,
          width: "90%",
          maxHeight: "90vh",
          overflowY: "auto",
          boxShadow: "0 12px 30px rgba(0,0,0,0.6)",
          margin: "auto",
        }}
      >
        <h1 style={{ marginBottom: 12, fontWeight: 800, fontSize: 32 }}>
          Welcome to <span className="gradient-text">SpotTox</span>
        </h1>

        <p
          style={{
            marginTop: 8,
            color: "#9ca3af",
            fontStyle: "italic",
            minHeight: "1.4em",
          }}
        >
          {typed}
          <span className="caret" style={{ color: "#9ca3af" }}>|</span>
        </p>

        <p style={{ color: "#d1d5db", marginBottom: 10, fontSize: 14 }}>
          🕐 {now}
        </p>

        {/* MODEL SELECTOR */}
        <div style={{ marginBottom: 20 }}>
          <label style={{ color: "#9ca3af", marginRight: 8, fontWeight: 600 }}>
            🤖 Model:
          </label>
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            style={{
              padding: "8px 12px",
              background: "#1f2937",
              color: "white",
              borderRadius: 8,
              border: "1px solid #374151",
              cursor: "pointer",
              fontWeight: 600,
              boxShadow: "0 4px 10px rgba(0,0,0,0.4)",
            }}
          >
            {models?.length > 0 ? (
              models.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))
            ) : (
              <option value="">Loading models...</option>
            )}
          </select>
        </div>

        {/* ============================================================
             "IS IT TOXIC?" BUTTON
        ============================================================ */}
        {file && !isAnalyzing && !quickLoading && (
          <div style={{ display: "flex", justifyContent: "center", marginBottom: 20 }}>
            <button
              className="pulse-btn"
              onClick={async () => {
                setQuickLoading(true);
                try {
                  await handleQuickCheck();
                } finally {
                  setQuickLoading(false);
                }
              }}
              style={{
                ...btn("red"),
                width: 300,
                padding: "16px 24px",
                fontSize: 18,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                gap: 10,
              }}
            >
              🔍 Is It Toxic?
            </button>
          </div>
        )}

        {quickLoading && (
          <div style={{ marginBottom: 20 }}>
            <div style={spinner} />
            <p style={{ color: "#9ca3af", fontSize: 14 }}>Analyzing toxicity...</p>
          </div>
        )}

        {/* ============================================================
            QUICK RESULT DISPLAY - Shows simple YES/NO answer
        ============================================================ */}
        {quickResult && !showChart && (
          <div
            style={{
              marginBottom: 20,
              padding: 20,
              borderRadius: 12,
              background: "#0f172a",
              border: `2px solid ${getToxicityVerdict(quickResult.mean).color}`,
            }}
          >
            <div style={{ fontSize: 48, marginBottom: 8 }}>
              {getToxicityVerdict(quickResult.mean).emoji}
            </div>
            <div
              style={{
                fontSize: 24,
                fontWeight: 800,
                color: getToxicityVerdict(quickResult.mean).color,
                marginBottom: 8,
              }}
            >
              {getToxicityVerdict(quickResult.mean).label}
            </div>
            <div style={{ color: "#9ca3af", fontSize: 14, marginBottom: 12 }}>
              Average Toxicity Score: <strong style={{ color: "white" }}>{(quickResult.mean * 100).toFixed(1)}%</strong>
            </div>
            
            {/* Message and Thread counts */}
            <div style={{ 
              display: "flex", 
              justifyContent: "center", 
              gap: 20, 
              marginBottom: 12,
              flexWrap: "wrap"
            }}>
              <div style={{ 
                background: "#1f2937", 
                padding: "8px 16px", 
                borderRadius: 8,
                display: "flex",
                alignItems: "center",
                gap: 8
              }}>
                <span>💬</span>
                <span style={{ color: "#d1d5db" }}>
                  <strong style={{ color: "white" }}>{quickResult.messageCount || 0}</strong> messages
                </span>
              </div>
              {quickResult.threadCount > 1 && (
                <div style={{ 
                  background: "#1f2937", 
                  padding: "8px 16px", 
                  borderRadius: 8,
                  display: "flex",
                  alignItems: "center",
                  gap: 8
                }}>
                  <span>🧵</span>
                  <span style={{ color: "#d1d5db" }}>
                    <strong style={{ color: "white" }}>{quickResult.threadCount}</strong> threads
                  </span>
                </div>
              )}
            </div>

            <div style={{ display: "flex", gap: 10, justifyContent: "center", flexWrap: "wrap" }}>
              <button
                onClick={() => {
                  setQuickResult(null);
                  setShowChart(true);
                }}
                style={{
                  ...btn("blue"),
                  padding: "10px 16px",
                  fontSize: 14,
                }}
              >
                📊 View Full Analysis
              </button>
              <button
                onClick={() => setQuickResult(null)}
                style={{
                  ...btn("green"),
                  padding: "10px 16px",
                  fontSize: 14,
                  background: "#374151",
                }}
              >
                ✕ Close
              </button>
            </div>
          </div>
        )}

        {/* UPLOAD BUTTONS */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 12,
            alignItems: "center",
            marginBottom: 10,
          }}
        >
          {/* Upload Thread (Single) */}
          <label
            style={{
              ...btn("red"),
              width: 240,
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              paddingTop: 12,
              paddingBottom: 12,
            }}
          >
            <span style={{ fontSize: 16, fontWeight: 700, display: "flex", alignItems: "center", gap: 8 }}>
              📄 Upload Thread
            </span>
            <span style={{ fontSize: 11, opacity: 0.85, marginTop: 4 }}>
              CSV, TXT, or JSON file
            </span>
            <input
              type="file"
              accept=".csv,.txt,.json"
              style={{ display: "none" }}
              onChange={handleUpload}
            />
          </label>

          {/* Upload Image/Screenshot */}
          <label
            style={{
              ...btn("orange"),
              width: 240,
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              paddingTop: 12,
              paddingBottom: 12,
            }}
          >
            <span style={{ fontSize: 16, fontWeight: 700, display: "flex", alignItems: "center", gap: 8 }}>
              📸 Upload Image
            </span>
            <span style={{ fontSize: 11, opacity: 0.85, marginTop: 4 }}>
              Image of conversation (PNG, JPG, JPEG, GIF)
            </span>
            <input
              type="file"
              accept="image/*"
              style={{ display: "none" }}
              onChange={handleImageUpload}
            />
          </label>

          {/* Upload Multiple */}
          <label style={{ 
            ...btn("blue"), 
            width: 240, 
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            paddingTop: 12,
            paddingBottom: 12,
          }}>
            <span style={{ fontSize: 16, fontWeight: 700, display: "flex", alignItems: "center", gap: 8 }}>
              📚 Upload Multiple
            </span>
            <span style={{ fontSize: 11, opacity: 0.85, marginTop: 4 }}>
              Compare multiple threads
            </span>
            <input
              type="file"
              accept=".csv,.txt,.json"
              multiple
              style={{ display: "none" }}
              onChange={handleMultiUpload}
            />
          </label>
        </div>

        {/* Reddit Button */}
        <button
          onClick={() => setShowReddit(true)}
          style={{
            marginTop: 16,
            padding: "12px 20px",
            background: "linear-gradient(90deg,#ff4500,#ff6a00)",
            color: "white",
            borderRadius: 12,
            fontWeight: 700,
            cursor: "pointer",
            boxShadow: "0 4px 10px rgba(0,0,0,0.5)",
            border: "none",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: 8,
            width: 240,
            margin: "16px auto 0",
          }}
        >
          🔗 Analyze Reddit Link
        </button>

        {/* SEARCH THREAD BUTTON */}
        {(file || multiFiles.length > 0) && (
          <button
            onClick={() => setShowSearch(true)}
            style={{
              marginTop: 12,
              padding: "12px 20px",
              background: "linear-gradient(90deg,#3b82f6,#6366f1)",
              color: "white",
              borderRadius: 12,
              fontWeight: 700,
              cursor: "pointer",
              boxShadow: "0 4px 10px rgba(0,0,0,0.5)",
              border: "none",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              gap: 8,
              width: 240,
              margin: "12px auto 0",
            }}
          >
            🔎 Search Thread by ID
          </button>
        )}

        {multiFiles.length > 0 && (
          <div style={{ marginTop: 12, color: "#fca5a5", textAlign: "left" }}>
            <b>📂 Uploaded Threads ({multiFiles.length}):</b>
            <ul style={{ margin: "8px 0", paddingLeft: 20 }}>
              {multiFiles.map((f, i) => <li key={i} style={{ fontSize: 14 }}>{f}</li>)}
            </ul>
          </div>
        )}

        {uploadedName && (
          <div style={{ 
            marginTop: 16, 
            padding: "12px 16px",
            background: "#1f2937",
            borderRadius: 8,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: 8
          }}>
            <span>📄</span>
            <span style={{ color: "#fca5a5" }}>
              Uploaded: <strong>{uploadedName}</strong>
            </span>
          </div>
        )}

        {file && !isAnalyzing && !quickResult && (
          <button style={{ ...analyzeBtn, width: 240, margin: "16px auto 0", display: "block" }} onClick={handleAnalyze}>
            📊 Full Analysis (Charts)
          </button>
        )}

        {multiFiles.length >= 2 && !isAnalyzing && (
          <button style={{ ...analyzeBtn, width: 280, margin: "12px auto 0", display: "block" }} onClick={analyzeMultipleThreads}>
            📊 Analyze {multiFiles.length} Threads
          </button>
        )}

        {isAnalyzing && (
          <div style={{ marginTop: 16 }}>
            <div style={spinner} />
            <div style={progressBar}>
              <div
                style={{
                  width: `${progress}%`,
                  height: "100%",
                  background: "linear-gradient(90deg, #16a34a, #22c55e)",
                  transition: "width 0.3s ease",
                }}
              />
            </div>
            <div style={{ marginTop: 8, color: "#9ca3af", fontSize: 12 }}>
              {progressMessage || `Analyzing… ${progress}%`}
            </div>
          </div>
        )}

        {/* ======================================================
            REDDIT MODAL WITH PROGRESS BAR
        ====================================================== */}
        {showReddit && (
          <div
            style={{
              position: "fixed",
              top: 0,
              left: 0,
              width: "100vw",
              height: "100vh",
              background: "rgba(0,0,0,0.6)",
              zIndex: 99999,
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
            }}
          >
            <div
              style={{
                background: "#1f2937",
                padding: 25,
                borderRadius: 12,
                width: "90%",
                maxWidth: 500,
                boxShadow: "0 12px 30px rgba(0,0,0,0.8)",
              }}
            >
              <button
                onClick={() => !redditLoading && setShowReddit(false)}
                style={{
                  float: "right",
                  background: "#ef4444",
                  color: "white",
                  border: "none",
                  padding: "6px 12px",
                  borderRadius: 8,
                  cursor: redditLoading ? "not-allowed" : "pointer",
                }}
              >
                Close
              </button>

              <h2 style={{ marginBottom: 15, textAlign: "center" }}>
                🔗 Analyze Reddit Thread
              </h2>

              <input
                type="text"
                placeholder="Paste Reddit link here..."
                value={redditURL}
                disabled={redditLoading}
                onChange={(e) => setRedditURL(e.target.value)}
                style={{
                  width: "100%",
                  padding: "12px",
                  borderRadius: 8,
                  border: "1px solid #374151",
                  background: "#111827",
                  color: "white",
                  marginBottom: 12,
                  boxSizing: "border-box",
                }}
              />

              {!redditLoading ? (
                <button
                  onClick={async () => {
                    if (!redditURL.trim()) {
                      alert("Please paste a Reddit link.");
                      return;
                    }

                    setRedditLoading(true);
                    setRedditProgress(0);

                    const interval = setInterval(() => {
                      setRedditProgress((p) =>
                        Math.min(90, p + Math.random() * 3 + 1)
                      );
                    }, 300);

                    try {
                      const res = await fetch("http://127.0.0.1:5001/analyze_reddit", {
                        method: "POST",
                        headers: {
                          "Content-Type": "application/json",
                        },
                        body: JSON.stringify({
                          url: redditURL,
                          model: selectedModel,
                        }),
                      });

                      const data = await res.json();

                      clearInterval(interval);

                      if (!res.ok) {
                        alert(data.error || "Analysis failed.");
                        setRedditLoading(false);
                        return;
                      }

                      setRedditProgress(100);

                      setTimeout(() => {
                        setRedditLoading(false);
                        setShowReddit(false);

                        setSummary({
                          mean: data.mean,
                          p90: data.p90,
                          p95: data.p95,
                          max: data.max,
                        });

                        setHistogram(makeHistogram(data.histogram));
                        setTopFlagged(data.top || []);

                        setShowChart(true);
                      }, 400);
                    } catch (err) {
                      clearInterval(interval);
                      setRedditLoading(false);
                      alert("Backend error: " + err.message);
                    }
                  }}
                  style={{
                    width: "100%",
                    padding: "12px",
                    background: "linear-gradient(90deg,#ff4500,#ff6a00)",
                    color: "white",
                    borderRadius: 10,
                    fontWeight: 700,
                    cursor: "pointer",
                    border: "none",
                  }}
                >
                  🔍 Analyze
                </button>
              ) : (
                <div
                  style={{
                    width: "100%",
                    height: 14,
                    background: "#0f172a",
                    borderRadius: 10,
                    overflow: "hidden",
                    boxShadow: "inset 0 0 6px rgba(0,0,0,0.6)",
                  }}
                >
                  <div
                    style={{
                      width: `${redditProgress}%`,
                      height: "100%",
                      background: "linear-gradient(90deg,#ff4500,#ff6a00)",
                      transition: "width 0.2s ease",
                    }}
                  />
                </div>
              )}
            </div>
          </div>
        )}

        {/* ======================================================
            THREAD SEARCH MODAL
        ====================================================== */}
        {showSearch && (
          <div
            style={{
              position: "fixed",
              top: 0,
              left: 0,
              width: "100vw",
              height: "100vh",
              background: "rgba(0,0,0,0.6)",
              zIndex: 99999,
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
            }}
          >
            <div
              style={{
                background: "#1f2937",
                padding: 25,
                borderRadius: 12,
                width: "90%",
                maxWidth: 500,
                boxShadow: "0 12px 30px rgba(0,0,0,0.8)",
              }}
            >
              <button
                onClick={() => setShowSearch(false)}
                style={{
                  float: "right",
                  background: "#ef4444",
                  color: "white",
                  border: "none",
                  padding: "6px 12px",
                  borderRadius: 8,
                  cursor: "pointer",
                }}
              >
                Close
              </button>

              <h2 style={{ marginBottom: 15, textAlign: "center" }}>
                🔎 Search Thread by ID
              </h2>

              <input
                type="text"
                placeholder="Enter thread_id..."
                value={searchThread}
                onChange={(e) => setSearchThread(e.target.value)}
                style={{
                  width: "100%",
                  padding: "12px",
                  borderRadius: 8,
                  border: "1px solid #374151",
                  background: "#111827",
                  color: "white",
                  marginBottom: 12,
                  boxSizing: "border-box",
                }}
              />

              <button
                onClick={async () => {
                  if (!searchThread.trim()) {
                    alert("Please enter a thread ID.");
                    return;
                  }

                  try {
                    const res = await fetch("http://127.0.0.1:5001/search_thread", {
                      method: "POST",
                      headers: {
                        "Content-Type": "application/json",
                      },
                      body: JSON.stringify({ thread_id: searchThread }),
                    });

                    const data = await res.json();

                    if (!data.found) {
                      alert("Thread not found in dataset.");
                    } else {
                      const verdict = getToxicityVerdict(data.mean);
                      alert(
                        `${verdict.emoji} Thread: ${data.thread_id}\n\n` +
                        `Verdict: ${verdict.label}\n` +
                        `─────────────────\n` +
                        `💬 Comments: ${data.count}\n` +
                        `📊 Mean: ${(data.mean * 100).toFixed(1)}%\n` +
                        `📈 P90: ${(data.p90 * 100).toFixed(1)}%\n` +
                        `📉 P95: ${(data.p95 * 100).toFixed(1)}%\n` +
                        `⚠️ Max: ${(data.max * 100).toFixed(1)}%`
                      );
                    }
                  } catch (err) {
                    alert("Backend error: " + err.message);
                  }
                }}
                style={{
                  width: "100%",
                  padding: "12px",
                  background: "linear-gradient(90deg,#3b82f6,#6366f1)",
                  color: "white",
                  borderRadius: 10,
                  fontWeight: 700,
                  cursor: "pointer",
                  border: "none",
                }}
              >
                🔍 Search
              </button>
            </div>
          </div>
        )}

        {/* ======================================================
            CHAT MODAL
        ====================================================== */}
        {showChat && (
          <div
            style={{
              position: "fixed",
              top: 0,
              left: 0,
              width: "100vw",
              height: "100vh",
              background: "rgba(0,0,0,0.6)",
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
              zIndex: 99999,
            }}
          >
            <div
              style={{
                background: "#1f2937",
                padding: 20,
                borderRadius: 12,
                width: "90%",
                maxWidth: 550,
                maxHeight: "80vh",
                overflowY: "auto",
                boxShadow: "0 12px 30px rgba(0,0,0,0.8)",
              }}
            >
              <button
                onClick={() => setShowChat(false)}
                style={{
                  float: "right",
                  background: "#ef4444",
                  color: "white",
                  border: "none",
                  padding: "6px 12px",
                  borderRadius: 8,
                  fontWeight: 700,
                  cursor: "pointer",
                }}
              >
                Close
              </button>

              <h2 style={{ textAlign: "center", marginBottom: 12 }}>
                💬 Real-Time Chat Mode
              </h2>

              <ChatUI
                model={selectedModel}
                models={models}
                onModelChange={setSelectedModel}
              />
            </div>
          </div>
        )}

      </main>

      {/* ======================================================
          CHART MODAL - Full Analysis View
      ====================================================== */}
      <ChartModal
        open={showChart}
        onClose={() => {
          setShowChart(false);
          setBinDetail(null);
        }}
        title="Toxicity Score Distribution"
        modelName={modelName}
      >
        {/* Summary Stats at top */}
        {summary && (
          <div style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(100px, 1fr))",
            gap: 12,
            marginBottom: 20,
          }}>
            <div style={{ background: "#1f2937", padding: 12, borderRadius: 8, textAlign: "center" }}>
              <div style={{ color: "#9ca3af", fontSize: 12, marginBottom: 4 }}>Mean</div>
              <div style={{ color: getToxicityVerdict(summary.mean).color, fontSize: 20, fontWeight: 800 }}>
                {(summary.mean * 100).toFixed(1)}%
              </div>
            </div>
            <div style={{ background: "#1f2937", padding: 12, borderRadius: 8, textAlign: "center" }}>
              <div style={{ color: "#9ca3af", fontSize: 12, marginBottom: 4 }}>P90</div>
              <div style={{ color: "#f59e0b", fontSize: 20, fontWeight: 800 }}>
                {(summary.p90 * 100).toFixed(1)}%
              </div>
            </div>
            <div style={{ background: "#1f2937", padding: 12, borderRadius: 8, textAlign: "center" }}>
              <div style={{ color: "#9ca3af", fontSize: 12, marginBottom: 4 }}>P95</div>
              <div style={{ color: "#ef4444", fontSize: 20, fontWeight: 800 }}>
                {(summary.p95 * 100).toFixed(1)}%
              </div>
            </div>
            <div style={{ background: "#1f2937", padding: 12, borderRadius: 8, textAlign: "center" }}>
              <div style={{ color: "#9ca3af", fontSize: 12, marginBottom: 4 }}>Max</div>
              <div style={{ color: "#dc2626", fontSize: 20, fontWeight: 800 }}>
                {(summary.max * 100).toFixed(1)}%
              </div>
            </div>
          </div>
        )}

        {/* Message/Thread counts */}
        {(messageCount || threadCount) && (
          <div style={{
            display: "flex",
            justifyContent: "center",
            gap: 16,
            marginBottom: 16,
            flexWrap: "wrap"
          }}>
            {messageCount && (
              <div style={{ 
                background: "#1f2937", 
                padding: "8px 16px", 
                borderRadius: 8,
                display: "flex",
                alignItems: "center",
                gap: 8
              }}>
                <span>💬</span>
                <span style={{ color: "#d1d5db" }}>
                  <strong style={{ color: "white" }}>{messageCount}</strong> messages analyzed
                </span>
              </div>
            )}
            {threadCount > 1 && (
              <div style={{ 
                background: "#1f2937", 
                padding: "8px 16px", 
                borderRadius: 8,
                display: "flex",
                alignItems: "center",
                gap: 8
              }}>
                <span>🧵</span>
                <span style={{ color: "#d1d5db" }}>
                  <strong style={{ color: "white" }}>{threadCount}</strong> threads
                </span>
              </div>
            )}
          </div>
        )}

        <Histogram
          data={histogram}
          threshold={0.2}
          onBinClick={(info) => setBinDetail(info)}
        />

        {binDetail && (
          <div
            style={{
              textAlign: "center",
              color: "#9ca3af",
              marginTop: 6,
            }}
          >
            Range: {(binDetail.from * 100).toFixed(0)}% – {(binDetail.to * 100).toFixed(0)}% • Count: {binDetail.count}
          </div>
        )}

        {/* ===================== SCROLLABLE FLAGGED COMMENTS ===================== */}
        {topFlagged && topFlagged.length > 0 && (
          <div
            style={{
              marginTop: 20,
              maxHeight: "250px",
              overflowY: "auto",
              paddingRight: 10,
            }}
          >
            <h3 style={{ color: "white", marginBottom: 10, fontWeight: 800, display: "flex", alignItems: "center", gap: 8 }}>
              🚨 Most Toxic Comments
            </h3>

            {topFlagged.map((item, idx) => (
              <div
                key={idx}
                style={{
                  background: "#1f2937",
                  padding: "10px 14px",
                  borderRadius: 8,
                  marginBottom: 10,
                  boxShadow: "0 4px 10px rgba(0,0,0,0.4)",
                  color: "white",
                  borderLeft: `4px solid ${getToxicityVerdict(item.score).color}`,
                }}
              >
                <div
                  style={{
                    color: getToxicityVerdict(item.score).color,
                    fontWeight: 700,
                    marginBottom: 6,
                    display: "flex",
                    alignItems: "center",
                    gap: 8,
                  }}
                >
                  <span>{getToxicityVerdict(item.score).emoji}</span>
                  <span>Score: {(item.score * 100).toFixed(1)}%</span>
                </div>
                <div
                  style={{
                    fontSize: 14,
                    whiteSpace: "pre-wrap",
                    wordBreak: "break-word",
                    color: "#d1d5db",
                  }}
                >
                  {item.text}
                </div>
              </div>
            ))}
          </div>
        )}
      </ChartModal>
    </div>
  );
}