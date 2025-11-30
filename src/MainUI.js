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

  // analysis
  handleAnalyze,
  analyzeMultipleThreads,

  // state
  isAnalyzing,
  progress,
  summary,
  histogram,
  topFlagged,

  // chart
  showChart,
  setShowChart,
  binDetail,
  setBinDetail,

  // models
  models,
  selectedModel,
  setSelectedModel,
  modelName
}) {

  const [showChat, setShowChat] = useState(false);

  // SEARCH THREAD STATE
  const [showSearch, setShowSearch] = useState(false);
  const [searchThread, setSearchThread] = useState("");

  const btn = (color) => ({
    padding: "12px 20px",
    background:
      color === "red"
        ? "linear-gradient(90deg,#dc2626,#ef4444)"
        : "linear-gradient(90deg,#4f46e5,#6366f1)",
    color: "white",
    borderRadius: "12px",
    cursor: "pointer",
    fontWeight: 600,
    boxShadow: "0 4px 10px rgba(0,0,0,0.5)",
  });

  const analyzeBtn = {
    marginTop: 12,
    padding: "12px 20px",
    background: "linear-gradient(90deg,#16a34a,#22c55e)",
    color: "white",
    borderRadius: "12px",
    fontWeight: 700,
    cursor: "pointer",
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

  return (
    <div
      style={{
        position: "relative",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        height: "100vh",
        overflow: "hidden",
        background:
          "linear-gradient(135deg, #000000, #111827, #1f2937, #000000)",
      }}
    >

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

        {/* CHAT BUTTON (MOVED HERE) */}
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
          Chat Mode
        </button>

        {/* RECENT UPLOADS BUTTON */}
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
          Recent uploads ▾
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
          padding: 40,
          borderRadius: 16,
          background: "#111827",
          color: "white",
          textAlign: "center",
          maxWidth: 520,
          width: "90%",
          boxShadow: "0 12px 30px rgba(0,0,0,0.6)",
        }}
      >
        <h1 style={{ marginBottom: 12, fontWeight: 800 }}>
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
          <span className="caret" style={{ color: "#9ca3af" }} />
        </p>

        <p style={{ color: "#d1d5db", marginBottom: 10 }}>
          Time & Date: <strong>{now}</strong>
        </p>

        {/* MODEL SELECTOR */}
        <div style={{ marginBottom: 20 }}>
          <label style={{ color: "#9ca3af", marginRight: 8, fontWeight: 600 }}>
            Model:
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

        {/* UPLOAD BUTTONS — UPDATED */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 12,
            alignItems: "center",
            marginBottom: 10,
          }}
        >

          {/* Upload Dataset */}
          <label
            style={{
              ...btn("red"),
              width: 220,
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              paddingTop: 10,
              paddingBottom: 10,
            }}
          >
            <span style={{ fontSize: 16, fontWeight: 700 }}>
              Upload Dataset
            </span>
            <span style={{ fontSize: 10, opacity: 0.85, marginTop: 2 }}>
              For larger files containing many threads
            </span>

            <input
              type="file"
              accept=".csv,.txt,.json"
              style={{ display: "none" }}
              onChange={handleUpload}
            />
          </label>

          {/* NEW Upload Thread */}
          <label
            style={{
              ...btn("red"),
              width: 220,
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              paddingTop: 10,
              paddingBottom: 10,
              background: "linear-gradient(90deg,#b91c1c,#dc2626)"
            }}
          >
            <span style={{ fontSize: 16, fontWeight: 700 }}>
              Upload Thread
            </span>
            <span style={{ fontSize: 10, opacity: 0.85, marginTop: 2 }}>
              For smaller files containing 1 thread
            </span>

            <input
              type="file"
              accept=".csv,.txt,.json"
              style={{ display: "none" }}
              onChange={handleUpload}
            />
          </label>

          {/* Upload Multiple */}
          <label style={{ ...btn("blue"), width: 220, textAlign: "center" }}>
            Upload Multiple
            <input
              type="file"
              accept=".csv,.txt,.json"
              multiple
              style={{ display: "none" }}
              onChange={handleMultiUpload}
            />
          </label>

        </div>

        {/* SEARCH THREAD BUTTON */}
        {(file || multiFiles.length > 0) && (
          <button
            onClick={() => setShowSearch(true)}
            style={{
              marginTop: 16,
              padding: "12px 20px",
              background: "linear-gradient(90deg,#3b82f6,#6366f1)",
              color: "white",
              borderRadius: 12,
              fontWeight: 700,
              cursor: "pointer",
              boxShadow: "0 4px 10px rgba(0,0,0,0.5)",
            }}
          >
            Search Thread
          </button>
        )}

        {multiFiles.length > 0 && (
          <div style={{ marginTop: 12, color: "#fca5a5", textAlign: "left" }}>
            <b>Uploaded Threads:</b>
            <ul>{multiFiles.map((f, i) => <li key={i}>{f}</li>)}</ul>
          </div>
        )}

        {uploadedName && (
          <p style={{ marginTop: 16, color: "#fca5a5" }}>
            Uploaded: <strong>{uploadedName}</strong>
          </p>
        )}

        {file && !isAnalyzing && (
          <button style={analyzeBtn} onClick={handleAnalyze}>
            Analyze Thread
          </button>
        )}

        {multiFiles.length >= 2 && !isAnalyzing && (
          <button style={analyzeBtn} onClick={analyzeMultipleThreads}>
            Analyze {multiFiles.length} Threads
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
                }}
              />
            </div>
            <div style={{ marginTop: 8, color: "#9ca3af", fontSize: 12 }}>
              Analyzing… {progress}%
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
                Search Thread by ID
              </h2>

              <input
                type="text"
                placeholder="Enter thread_id"
                value={searchThread}
                onChange={(e) => setSearchThread(e.target.value)}
                style={{
                  width: "100%",
                  padding: "10px",
                  borderRadius: 8,
                  border: "1px solid #374151",
                  background: "#111827",
                  color: "white",
                  marginBottom: 12,
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
                      headers: { "Content-Type": "application/json" },
                      body: JSON.stringify({ thread_id: searchThread }),
                    });

                    const data = await res.json();

                    if (!data.found) {
                      alert("Thread not found in dataset.");
                    } else {
                      alert(
                        `Thread ID: ${data.thread_id}\n` +
                        `Number of comments: ${data.count}\n` +
                        `Mean toxicity: ${data.mean.toFixed(3)}\n` +
                        `P90 toxicity: ${data.p90.toFixed(3)}\n` +
                        `P95 toxicity: ${data.p95.toFixed(3)}\n` +
                        `Max toxicity: ${data.max.toFixed(3)}`
                      );
                      console.log("Thread stats:", data);
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
                }}
              >
                Search
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
                Real-Time Chat Mode
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

      {/* CHART MODAL */}
      <ChartModal
        open={showChart}
        onClose={() => {
          setShowChart(false);
          setBinDetail(null);
        }}
        title="Toxicity Score Distribution"
        modelName={modelName}
      >
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
            {binDetail.from?.toFixed(3)} – {binDetail.to?.toFixed(3)} • Count:{" "}
            {binDetail.count}
          </div>
        )}
      </ChartModal>
    </div>
  );
}
