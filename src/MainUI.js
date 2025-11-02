import React from "react";
import { Histogram, ChartModal } from "./Chart";

export default function MainUI({
  now,
  typed,

  // recent uploads state
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

  // analysis actions
  handleAnalyze,
  analyzeMultipleThreads,

  // analysis state
  isAnalyzing,
  progress,
  summary,
  histogram,
  topFlagged,

  // chart modal
  showChart,
  setShowChart,
  binDetail,
  setBinDetail,

  // ✅ MODEL SELECTION
  selectedModel,
  setSelectedModel,
  modelName
}) {
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
        background: "linear-gradient(135deg, #000000, #111827, #1f2937, #000000)",
      }}
    >
      <style>{`
        @keyframes blink {0%,50%{opacity:1;}51%,100%{opacity:0;}}
        .gradient-text {
          background: linear-gradient(90deg,#dc2626,#ef4444,#f87171);
          background-size:200% 200%;
          -webkit-background-clip:text;
          color:transparent;
          animation:moveGradient 4s ease infinite;
          font-weight:900;
        }
        .caret{display:inline-block;width:10px;margin-left:4px;background:currentColor;height:1.05em;vertical-align:text-bottom;animation:blink 1s steps(1) infinite;}
      `}</style>

      {/* ───────────────────── Recent Uploads Dropdown ───────────────────── */}
      <div style={{ position: "fixed", top: 16, right: 16, zIndex: 9998 }}>
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
              marginTop: 8,
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
                    <span style={{ color: "#9ca3af", fontSize: 12 }}>{r.time}</span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
      </div>

      {/* ───────────────────── Main UI Card ───────────────────── */}
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
          position: "relative",
        }}
      >
        <h1 style={{ marginBottom: 12, fontWeight: 800 }}>
          Welcome to <span className="gradient-text">SpotTox</span>
        </h1>

        <p style={{ marginTop: 8, color: "#9ca3af", fontStyle: "italic", minHeight: "1.4em" }}>
          {typed}<span className="caret" style={{ color: "#9ca3af" }} />
        </p>

        <p style={{ color: "#d1d5db", marginBottom: 10 }}>
          Time & Date: <strong>{now}</strong>
        </p>

        {/* ✅ MODEL DROPDOWN */}
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
            <option value="SpotToxBERT">SpotToxBERT</option>
            <option value="SpotToxRoBERTa">SpotToxRoBERTa</option>
          </select>
        </div>

        {/* UPLOAD BUTTONS */}
        <div style={{ display: "flex", gap: 12, justifyContent: "center" }}>
          <label style={btn("red")}>
            Upload Thread
            <input type="file" accept=".csv,.txt,.json" style={{ display: "none" }} onChange={handleUpload} />
          </label>

          <label style={btn("blue")}>
            Upload Multiple
            <input type="file" accept=".csv,.txt,.json" multiple style={{ display: "none" }} onChange={handleMultiUpload} />
          </label>
        </div>

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

        {/* ▸ Analyze Buttons */}
        {file && !isAnalyzing && <button style={analyzeBtn} onClick={handleAnalyze}>Analyze Thread</button>}
        {multiFiles.length >= 2 && !isAnalyzing && <button style={analyzeBtn} onClick={analyzeMultipleThreads}>Analyze {multiFiles.length} Threads</button>}

        {isAnalyzing && (
          <div style={{ marginTop: 16 }}>
            <div style={spinner} />
            <div style={progressBar}>
              <div style={{
                width: `${progress}%`,
                height: "100%",
                background: "linear-gradient(90deg, #16a34a, #22c55e)",
              }} />
            </div>
            <div style={{ marginTop: 8, color: "#9ca3af", fontSize: 12 }}>Analyzing… {progress}%</div>
          </div>
        )}

        {/* RESULTS PANEL (scrollable) */}
        {summary && histogram && (
          <div
            style={{
              marginTop: 18,
              padding: 12,
              background: "#1f2937",
              borderRadius: 12,
              maxHeight: "350px",
              overflowY: "auto",
              scrollbarWidth: "thin",
            }}
          >
            <div style={{ color: "#e5e7eb", marginBottom: 8 }}>
              <div>Mean: <b>{summary.mean.toFixed(3)}</b></div>
              <div>P90: <b>{summary.p90.toFixed(3)}</b></div>
              <div>P95: <b>{summary.p95.toFixed(3)}</b></div>
              <div>Max: <b>{summary.max.toFixed(3)}</b></div>
            </div>

            <button
              onClick={() => setShowChart(true)}
              style={{
                padding: "10px 16px",
                background: "#374151",
                color: "white",
                borderRadius: 10,
                fontWeight: 700,
                cursor: "pointer",
              }}
            >
              View Histogram
            </button>

            {topFlagged?.length > 0 && (
              <div style={{ marginTop: 10, textAlign: "left" }}>
                <div style={{ color: "#fca5a5", fontWeight: 700 }}>Top Flagged Comments</div>
                <ul>
                  {topFlagged.map((r, i) => (
                    <li key={i} style={{ marginBottom: 8 }}>
                      <i style={{ color: "#f87171" }}>{r.text}</i>{" — Score: "}{r.score.toFixed(3)}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </main>

      {/* HISTOGRAM MODAL */}
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
          <div style={{ textAlign: "center", color: "#9ca3af", marginTop: 6 }}>
            {binDetail.from.toFixed(3)} – {binDetail.to.toFixed(3)} • Count: {binDetail.count}
          </div>
        )}
      </ChartModal>
    </div>
  );
}
