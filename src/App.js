import { useState, useEffect, useMemo } from "react";

export default function App() {
  const [count, setCount] = useState(0);
  const [now, setNow] = useState(() => new Date().toLocaleString());
  const [file, setFile] = useState(null);

  // recent uploads
  const [recent, setRecent] = useState([]);
  const [showRecent, setShowRecent] = useState(false);

  // tagline list
  const taglines = useMemo(
    () => [
      "AI-powered early warning detection for toxic conversations.",
      "Predict toxicity before it spreads.",
      "Safer online spaces.",
    ],
    []
  );

  // Typing animation state
  const [tagIndex, setTagIndex] = useState(0);
  const [typed, setTyped] = useState("");
  const [isDeleting, setIsDeleting] = useState(false);

  // Analyze loading state
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);

  // Card tilt
  const [tilt, setTilt] = useState({ x: 0, y: 0 });

  // Clock
  useEffect(() => {
    const t = setInterval(() => setNow(new Date().toLocaleString()), 1000);
    return () => clearInterval(t);
  }, []);

  // Typing effect for taglines
  useEffect(() => {
    const full = taglines[tagIndex];
    const speed = isDeleting ? 35 : 55; // typing speed
    const donePause = 1000; // pause when a line finished

    const tick = setTimeout(() => {
      if (!isDeleting) {
        // typing forward
        const next = full.slice(0, typed.length + 1);
        setTyped(next);
        if (next === full) {
          // pause then start deleting
          setTimeout(() => setIsDeleting(true), donePause);
        }
      } else {
        // deleting
        const next = full.slice(0, typed.length - 1);
        setTyped(next);
        if (next.length === 0) {
          setIsDeleting(false);
          setTagIndex((i) => (i + 1) % taglines.length);
        }
      }
    }, speed);

    return () => clearTimeout(tick);
  }, [typed, isDeleting, tagIndex, taglines]);

  function handleUpload(e) {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setRecent((prev) => {
        const item = { name: selectedFile.name, time: new Date().toLocaleString() };
        return [item, ...prev].slice(0, 5);
      });
    }
  }

  // Simulated analyze with progress bar
  function handleAnalyze() {
    if (!file) return;
    setIsAnalyzing(true);
    setProgress(0);
    
    const interval = setInterval(() => {
      setProgress((p) => {
        const inc = Math.random() * 18 + 7; // 7–25%
        const next = Math.min(100, Math.round(p + inc));
        if (next >= 100) {
          clearInterval(interval);
          // small finish delay
          setTimeout(() => {
            setIsAnalyzing(false);
            alert(`Analysis complete for "${file.name}" (connect to backend next).`);
          }, 350);
        }
        return next;
      });
    }, 350);
  }

  function handleCardMouseMove(e) {
    const rect = e.currentTarget.getBoundingClientRect();
    const cx = rect.left + rect.width / 2;
    const cy = rect.top + rect.height / 2;
    const dx = (e.clientX - cx) / (rect.width / 2);  // -1..1
    const dy = (e.clientY - cy) / (rect.height / 2); // -1..1
    setTilt({ x: dy * 6, y: -dx * 6 }); // rotateX by dy, rotateY by -dx
  }

  function handleCardMouseLeave() {
    setTilt({ x: 0, y: 0 });
  }


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
      {/* Floating particles */}
      <style>{`
        @keyframes floatUp {
          0% { transform: translateY(20vh); opacity: 0; }
          10% { opacity: 1; }
          90% { opacity: 1; }
          100% { transform: translateY(-120vh); opacity: 0; }
        }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(6px);} to { opacity: 1; transform: translateY(0);} }
        @keyframes pulseShadow { 
          0% { box-shadow: 0 12px 30px rgba(0,0,0,0.5);} 
          50% { box-shadow: 0 16px 40px rgba(0,0,0,0.7);} 
          100% { box-shadow: 0 12px 30px rgba(0,0,0,0.5);} 
        }
        @keyframes moveGradient {
          0% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
          100% { background-position: 0% 50%; }
        }
        .gradient-text {
          background: linear-gradient(90deg, #dc2626, #ef4444, #f87171);
          background-size: 200% 200%;
          -webkit-background-clip: text;
          background-clip: text;
          color: transparent;
          animation: moveGradient 4s ease infinite;
          font-weight: 900;
        }
        .caret {
          display: inline-block;
          width: 10px;
          margin-left: 4px;
          background: currentColor;
          height: 1.05em;
          vertical-align: text-bottom;
          opacity: 1;
          animation: blink 1s steps(1) infinite;
        }
        @keyframes blink {
          0%, 50% { opacity: 1; }
          51%, 100% { opacity: 0; }
        }
      `}</style>


      <main
        onMouseMove={handleCardMouseMove}
        onMouseLeave={handleCardMouseLeave}
        style={{
          fontFamily: "system-ui, sans-serif",
          padding: 40,
          borderRadius: 16,
          background: "#111827",
          color: "white",
          textAlign: "center",
          maxWidth: 520,
          width: "90%",
          transform: `perspective(800px) rotateX(${tilt.x}deg) rotateY(${tilt.y}deg)`,
          transition: "transform 120ms ease",
          boxShadow: "0 12px 30px rgba(0,0,0,0.6)",
          animation: "fadeIn 0.8s ease, pulseShadow 3.5s ease-in-out infinite",
        }}
      >
        <h1 style={{ marginBottom: 12, fontWeight: 800 }}>
          Welcome to <span className="gradient-text">SpotTox</span>
        </h1>

        {/* Typing tagline */}
        <p
          style={{
            marginTop: 8,
            color: "#9ca3af",
            fontStyle: "italic",
            minHeight: "1.4em",
            whiteSpace: "nowrap",
            overflow: "hidden",
          }}
        >
          {typed}
          <span className="caret" style={{ color: "#9ca3af" }} />
        </p>

        <p style={{ color: "#d1d5db", marginBottom: 24 }}>
          Time: <strong>{now}</strong>
        </p>

        {/* Upload + Recent button */}
        <div style={{ display: "flex", gap: 12, justifyContent: "center", flexWrap: "wrap" }}>
          <label
            style={{
              display: "inline-block",
              padding: "12px 20px",
              background: "linear-gradient(90deg, #dc2626, #ef4444)",
              color: "white",
              borderRadius: "12px",
              cursor: "pointer",
              fontWeight: "600",
              boxShadow: "0 4px 10px rgba(0,0,0,0.5)",
              transition: "all 0.2s ease-in-out",
              opacity: isAnalyzing ? 0.6 : 1,
              pointerEvents: isAnalyzing ? "none" : "auto",
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.transform = "scale(1.05)";
              e.currentTarget.style.boxShadow = "0 6px 14px rgba(0,0,0,0.6)";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.transform = "scale(1)";
              e.currentTarget.style.boxShadow = "0 4px 10px rgba(0,0,0,0.5)";
            }}
          >
            Upload Thread
            <input
              type="file"
              accept=".txt,.json"
              style={{ display: "none" }}
              onChange={handleUpload}
            />
          </label>

          <button
            onClick={() => setShowRecent((s) => !s)}
            style={{
              padding: "12px 20px",
              background: "#374151",
              color: "white",
              border: "none",
              borderRadius: "12px",
              fontWeight: 700,
              cursor: "pointer",
              boxShadow: "0 4px 10px rgba(0,0,0,0.4)",
              transition: "transform .15s ease",
            }}
            onMouseEnter={(e) => (e.currentTarget.style.transform = "scale(1.05)")}
            onMouseLeave={(e) => (e.currentTarget.style.transform = "scale(1)")}
          >
            Recently Uploaded Threads
          </button>
        </div>

        {/* File uploaded info + Analyze */}
        {file && (
          <>
            <p style={{ marginTop: 16, color: "#fca5a5" }}>
              Uploaded: <strong>{file.name}</strong>
            </p>

            {/* Analyze button or progress */}
            {!isAnalyzing ? (
              <button
                onClick={handleAnalyze}
                style={{
                  marginTop: 12,
                  padding: "12px 20px",
                  background: "linear-gradient(90deg, #16a34a, #22c55e)",
                  color: "white",
                  border: "none",
                  borderRadius: "12px",
                  fontWeight: 700,
                  cursor: "pointer",
                  boxShadow: "0 4px 10px rgba(0,0,0,0.4)",
                  transition: "transform .15s ease",
                }}
                onMouseEnter={(e) => (e.currentTarget.style.transform = "scale(1.05)")}
                onMouseLeave={(e) => (e.currentTarget.style.transform = "scale(1)")}
              >
                Analyze Thread
              </button>
            ) : (
              <div style={{ marginTop: 16 }}>
                {/* Spinner */}
                <div
                  style={{
                    margin: "0 auto 10px",
                    width: 28,
                    height: 28,
                    borderRadius: "50%",
                    border: "3px solid rgba(255,255,255,0.2)",
                    borderTopColor: "#22c55e",
                    animation: "spin 0.9s linear infinite",
                  }}
                />
                <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>

                {/* Progress bar */}
                <div
                  style={{
                    width: 280,
                    maxWidth: "90%",
                    height: 10,
                    background: "#0f172a",
                    borderRadius: 999,
                    overflow: "hidden",
                    margin: "0 auto",
                    boxShadow: "inset 0 0 6px rgba(0,0,0,0.6)",
                  }}
                >
                  <div
                    style={{
                      width: `${progress}%`,
                      height: "100%",
                      background: "linear-gradient(90deg, #16a34a, #22c55e)",
                      transition: "width 220ms ease",
                    }}
                  />
                </div>
                <div style={{ marginTop: 8, color: "#9ca3af", fontSize: 12 }}>
                  Analyzing… {progress}%
                </div>
              </div>
            )}
          </>
        )}

        {/* Recent uploads panel */}
        {showRecent && (
          <div
            style={{
              marginTop: 16,
              padding: 12,
              borderRadius: 10,
              background: "#1f2937",
              textAlign: "left",
              maxHeight: 160,
              overflowY: "auto",
              animation: "fadeIn .3s ease",
            }}
          >
            <div style={{ fontWeight: 700, marginBottom: 8, color: "#f87171" }}>
              Recent uploads
            </div>
            {recent.length === 0 ? (
              <div style={{ color: "#9ca3af" }}>No uploads yet.</div>
            ) : (
              <ul style={{ margin: 0, paddingLeft: 18 }}>
                {recent.map((r, i) => (
                  <li key={i} style={{ marginBottom: 6, color: "#e5e7eb" }}>
                    <span style={{ fontWeight: 600 }}>{r.name}</span>
                    <span style={{ color: "#9ca3af" }}> — {r.time}</span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
