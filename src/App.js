import { useState, useEffect, useMemo } from "react";

export default function App() {
  const [now, setNow] = useState(() => new Date().toLocaleString());
  const [files, setFiles] = useState([]); // Changed to array for multiple files
  const [selectedModel, setSelectedModel] = useState("SpotToxBERT");
  
  // Recent uploads
  const [recent, setRecent] = useState([]);
  const [showRecent, setShowRecent] = useState(false);

  // Available AI models
  const availableModels = [
    { id: "SpotToxBERT", name: "SpotTox BERT Model" },
    { id: "perspective-api", name: "Google Perspective API" },
    { id: "custom-nlp", name: "Custom NLP Model" }
  ];

  // Tagline list
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
    const speed = isDeleting ? 35 : 55;
    const donePause = 1000;

    const tick = setTimeout(() => {
      if (!isDeleting) {
        const next = full.slice(0, typed.length + 1);
        setTyped(next);
        if (next === full) {
          setTimeout(() => setIsDeleting(true), donePause);
        }
      } else {
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

  // Handle multiple file upload
  function handleUpload(e) {
    const selectedFiles = Array.from(e.target.files);
    if (selectedFiles.length > 0) {
      setFiles(selectedFiles);
      
      // Add to recent uploads
      const newUploads = selectedFiles.map(file => ({
        name: file.name,
        time: new Date().toLocaleString(),
        size: (file.size / 1024).toFixed(2) + " KB"
      }));
      
      setRecent((prev) => [...newUploads, ...prev].slice(0, 10));
    }
  }

  // Remove individual file
  function removeFile(index) {
    setFiles(files.filter((_, i) => i !== index));
  }

  // Simulated analyze with progress bar
  function handleAnalyze() {
    if (files.length === 0) {
      alert("Please upload at least one CSV file!");
      return;
    }
    
    setIsAnalyzing(true);
    setProgress(0);
    
    const interval = setInterval(() => {
      setProgress((p) => {
        const inc = Math.random() * 15 + 5;
        const next = Math.min(100, Math.round(p + inc));
        if (next >= 100) {
          clearInterval(interval);
          setTimeout(() => {
            setIsAnalyzing(false);
            alert(
              `Analysis complete!\n\nFiles analyzed: ${files.length}\nModel used: ${selectedModel}\n\n(Backend integration pending)`
            );
          }, 350);
        }
        return next;
      });
    }, 400);
  }

  function handleCardMouseMove(e) {
    const rect = e.currentTarget.getBoundingClientRect();
    const cx = rect.left + rect.width / 2;
    const cy = rect.top + rect.height / 2;
    const dx = (e.clientX - cx) / (rect.width / 2);
    const dy = (e.clientY - cy) / (rect.height / 2);
    setTilt({ x: dy * 6, y: -dx * 6 });
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
        minHeight: "100vh",
        overflow: "auto",
        background: "linear-gradient(135deg, #000000, #111827, #1f2937, #000000)",
        padding: "20px",
      }}
    >
      <style>{`
        @keyframes fadeIn { 
          from { opacity: 0; transform: translateY(6px);} 
          to { opacity: 1; transform: translateY(0);} 
        }
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
        @keyframes spin { 
          to { transform: rotate(360deg); } 
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
          maxWidth: 600,
          width: "100%",
          transform: `perspective(800px) rotateX(${tilt.x}deg) rotateY(${tilt.y}deg)`,
          transition: "transform 120ms ease",
          boxShadow: "0 12px 30px rgba(0,0,0,0.6)",
          animation: "fadeIn 0.8s ease, pulseShadow 3.5s ease-in-out infinite",
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

        {/* Model Selection Dropdown */}
        <div style={{ marginBottom: 20 }}>
          <label style={{ display: "block", marginBottom: 8, color: "#e5e7eb", fontWeight: 600 }}>
            Select AI Model:
          </label>
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            style={{
              padding: "10px 16px",
              background: "#1f2937",
              color: "white",
              border: "2px solid #374151",
              borderRadius: "8px",
              fontSize: "14px",
              fontWeight: 600,
              cursor: "pointer",
              width: "100%",
              maxWidth: "300px",
            }}
          >
            {availableModels.map((model) => (
              <option key={model.id} value={model.id}>
                {model.name}
              </option>
            ))}
          </select>
        </div>

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
            Upload Thread(s)
            <input
              type="file"
              accept=".csv,.txt,.json"
              multiple
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
            Recent Uploads
          </button>
        </div>

        {/* Display uploaded files */}
        {files.length > 0 && (
          <div
            style={{
              marginTop: 20,
              padding: 16,
              background: "#1f2937",
              borderRadius: 12,
              textAlign: "left",
            }}
          >
            <h3 style={{ color: "#f87171", marginTop: 0, marginBottom: 12, fontSize: 16 }}>
              Uploaded Files ({files.length})
            </h3>
            <div style={{ maxHeight: 200, overflowY: "auto" }}>
              {files.map((file, index) => (
                <div
                  key={index}
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    padding: "8px 12px",
                    background: "#111827",
                    borderRadius: 8,
                    marginBottom: 8,
                  }}
                >
                  <div>
                    <div style={{ fontWeight: 600, color: "#e5e7eb" }}>{file.name}</div>
                    <div style={{ fontSize: 12, color: "#9ca3af" }}>
                      {(file.size / 1024).toFixed(2)} KB
                    </div>
                  </div>
                  <button
                    onClick={() => removeFile(index)}
                    style={{
                      padding: "6px 12px",
                      background: "#dc2626",
                      color: "white",
                      border: "none",
                      borderRadius: 6,
                      cursor: "pointer",
                      fontSize: 12,
                      fontWeight: 600,
                    }}
                  >
                    Remove
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Analyze button or progress */}
        {files.length > 0 && !isAnalyzing && (
          <button
            onClick={handleAnalyze}
            style={{
              marginTop: 16,
              padding: "14px 24px",
              background: "linear-gradient(90deg, #16a34a, #22c55e)",
              color: "white",
              border: "none",
              borderRadius: "12px",
              fontWeight: 700,
              cursor: "pointer",
              boxShadow: "0 4px 10px rgba(0,0,0,0.4)",
              transition: "transform .15s ease",
              fontSize: 16,
            }}
            onMouseEnter={(e) => (e.currentTarget.style.transform = "scale(1.05)")}
            onMouseLeave={(e) => (e.currentTarget.style.transform = "scale(1)")}
          >
            Analyze {files.length} Thread{files.length > 1 ? "s" : ""} with {selectedModel}
          </button>
        )}

        {isAnalyzing && (
          <div style={{ marginTop: 16 }}>
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
            <div
              style={{
                width: "100%",
                maxWidth: 300,
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
              Analyzing {files.length} file{files.length > 1 ? "s" : ""}... {progress}%
            </div>
          </div>
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
              maxHeight: 200,
              overflowY: "auto",
              animation: "fadeIn .3s ease",
            }}
          >
            <div style={{ fontWeight: 700, marginBottom: 8, color: "#f87171" }}>
              Recent Uploads
            </div>
            {recent.length === 0 ? (
              <div style={{ color: "#9ca3af" }}>No uploads yet.</div>
            ) : (
              <div>
                {recent.map((r, i) => (
                  <div key={i} style={{ marginBottom: 6, color: "#e5e7eb", fontSize: 13 }}>
                    <span style={{ fontWeight: 600 }}>{r.name}</span>
                    <div style={{ color: "#9ca3af", fontSize: 11 }}>
                      {r.time} • {r.size}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}