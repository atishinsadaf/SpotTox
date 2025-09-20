import { useState, useEffect } from "react";

export default function App() {
  const [count, setCount] = useState(0);
  const [now, setNow] = useState(() => new Date().toLocaleString());
  const [file, setFile] = useState(null);

  useEffect(() => {
    const t = setInterval(() => setNow(new Date().toLocaleString()), 1000);
    return () => clearInterval(t);
  }, []);

  function handleUpload(e) {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      alert(`Selected file: ${selectedFile.name}`);
      // later for backend
    }
  }

  return (
    <div
      style={{
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        height: "100vh",
        background: "linear-gradient(135deg, #4f46e5, #06b6d4)",
      }}
    >
      <main
        style={{
          fontFamily: "system-ui, sans-serif",
          padding: 40,
          borderRadius: 16,
          background: "white",
          boxShadow: "0 12px 30px rgba(0,0,0,0.2)",
          textAlign: "center",
          maxWidth: 400,
          width: "90%",
          animation: "fadeIn 1s ease",
        }}
      >
        <h1 style={{ color: "#1e293b", marginBottom: 12 }}>
          Welcome to <span style={{ color: "#2563eb" }}>SpotTox</span>!
        </h1>

        <p style={{ marginTop: 8, color: "#64748b", fontStyle: "italic" }}>
          AI-powered early warning detection for toxic conversations
        </p>

        <p style={{ color: "#475569", marginBottom: 24 }}>
          Time: <strong>{now}</strong>
        </p>

        {/* Upload Thread button */}
        <div style={{ marginTop: 20 }}>
          <label
            style={{
              display: "inline-block",
              padding: "12px 20px",
              background: "linear-gradient(90deg, #2563eb, #06b6d4)",
              color: "white",
              borderRadius: "12px",
              cursor: "pointer",
              fontWeight: "600",
              boxShadow: "0 4px 10px rgba(0,0,0,0.15)",
              transition: "all 0.2s ease-in-out",
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.transform = "scale(1.05)";
              e.currentTarget.style.boxShadow = "0 6px 14px rgba(0,0,0,0.2)";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.transform = "scale(1)";
              e.currentTarget.style.boxShadow = "0 4px 10px rgba(0,0,0,0.15)";
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
        </div>

        {file && (
          <p style={{ marginTop: 16, color: "#0f172a" }}>
            ✅ Uploaded: <strong>{file.name}</strong>
          </p>
        )}
      </main>
    </div>
  );
}
