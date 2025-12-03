import React, { useState } from "react";

const BASE_URL = "http://127.0.0.1:5001";

export default function ChatUI({ model, models, onModelChange }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // Helper function to get toxicity verdict
  const getToxicityVerdict = (score) => {
    if (score < 0.3) return { label: "Safe", color: "#22c55e", emoji: "✅", bg: "#052e16" };
    if (score < 0.5) return { label: "Low Risk", color: "#84cc16", emoji: "⚠️", bg: "#1a2e05" };
    if (score < 0.7) return { label: "Moderate", color: "#f59e0b", emoji: "⚠️", bg: "#451a03" };
    if (score < 0.85) return { label: "Toxic", color: "#ef4444", emoji: "🚨", bg: "#450a0a" };
    return { label: "Highly Toxic", color: "#dc2626", emoji: "🚫", bg: "#450a0a" };
  };

  const handleSend = async () => {
    if (!input.trim() || isAnalyzing) return;

    const userMessage = input.trim();
    setInput("");
    setIsAnalyzing(true);

    // Add user message immediately
    const newMessage = {
      id: Date.now(),
      text: userMessage,
      type: "user",
      score: null,
      analyzing: true,
    };
    
    setMessages((prev) => [...prev, newMessage]);

    try {
      const res = await fetch(`${BASE_URL}/analyze_chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: userMessage,
          model: model,
        }),
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.error || "Analysis failed");
      }

      // Update the message with score
      const score = data.score ?? data.toxicity ?? 0;
      
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === newMessage.id
            ? { ...msg, score, analyzing: false }
            : msg
        )
      );
    } catch (err) {
      // Update message to show error
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === newMessage.id
            ? { ...msg, error: err.message, analyzing: false }
            : msg
        )
      );
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const clearChat = () => {
    setMessages([]);
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      {/* Header with model selector */}
      <div style={{ 
        display: "flex", 
        justifyContent: "space-between", 
        alignItems: "center",
        marginBottom: 12,
        paddingBottom: 12,
        borderBottom: "1px solid #374151"
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ color: "#9ca3af", fontSize: 13 }}>Model:</span>
          <select
            value={model}
            onChange={(e) => onModelChange(e.target.value)}
            style={{
              padding: "6px 10px",
              background: "#111827",
              color: "white",
              borderRadius: 6,
              border: "1px solid #374151",
              fontSize: 13,
              cursor: "pointer",
            }}
          >
            {models?.map((m) => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
        </div>
        
        <button
          onClick={clearChat}
          style={{
            padding: "6px 12px",
            background: "#374151",
            color: "#d1d5db",
            border: "none",
            borderRadius: 6,
            fontSize: 12,
            cursor: "pointer",
          }}
        >
          🗑️ Clear
        </button>
      </div>

      {/* Messages area */}
      <div
        style={{
          flex: 1,
          overflowY: "auto",
          marginBottom: 12,
          padding: 8,
          background: "#0f172a",
          borderRadius: 8,
          minHeight: 200,
          maxHeight: 350,
        }}
      >
        {messages.length === 0 ? (
          <div style={{ 
            textAlign: "center", 
            color: "#6b7280", 
            padding: 40,
            fontSize: 14
          }}>
            <div style={{ fontSize: 32, marginBottom: 12 }}>💬</div>
            <p>Type a message to check its toxicity in real-time.</p>
            <p style={{ fontSize: 12, marginTop: 8, color: "#4b5563" }}>
              Each message will be analyzed by the {model} model.
            </p>
          </div>
        ) : (
          messages.map((msg) => {
            const verdict = msg.score !== null ? getToxicityVerdict(msg.score) : null;
            
            return (
              <div
                key={msg.id}
                style={{
                  marginBottom: 12,
                  padding: 12,
                  background: verdict ? verdict.bg : "#1f2937",
                  borderRadius: 10,
                  borderLeft: `4px solid ${verdict ? verdict.color : "#374151"}`,
                }}
              >
                {/* Message text */}
                <div style={{ 
                  color: "#e5e7eb", 
                  marginBottom: 8,
                  wordBreak: "break-word",
                  whiteSpace: "pre-wrap"
                }}>
                  {msg.text}
                </div>
                
                {/* Score/Status bar */}
                <div style={{ 
                  display: "flex", 
                  alignItems: "center", 
                  justifyContent: "space-between",
                  fontSize: 13
                }}>
                  {msg.analyzing ? (
                    <div style={{ color: "#9ca3af", display: "flex", alignItems: "center", gap: 6 }}>
                      <span className="analyzing-spinner">⏳</span>
                      <span>Analyzing...</span>
                    </div>
                  ) : msg.error ? (
                    <div style={{ color: "#ef4444" }}>
                      ❌ Error: {msg.error}
                    </div>
                  ) : verdict ? (
                    <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                      <span style={{ 
                        color: verdict.color, 
                        fontWeight: 700,
                        display: "flex",
                        alignItems: "center",
                        gap: 4
                      }}>
                        {verdict.emoji} {verdict.label}
                      </span>
                      <span style={{ color: "#9ca3af" }}>
                        Score: <strong style={{ color: "white" }}>{(msg.score * 100).toFixed(1)}%</strong>
                      </span>
                    </div>
                  ) : null}
                  
                  {/* Timestamp */}
                  <span style={{ color: "#4b5563", fontSize: 11 }}>
                    {new Date(msg.id).toLocaleTimeString()}
                  </span>
                </div>

                {/* Score bar visualization */}
                {verdict && (
                  <div style={{ marginTop: 8 }}>
                    <div style={{
                      width: "100%",
                      height: 6,
                      background: "#1f2937",
                      borderRadius: 3,
                      overflow: "hidden"
                    }}>
                      <div style={{
                        width: `${msg.score * 100}%`,
                        height: "100%",
                        background: verdict.color,
                        transition: "width 0.5s ease",
                        borderRadius: 3,
                      }} />
                    </div>
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>

      {/* Input area */}
      <div style={{ display: "flex", gap: 8 }}>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Type a message to check toxicity..."
          disabled={isAnalyzing}
          style={{
            flex: 1,
            padding: "10px 12px",
            background: "#111827",
            color: "white",
            border: "1px solid #374151",
            borderRadius: 8,
            resize: "none",
            height: 60,
            fontSize: 14,
            fontFamily: "inherit",
          }}
        />
        <button
          onClick={handleSend}
          disabled={!input.trim() || isAnalyzing}
          style={{
            padding: "0 20px",
            background: input.trim() && !isAnalyzing
              ? "linear-gradient(90deg, #ef4444, #f97316)"
              : "#374151",
            color: "white",
            border: "none",
            borderRadius: 8,
            fontWeight: 700,
            cursor: input.trim() && !isAnalyzing ? "pointer" : "not-allowed",
            fontSize: 14,
            transition: "background 0.2s ease",
          }}
        >
          {isAnalyzing ? "..." : "Check"}
        </button>
      </div>

      {/* Legend */}
      <div style={{ 
        marginTop: 12, 
        padding: 10, 
        background: "#111827", 
        borderRadius: 8,
        fontSize: 11,
        color: "#9ca3af"
      }}>
        <div style={{ fontWeight: 600, marginBottom: 6 }}>Toxicity Levels:</div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 12 }}>
          <span>✅ Safe (0-30%)</span>
          <span>⚠️ Low (30-50%)</span>
          <span>⚠️ Moderate (50-70%)</span>
          <span>🚨 Toxic (70-85%)</span>
          <span>🚫 Highly Toxic (85%+)</span>
        </div>
      </div>

      <style>{`
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
        .analyzing-spinner {
          display: inline-block;
          animation: spin 1s linear infinite;
        }
      `}</style>
    </div>
  );
}