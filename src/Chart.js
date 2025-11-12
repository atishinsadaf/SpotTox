import { useState } from "react";

/**
 * Enhanced Interactive Histogram with color-coded bars and click functionality
 * 
 * Features:
 * - Color-coded bars (green: low toxicity, yellow: medium, red: high)
 * - Interactive hover effects with visual feedback
 * - Clickable bars that trigger onBinClick callback
 * - Grid lines for easier value reading
 * - Y-axis labels showing message counts
 * - X-axis labels showing toxicity scores (0.0 - 1.0)
 * - Built-in legend explaining color scheme
 * - Tooltips on hover showing range and count
 * 
 * Props:
 *   - data: { counts: number[], edges: number[] }
 *     - counts: array of message counts per bin
 *     - edges: array of toxicity score boundaries (length = counts.length + 1)
 *   - onBinClick: (info: { bin: number, count: number, from: number, to: number }) => void (optional)
 *     - Callback function triggered when a bar is clicked
 *     - Receives bin index, count, and score range (from/to)
 */
export function Histogram({ data, onBinClick }) {
  const [hoveredBin, setHoveredBin] = useState(null);
  
  if (!data) return null;
  const { counts = [], edges = [] } = data;

  const maxCount = Math.max(1, ...counts);
  const width = 420;
  const height = 240;
  const pad = 50;
  const barW = (width - pad * 2) / (counts.length || 1);

  // Color gradient based on toxicity level
  const getBarColor = (index, isHovered) => {
    const position = index / counts.length;
    if (isHovered) {
      return position < 0.3 ? "#34d399" : position < 0.7 ? "#fbbf24" : "#f87171";
    }
    return position < 0.3 ? "#10b981" : position < 0.7 ? "#f59e0b" : "#ef4444";
  };

  return (
    <div style={{ position: "relative" }}>
      <svg
        width={width}
        height={height + 40}
        style={{ display: "block", margin: "12px auto" }}
        aria-label="Interactive histogram of toxicity scores"
      >
        {/* Background grid lines */}
        {[0, 0.25, 0.5, 0.75, 1].map((t, i) => {
          const y = height - pad - (t * (height - pad * 1.5));
          return (
            <line
              key={`grid-${i}`}
              x1={pad}
              y1={y}
              x2={width - pad}
              y2={y}
              stroke="#374151"
              strokeWidth="1"
              strokeDasharray="4,4"
              opacity="0.3"
            />
          );
        })}

        {/* Axes */}
        <line x1={pad} y1={height - pad} x2={width - pad} y2={height - pad} stroke="#9ca3af" strokeWidth="2" />
        <line x1={pad} y1={pad / 2} x2={pad} y2={height - pad} stroke="#9ca3af" strokeWidth="2" />

        {/* Bars with hover effects */}
        {counts.map((c, i) => {
          const h = ((c / maxCount) * (height - pad * 1.5)) || 0;
          const x = pad + i * barW + 2;
          const y = height - pad - h;
          const isHovered = hoveredBin === i;

          return (
            <g key={i}>
              <rect
                x={x}
                y={y}
                width={barW - 4}
                height={h}
                fill={getBarColor(i, isHovered)}
                rx="4"
                style={{
                  cursor: "pointer",
                  transition: "all 0.2s ease",
                  filter: isHovered ? "brightness(1.2)" : "none",
                }}
                onMouseEnter={() => setHoveredBin(i)}
                onMouseLeave={() => setHoveredBin(null)}
                onClick={() => {
                  if (onBinClick) {
                    const from = edges[i] || (i / counts.length);
                    const to = edges[i + 1] || ((i + 1) / counts.length);
                    onBinClick({ bin: i, count: c, from, to });
                  }
                }}
              >
                <title>{`Score: ${(edges[i] || 0).toFixed(2)} - ${(edges[i + 1] || 1).toFixed(2)}\nCount: ${c}`}</title>
              </rect>
              
              {/* Count labels */}
              {c > 0 && (
                <text
                  x={x + (barW - 4) / 2}
                  y={y - 6}
                  fontSize="11"
                  fontWeight="600"
                  textAnchor="middle"
                  fill={isHovered ? "#fff" : "#d1d5db"}
                  style={{ pointerEvents: "none" }}
                >
                  {c}
                </text>
              )}
            </g>
          );
        })}

        {/* X-axis labels */}
        {[0, 0.25, 0.5, 0.75, 1].map((t, i) => {
          const x = pad + t * (width - pad * 2);
          return (
            <text key={i} x={x} y={height - 18} fontSize="11" fontWeight="500" textAnchor="middle" fill="#9ca3af">
              {t.toFixed(2)}
            </text>
          );
        })}

        {/* Y-axis labels */}
        {[0, 0.25, 0.5, 0.75, 1].map((t, i) => {
          const y = height - pad - (t * (height - pad * 1.5));
          const value = Math.round(t * maxCount);
          return (
            <text key={i} x={pad - 8} y={y + 4} fontSize="11" fontWeight="500" textAnchor="end" fill="#9ca3af">
              {value}
            </text>
          );
        })}

        {/* Axis titles */}
        <text x={width / 2} y={height + 28} fontSize="13" fontWeight="600" textAnchor="middle" fill="#e5e7eb">
          Toxicity Score (0 = Safe, 1 = Highly Toxic)
        </text>

        <text
          x={-height / 2}
          y={16}
          transform="rotate(-90)"
          fontSize="13"
          fontWeight="600"
          textAnchor="middle"
          fill="#e5e7eb"
        >
          Message Count
        </text>
      </svg>

      {/* Legend */}
      <div style={{ textAlign: "center", marginTop: "16px", fontSize: "12px", color: "#9ca3af" }}>
        <div style={{ display: "flex", justifyContent: "center", gap: "16px", flexWrap: "wrap" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <div style={{ width: "16px", height: "16px", background: "#10b981", borderRadius: "3px" }} />
            <span>Low (0.0-0.3)</span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <div style={{ width: "16px", height: "16px", background: "#f59e0b", borderRadius: "3px" }} />
            <span>Medium (0.3-0.7)</span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <div style={{ width: "16px", height: "16px", background: "#ef4444", borderRadius: "3px" }} />
            <span>High (0.7-1.0)</span>
          </div>
        </div>
        <div style={{ marginTop: "8px", fontStyle: "italic" }}>
          💡 Click on any bar for details • Hover for quick info
        </div>
      </div>
    </div>
  );
}

/**
 Chart Modal
 */
export function ChartModal({ open, onClose, title, modelName, children }) {
  if (!open) return null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={title || "Chart"}
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 9999,
        background: "rgba(0,0,0,0.75)",
        backdropFilter: "blur(4px)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: 16,
        animation: "fadeIn 0.2s ease",
      }}
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose?.();
      }}
    >
      <style>{`
        @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        @keyframes slideUp {
          from { transform: translateY(20px); opacity: 0; }
          to { transform: translateY(0); opacity: 1; }
        }
      `}</style>
      
      <div
        style={{
          width: "min(780px, 95vw)",
          background: "linear-gradient(135deg, #1f2937 0%, #111827 100%)",
          color: "white",
          borderRadius: 20,
          boxShadow: "0 25px 70px rgba(0,0,0,0.7)",
          padding: 24,
          position: "relative",
          border: "1px solid #374151",
          animation: "slideUp 0.3s ease",
        }}
      >
        <button
          aria-label="Close chart"
          onClick={onClose}
          style={{
            position: "absolute",
            top: 16,
            right: 16,
            background: "#374151",
            border: "none",
            color: "#d1d5db",
            fontSize: 20,
            cursor: "pointer",
            width: "36px",
            height: "36px",
            borderRadius: "50%",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            transition: "all 0.2s ease",
          }}
          onMouseOver={(e) => {
            e.currentTarget.style.background = "#4b5563";
            e.currentTarget.style.color = "#fff";
          }}
          onMouseOut={(e) => {
            e.currentTarget.style.background = "#374151";
            e.currentTarget.style.color = "#d1d5db";
          }}
        >
          ✕
        </button>

        <div style={{ marginBottom: 20 }}>
          <h2 style={{ fontWeight: 800, fontSize: 22, margin: 0, color: "#f3f4f6" }}>
            {title || "Analysis Results"}
          </h2>
          {modelName && (
            <div style={{ 
              marginTop: 8, 
              fontSize: 14, 
              color: "#93c5fd", 
              fontWeight: 600,
              display: "flex",
              alignItems: "center",
              gap: "8px"
            }}>
              <span style={{ fontSize: "16px" }}>🤖</span>
              <span>Model: {modelName}</span>
            </div>
          )}
        </div>

        <div style={{ 
          borderTop: "1px solid #374151", 
          paddingTop: 16,
          background: "#0f172a",
          borderRadius: "12px",
          padding: "16px"
        }}>
          {children}
        </div>
      </div>
    </div>
  );
}

/**
 Demo Compoenent
 */
export default function ChartDemo() {
  const [showHistogram, setShowHistogram] = useState(false);
  const [binDetail, setBinDetail] = useState(null);

  // Sample data
  const histogramData = {
    counts: [45, 32, 28, 15, 12, 8, 5, 3, 2, 1],
    edges: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
  };

  return (
    <div style={{ 
      minHeight: "100vh", 
      background: "linear-gradient(135deg, #000000, #111827)",
      padding: "40px 20px",
      fontFamily: "system-ui, -apple-system, sans-serif"
    }}>
      <div style={{ maxWidth: "900px", margin: "0 auto" }}>
        <h1 style={{ 
          color: "white", 
          textAlign: "center", 
          marginBottom: "40px",
          fontSize: "36px",
          fontWeight: "900"
        }}>
          Enhanced SpotTox Charts 📊
        </h1>

        <div style={{ 
          background: "#1f2937", 
          padding: "30px", 
          borderRadius: "16px",
          marginBottom: "24px"
        }}>
          <h2 style={{ color: "#f3f4f6", marginTop: 0 }}>Interactive Histogram</h2>
          <p style={{ color: "#9ca3af", marginBottom: "20px" }}>
            Click on bars for details, hover for quick info. Color-coded by toxicity level.
          </p>
          
          <Histogram 
            data={histogramData}
            onBinClick={(info) => {
              setBinDetail(info);
              console.log("Bin clicked:", info);
            }}
          />

          {binDetail && (
            <div style={{
              marginTop: "20px",
              padding: "16px",
              background: "#111827",
              borderRadius: "12px",
              border: "2px solid #3b82f6",
              color: "#e5e7eb"
            }}>
              <div style={{ fontWeight: "700", marginBottom: "8px", color: "#60a5fa" }}>
                📍 Selected Range
              </div>
              <div>Score Range: {binDetail.from.toFixed(3)} – {binDetail.to.toFixed(3)}</div>
              <div>Message Count: {binDetail.count}</div>
            </div>
          )}

          <button
            onClick={() => setShowHistogram(true)}
            style={{
              marginTop: "20px",
              padding: "12px 24px",
              background: "linear-gradient(90deg, #3b82f6, #2563eb)",
              color: "white",
              border: "none",
              borderRadius: "10px",
              fontWeight: "700",
              cursor: "pointer",
              width: "100%",
              fontSize: "16px"
            }}
          >
            View in Modal
          </button>
        </div>

        <div style={{
          background: "#1f2937",
          padding: "30px",
          borderRadius: "16px",
          color: "#e5e7eb"
        }}>
          <h2 style={{ color: "#f3f4f6", marginTop: 0 }}>Features Added ✨</h2>
          <ul style={{ lineHeight: "1.8" }}>
            <li><strong>Interactive bars:</strong> Click to see detailed range information</li>
            <li><strong>Hover effects:</strong> Instant visual feedback and tooltips</li>
            <li><strong>Color coding:</strong> Green (safe), yellow (moderate), red (toxic)</li>
            <li><strong>Grid lines:</strong> Easier to read values</li>
            <li><strong>Axis labels:</strong> Clear Y-axis count indicators</li>
            <li><strong>Legend:</strong> Explains color scheme</li>
            <li><strong>Enhanced modal:</strong> Better styling with gradient background</li>
            <li><strong>Smooth animations:</strong> Professional look and feel</li>
          </ul>
        </div>
      </div>

      <ChartModal
        open={showHistogram}
        onClose={() => {
          setShowHistogram(false);
          setBinDetail(null);
        }}
        title="Toxicity Score Distribution"
        modelName="SpotToxBERT"
      >
        <Histogram 
          data={histogramData}
          onBinClick={(info) => setBinDetail(info)}
        />
        {binDetail && (
          <div style={{ 
            textAlign: "center", 
            color: "#93c5fd", 
            marginTop: 16,
            padding: "12px",
            background: "#1f2937",
            borderRadius: "8px",
            fontWeight: "600"
          }}>
            Selected: {binDetail.from.toFixed(3)} – {binDetail.to.toFixed(3)} • Count: {binDetail.count}
          </div>
        )}
      </ChartModal>
    </div>
  );
}
