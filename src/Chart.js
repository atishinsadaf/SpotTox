import React from "react";

/**
 * Histogram with axis labels and bar counts
 * props:
 *   - data: { counts:number[], edges:number[] }
 */
export function Histogram({ data }) {
  if (!data) return null;
  const { counts = [], edges = [] } = data;

  const maxCount = Math.max(1, ...counts);
  const width = 380;
  const height = 200;
  const pad = 36;
  const barW = (width - pad * 2) / (counts.length || 1);

  return (
    <svg
      width={width}
      height={height + 25}
      style={{ display: "block", margin: "12px auto" }}
      aria-label="Histogram of toxicity scores"
    >
      {/* Axes */}
      <line x1={pad} y1={height - pad} x2={width - pad} y2={height - pad} stroke="#9ca3af" />
      <line x1={pad} y1={pad / 2} x2={pad} y2={height - pad} stroke="#9ca3af" />

      {/* Bars */}
      {counts.map((c, i) => {
        const h = ((c / maxCount) * (height - pad * 1.5)) || 0;
        const x = pad + i * barW + 2;
        const y = height - pad - h;

        return (
          <g key={i}>
            <rect x={x} y={y} width={barW - 4} height={h} fill="#22c55e" rx="3" />
            <text
              x={x + (barW - 4) / 2}
              y={y - 4}
              fontSize="10"
              textAnchor="middle"
              fill="#d1d5db"
            >
              {c}
            </text>
          </g>
        );
      })}

      {/* X-axis tick labels */}
      {[0, 0.25, 0.5, 0.75, 1].map((t, i) => {
        const x = pad + t * (width - pad * 2);
        return (
          <text key={i} x={x} y={height - 8} fontSize="10" textAnchor="middle" fill="#9ca3af">
            {t.toFixed(2)}
          </text>
        );
      })}

      {/* Axis titles */}
      <text x={width / 2} y={height + 18} fontSize="12" textAnchor="middle" fill="#d1d5db">
        Toxicity Score
      </text>

      <text
        x={-height / 2}
        y={14}
        transform="rotate(-90)"
        fontSize="12"
        textAnchor="middle"
        fill="#d1d5db"
      >
        Message Count
      </text>
    </svg>
  );
}

/**
 * Modal for charts
 * ✅ Shows model name in the title
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
        background: "rgba(0,0,0,0.6)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: 16,
      }}
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose?.();
      }}
    >
      <div
        style={{
          width: "min(720px, 95vw)",
          background: "#111827",
          color: "white",
          borderRadius: 16,
          boxShadow: "0 20px 60px rgba(0,0,0,0.6)",
          padding: 16,
          position: "relative",
        }}
      >
        <button
          aria-label="Close chart"
          onClick={onClose}
          style={{
            position: "absolute",
            top: 8,
            right: 10,
            background: "transparent",
            border: "none",
            color: "#9ca3af",
            fontSize: 22,
            cursor: "pointer",
          }}
        >
          ✕
        </button>

        {/* ✅ Title with model name */}
        <div style={{ fontWeight: 800, fontSize: 18, marginBottom: 12 }}>
          {title || "Histogram"}
          {modelName && (
            <span style={{ color: "#93c5fd", fontWeight: 600 }}>
              {" — "}{modelName}
            </span>
          )}
        </div>

        <div style={{ borderTop: "1px solid #374151", paddingTop: 8 }}>
          {children}
        </div>
      </div>
    </div>
  );
}
