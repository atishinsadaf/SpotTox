import { useState } from "react";

/*
* Custom svg pie chart, no external chart libraried needed
*/

function CustomPieChart({ data, labels, colors, onSliceClick, hoveredIndex, onHover }) {
  const size = 320;
  const centerX = size / 2;
  const centerY = size / 2;
  const radius = 130;
  const hoverRadius = 145;

  // Calculate total and angles
  const total = data.reduce((sum, val) => sum + val, 0);
  let currentAngle = -Math.PI / 2; // Start at top

  const slices = data.map((value, index) => {
    const sliceAngle = (value / total) * 2 * Math.PI;
    const startAngle = currentAngle;
    const endAngle = currentAngle + sliceAngle;
    const midAngle = (startAngle + endAngle) / 2;

    currentAngle = endAngle;

    const isHovered = hoveredIndex === index;
    const r = isHovered ? hoverRadius : radius;

    // Calculate path
    const x1 = centerX + r * Math.cos(startAngle);
    const y1 = centerY + r * Math.sin(startAngle);
    const x2 = centerX + r * Math.cos(endAngle);
    const y2 = centerY + r * Math.sin(endAngle);

    const largeArc = sliceAngle > Math.PI ? 1 : 0;

    const pathData = [
      `M ${centerX} ${centerY}`,
      `L ${x1} ${y1}`,
      `A ${r} ${r} 0 ${largeArc} 1 ${x2} ${y2}`,
      'Z'
    ].join(' ');

    // Label position
    const labelRadius = r * 0.7;
    const labelX = centerX + labelRadius * Math.cos(midAngle);
    const labelY = centerY + labelRadius * Math.sin(midAngle);

    return {
      pathData,
      labelX,
      labelY,
      midAngle,
      value,
      percentage: ((value / total) * 100).toFixed(1)
    };
  });

  return (
    <svg width={size} height={size} style={{ display: 'block', margin: '0 auto' }}>
      {slices.map((slice, index) => (
        <g key={index}>
          <path
            d={slice.pathData}
            fill={colors[index]}
            stroke="#1f2937"
            strokeWidth="3"
            style={{
              cursor: 'pointer',
              transition: 'all 0.3s ease',
              filter: hoveredIndex === index ? 'brightness(1.2)' : 'none'
            }}
            onClick={() => onSliceClick(index)}
            onMouseEnter={() => onHover(index)}
            onMouseLeave={() => onHover(null)}
          >
            <title>{`${labels[index]}\n${slice.value.toFixed(3)} (${slice.percentage}%)`}</title>
          </path>
          
          {/* Label */}
          <text
            x={slice.labelX}
            y={slice.labelY}
            textAnchor="middle"
            dominantBaseline="middle"
            fill="white"
            fontSize="12"
            fontWeight="bold"
            style={{ pointerEvents: 'none', textShadow: '0 1px 3px rgba(0,0,0,0.8)' }}
          >
            {slice.value.toFixed(3)}
          </text>
        </g>
      ))}
    </svg>
  );
}

/*
* Enhanced pie chart with clickable slices & detail view
*/
export default function EnhancedPieChart({ results, onClose, modelName }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [hoveredIndex, setHoveredIndex] = useState(null);
  
  console.log("PieChart results:", results);

  const valid = (results || []).filter(
    (r) => r.summary && r.summary.mean != null
  );

  if (valid.length === 0) {
    return (
      <PieChartModal onClose={onClose}>
        <div style={{ textAlign: "center", padding: "40px" }}>
          <div style={{ fontSize: "48px", marginBottom: "16px" }}>⚠️</div>
          <p style={{ color: "#f87171", fontSize: "18px", fontWeight: "600" }}>
            No valid toxicity results to display
          </p>
        </div>
      </PieChartModal>
    );
  }

  const labels = valid.map((r) => r.filename);
  const data = valid.map((r) => r.summary.mean);

  // Find most toxic thread
  const mostToxicEntry = valid.reduce((max, r) =>
    r.summary.mean > max.summary.mean ? r : max
  );
  const mostToxic = mostToxicEntry?.filename;
  const mostToxicScore = mostToxicEntry?.summary.mean;

  // Color palette
  const colors = [
    "#ef4444", // red
    "#f59e0b", // amber
    "#10b981", // green
    "#3b82f6", // blue
    "#8b5cf6", // purple
    "#ec4899", // pink
    "#14b8a6", // teal
    "#f97316", // orange
  ];

  const handleSliceClick = (index) => {
    setSelectedFile(valid[index]);
  };

  return (
    <PieChartModal onClose={onClose}>
      <h2 style={{ 
        marginBottom: 20, 
        textAlign: "center",
        fontSize: "24px",
        fontWeight: "800",
        color: "#f3f4f6"
      }}>
        Multi-Thread Toxicity Comparison
      </h2>

      {modelName && (
        <div style={{
          textAlign: "center",
          marginBottom: 16,
          fontSize: 14,
          color: "#93c5fd",
          fontWeight: 600,
        }}>
          🤖 Model: {modelName}
        </div>
      )}

      <div style={{ 
        padding: "20px", 
        background: "#0f172a",
        borderRadius: "12px",
        marginBottom: "20px"
      }}>
        <CustomPieChart 
          data={data}
          labels={labels}
          colors={colors}
          onSliceClick={handleSliceClick}
          hoveredIndex={hoveredIndex}
          onHover={setHoveredIndex}
        />
      </div>

      {/* Legend */}
      <div style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
        gap: "8px",
        marginBottom: "20px",
        padding: "16px",
        background: "#1e293b",
        borderRadius: "12px"
      }}>
        {labels.map((label, index) => (
          <div
            key={index}
            style={{
              display: "flex",
              alignItems: "center",
              gap: "8px",
              padding: "8px",
              background: hoveredIndex === index ? "#334155" : "transparent",
              borderRadius: "6px",
              cursor: "pointer",
              transition: "background 0.2s ease"
            }}
            onClick={() => handleSliceClick(index)}
            onMouseEnter={() => setHoveredIndex(index)}
            onMouseLeave={() => setHoveredIndex(null)}
          >
            <div style={{
              width: "16px",
              height: "16px",
              background: colors[index],
              borderRadius: "3px",
              flexShrink: 0
            }} />
            <div style={{ fontSize: "13px", color: "#e5e7eb", fontWeight: "600" }}>
              {label}
            </div>
            <div style={{ 
              marginLeft: "auto", 
              fontSize: "12px", 
              color: "#9ca3af",
              fontWeight: "700"
            }}>
              {data[index].toFixed(3)}
            </div>
          </div>
        ))}
      </div>

      {/* Instructions */}
      <div style={{
        textAlign: "center",
        padding: "12px",
        background: "#1e293b",
        borderRadius: "8px",
        marginBottom: "20px",
        fontSize: "13px",
        color: "#cbd5e1",
        fontStyle: "italic"
      }}>
        💡 Click on any slice or legend item to view detailed results
      </div>

      {/* Most Toxic Thread Display */}
      <div style={{
        padding: "16px",
        background: "linear-gradient(135deg, #7f1d1d, #991b1b)",
        borderRadius: "12px",
        marginBottom: "20px",
        border: "2px solid #dc2626"
      }}>
        <div style={{ 
          fontSize: 14, 
          fontWeight: 700, 
          marginBottom: 8,
          color: "#fecaca",
          textTransform: "uppercase",
          letterSpacing: "0.5px"
        }}>
          🚨 Most Toxic Thread
        </div>
        <div style={{ fontSize: 18, fontWeight: 800, color: "#fff" }}>
          {mostToxic}
        </div>
        <div style={{ 
          fontSize: 16, 
          marginTop: 8,
          color: "#fca5a5"
        }}>
          Toxicity Score: <span style={{ fontWeight: "700" }}>{mostToxicScore.toFixed(3)}</span>
        </div>
      </div>

      {/* Selected File Detail View */}
      {selectedFile && (
        <div style={{
          padding: "20px",
          background: "#1e293b",
          borderRadius: "12px",
          border: "2px solid #3b82f6",
          animation: "slideIn 0.3s ease"
        }}>
          <style>{`
            @keyframes slideIn {
              from { opacity: 0; transform: translateY(-10px); }
              to { opacity: 1; transform: translateY(0); }
            }
          `}</style>
          
          <div style={{ 
            display: "flex", 
            justifyContent: "space-between", 
            alignItems: "center",
            marginBottom: 16
          }}>
            <h3 style={{ 
              margin: 0, 
              color: "#60a5fa",
              fontSize: "18px",
              fontWeight: "700"
            }}>
              📄 {selectedFile.filename}
            </h3>
            <button
              onClick={() => setSelectedFile(null)}
              style={{
                background: "#374151",
                border: "none",
                color: "#d1d5db",
                padding: "6px 12px",
                borderRadius: "6px",
                cursor: "pointer",
                fontSize: "12px",
                fontWeight: "600"
              }}
            >
              Close
            </button>
          </div>

          <div style={{ 
            display: "grid", 
            gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))",
            gap: "12px"
          }}>
            <StatCard 
              label="Mean Toxicity" 
              value={selectedFile.summary.mean.toFixed(3)}
              color="#ef4444"
            />
            
            {selectedFile.summary.p90 && (
              <StatCard 
                label="90th Percentile" 
                value={selectedFile.summary.p90.toFixed(3)}
                color="#f59e0b"
              />
            )}
            
            {selectedFile.summary.p95 && (
              <StatCard 
                label="95th Percentile" 
                value={selectedFile.summary.p95.toFixed(3)}
                color="#dc2626"
              />
            )}
            
            {selectedFile.summary.max && (
              <StatCard 
                label="Max Score" 
                value={selectedFile.summary.max.toFixed(3)}
                color="#b91c1c"
              />
            )}
          </div>

          {/* Toxicity Level Indicator */}
          <div style={{ marginTop: 16 }}>
            <ToxicityBar score={selectedFile.summary.mean} />
          </div>
        </div>
      )}

      {/* Summary Statistics */}
      <div style={{
        marginTop: "24px",
        padding: "16px",
        background: "#0f172a",
        borderRadius: "12px"
      }}>
        <h3 style={{ 
          margin: "0 0 12px 0", 
          color: "#e5e7eb",
          fontSize: "16px",
          fontWeight: "700"
        }}>
          📊 Summary Statistics
        </h3>
        <div style={{ display: "grid", gap: "8px", fontSize: "14px", color: "#cbd5e1" }}>
          <div style={{ display: "flex", justifyContent: "space-between" }}>
            <span>Total Threads:</span>
            <strong style={{ color: "#e5e7eb" }}>{valid.length}</strong>
          </div>
          <div style={{ display: "flex", justifyContent: "space-between" }}>
            <span>Average Toxicity:</span>
            <strong style={{ color: "#e5e7eb" }}>
              {(data.reduce((a, b) => a + b, 0) / data.length).toFixed(3)}
            </strong>
          </div>
          <div style={{ display: "flex", justifyContent: "space-between" }}>
            <span>Highest Score:</span>
            <strong style={{ color: "#ef4444" }}>{Math.max(...data).toFixed(3)}</strong>
          </div>
          <div style={{ display: "flex", justifyContent: "space-between" }}>
            <span>Lowest Score:</span>
            <strong style={{ color: "#10b981" }}>{Math.min(...data).toFixed(3)}</strong>
          </div>
        </div>
      </div>
    </PieChartModal>
  );
}

/*
* Helper components
*/

function PieChartModal({ children, onClose }) {
  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 9999,
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        background: "rgba(0, 0, 0, 0.8)",
        backdropFilter: "blur(4px)",
        padding: "20px",
      }}
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        style={{
          background: "linear-gradient(135deg, #1f2937 0%, #111827 100%)",
          borderRadius: 20,
          padding: 28,
          width: "min(680px, 95vw)",
          maxHeight: "90vh",
          overflowY: "auto",
          boxShadow: "0 25px 70px rgba(0,0,0,0.7)",
          color: "white",
          position: "relative",
          border: "1px solid #374151",
        }}
      >
        <button
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

        {children}
      </div>
    </div>
  );
}

function StatCard({ label, value, color }) {
  return (
    <div style={{
      padding: "12px",
      background: "#0f172a",
      borderRadius: "8px",
      border: `2px solid ${color}`,
    }}>
      <div style={{ 
        fontSize: "11px", 
        color: "#9ca3af", 
        marginBottom: "4px",
        textTransform: "uppercase",
        letterSpacing: "0.5px",
        fontWeight: "600"
      }}>
        {label}
      </div>
      <div style={{ 
        fontSize: "20px", 
        fontWeight: "800", 
        color: color 
      }}>
        {value}
      </div>
    </div>
  );
}

function ToxicityBar({ score }) {
  const percentage = score * 100;
  let color, label;
  
  if (score < 0.3) {
    color = "#10b981";
    label = "Low Risk";
  } else if (score < 0.7) {
    color = "#f59e0b";
    label = "Moderate Risk";
  } else {
    color = "#ef4444";
    label = "High Risk";
  }

  return (
    <div>
      <div style={{ 
        display: "flex", 
        justifyContent: "space-between", 
        marginBottom: "6px",
        fontSize: "12px",
        fontWeight: "600",
        color: "#cbd5e1"
      }}>
        <span>Toxicity Level</span>
        <span style={{ color }}>{label}</span>
      </div>
      <div style={{
        width: "100%",
        height: "12px",
        background: "#0f172a",
        borderRadius: "999px",
        overflow: "hidden",
        border: "1px solid #374151"
      }}>
        <div style={{
          width: `${percentage}%`,
          height: "100%",
          background: color,
          transition: "width 0.5s ease",
          borderRadius: "999px",
        }} />
      </div>
    </div>
  );
}
