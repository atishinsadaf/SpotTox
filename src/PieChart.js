import React from "react";
import { Pie } from "react-chartjs-2";
import {
  Chart as ChartJS,
  ArcElement,
  Tooltip,
  Legend
} from "chart.js";
import ChartDataLabels from "chartjs-plugin-datalabels";

// ✅ Prevent duplicate plugin registration (fixes _listened error)
if (!ChartJS.registry.plugins.get("datalabels")) {
  ChartJS.register(ArcElement, Tooltip, Legend, ChartDataLabels);
}

export default function PieChart({ results, onClose }) {
  console.log("PieChart results:", results);

  const valid = (results || []).filter(
    (r) => r.summary && r.summary.mean != null
  );

  const labels = valid.map((r) => r.filename);
  const data = valid.map((r) => r.summary.mean);

  // ✅ Compute most toxic thread (filename)
  const mostToxicEntry =
    valid.length > 0
      ? valid.reduce((max, r) =>
          r.summary.mean > max.summary.mean ? r : max
        )
      : null;

  const mostToxic = mostToxicEntry?.filename;
  const mostToxicScore = mostToxicEntry?.summary.mean;

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 9999,
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        pointerEvents: "auto",
      }}
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        style={{
          background: "#0f172a",
          borderRadius: 14,
          padding: 22,
          width: "520px",
          maxWidth: "90vw",
          boxShadow: "0 12px 35px rgba(0,0,0,0.6)",
          color: "white",
          position: "relative",
        }}
      >
        <button
          onClick={onClose}
          style={{
            position: "absolute",
            top: 10,
            right: 10,
            border: "none",
            background: "transparent",
            fontSize: 22,
            cursor: "pointer",
            color: "#94a3b8",
          }}
        >
          ✕
        </button>

        <h2 style={{ marginBottom: 15, textAlign: "center" }}>
          Multi-Thread Toxicity Comparison
        </h2>

        {valid.length === 0 ? (
          <p style={{ color: "#f87171", textAlign: "center" }}>
            No valid toxicity results to display
          </p>
        ) : (
          <>
            <Pie
              data={{
                labels,
                datasets: [
                  {
                    data,
                    backgroundColor: [
                      "#ef4444",
                      "#3b82f6",
                      "#22c55e",
                      "#f97316",
                      "#a855f7",
                    ],
                  },
                ],
              }}
              options={{
                plugins: {
                  datalabels: {
                    color: "#fff",
                    font: {
                      weight: "bold",
                      size: 12,
                    },
                    formatter: (value, ctx) => {
                      const label = ctx.chart.data.labels[ctx.dataIndex];
                      return `${label}\n${value.toFixed(3)}`;
                    },
                    textAlign: "center",
                    anchor: "center",
                    align: "center",
                  },
                  legend: {
                    labels: {
                      color: "#fff",
                    },
                  },
                },
              }}
            />

            {/* ✅ Most Toxic Thread Display */}
            {mostToxic && (
              <div
                style={{
                  marginTop: 18,
                  fontSize: 15,
                  textAlign: "center",
                  fontWeight: "bold",
                }}
              >
                Most Toxic Thread:{" "}
                <span style={{ color: "#f87171" }}>{mostToxic}</span> with a toxicity rating of:{" "}
                <span style={{ color: "#f87171" }}>
                  {mostToxicScore.toFixed(3)}
                </span>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
