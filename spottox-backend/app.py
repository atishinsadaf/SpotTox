# app.py - SpotTox Backend Server
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
from datetime import datetime
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXTENSIONS = {"txt", "json", "csv"}
CANONICAL_MODEL_NAME = "SpotToxBERT"
CANONICAL_MODEL_DIR = os.path.join(BASE_DIR, CANONICAL_MODEL_NAME)

# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def resolve_model_dir(name_or_path):
    if not name_or_path or name_or_path == CANONICAL_MODEL_NAME:
        return CANONICAL_MODEL_DIR
    candidate = name_or_path
    if not os.path.isabs(candidate):
        candidate = os.path.join(BASE_DIR, candidate)
    return os.path.abspath(candidate)

def model_dir_has_model(model_dir):
    return os.path.isdir(model_dir) and os.path.exists(os.path.join(model_dir, "config.json"))

def load_spottoxbert(model_dir):
    tok = AutoTokenizer.from_pretrained(model_dir)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_dir)
    mdl.eval()
    return tok, mdl

def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items() if hasattr(v, "to")}

def score_texts(tok, mdl, texts, max_length=256, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdl.to(device)
    scores = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        enc = tok(chunk, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
        enc = to_device(enc, device)
        with torch.no_grad():
            logits = mdl(**enc).logits.squeeze(-1).detach().cpu().numpy()
        scores.extend(np.clip(logits, 0.0, 1.0))
    return np.array(scores, dtype=float)

def read_thread_file(file_path):
    try:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                content = json.load(f)
            return {
                "success": True,
                "file_type": "json",
                "content": content if isinstance(content, (dict, list)) else str(content)[:5000],
                "message_count": len(content) if isinstance(content, list) else 1,
                "timestamp": datetime.now().isoformat(),
                "file_size_bytes": os.path.getsize(file_path),
            }
        elif ext == ".txt":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.read().splitlines()
            return {
                "success": True,
                "file_type": "txt",
                "content": lines[:200],
                "message_count": len(lines),
                "timestamp": datetime.now().isoformat(),
                "file_size_bytes": os.path.getsize(file_path),
            }
        elif ext == ".csv":
            df_head = pd.read_csv(file_path, nrows=10)
            preview_rows = (
                df_head.replace({np.nan: None})
                .astype(object)
                .where(pd.notnull(df_head), None)
                .to_dict(orient="records")
            )
            return {
                "success": True,
                "file_type": "csv",
                "columns": list(df_head.columns),
                "preview_rows": preview_rows,
                "message_count": len(df_head),
                "timestamp": datetime.now().isoformat(),
                "file_size_bytes": os.path.getsize(file_path),
            }
        return {"success": False, "error": f"Unsupported file type: {ext}", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return {"success": False, "error": str(e), "timestamp": datetime.now().isoformat()}

# ---------------------------------------------------------------------
# Flask app initialization
# ---------------------------------------------------------------------
app = Flask(__name__)
CORS(
    app,
    resources={r"/*": {"origins": ["http://localhost:3000", "http://127.0.0.1:3000",
                                   "http://localhost:5173", "http://127.0.0.1:5173"]}},
    supports_credentials=False,
    max_age=86400,
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------------------------------------------------------------
# Model preload
# ---------------------------------------------------------------------
default_tok, default_mdl = None, None
if model_dir_has_model(CANONICAL_MODEL_DIR):
    try:
        default_tok, default_mdl = load_spottoxbert(CANONICAL_MODEL_DIR)
        print(f"Loaded canonical model: {CANONICAL_MODEL_DIR}")
    except Exception as e:
        print(f"Could not load canonical model: {e}")
else:
    print(f"Canonical model not found at: {CANONICAL_MODEL_DIR}")

# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------
@app.get("/")
def home():
    return jsonify({
        "message": "SpotTox Backend API - Thread Toxicity Detection System",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "GET /": "API information",
            "GET /health": "Health check",
            "POST /upload": "Upload file",
            "POST /upload-multiple": "Upload multiple files",
            "GET /threads": "List uploaded files",
            "GET /models": "List models",
            "POST /analyze-text": "Analyze single text",
            "POST /analyze-file": "Analyze one CSV",
            "POST /analyze-multiple": "Analyze multiple CSVs side-by-side",
        }
    })

@app.get("/health")
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "SpotTox Backend",
        "has_model": model_dir_has_model(CANONICAL_MODEL_DIR),
    })

# ---------------------------------------------------------------------
# Upload routes
# ---------------------------------------------------------------------
@app.post("/upload")
def upload_thread():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type", "allowed": list(ALLOWED_EXTENSIONS)}), 400
    try:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        result = read_thread_file(filepath)
        return jsonify({"filename": file.filename, "file_info": result}), 200
    except Exception as e:
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500

@app.post("/upload-multiple")
def upload_multiple():
    if "files" not in request.files:
        return jsonify({"error": "No files provided"}), 400
    files = request.files.getlist("files")
    uploaded, failed = [], []
    for file in files:
        if file.filename == "":
            continue
        if not allowed_file(file.filename):
            failed.append({"filename": file.filename, "error": "Invalid file type"})
            continue
        try:
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)
            result = read_thread_file(path)
            if result.get("success"):
                uploaded.append({"filename": file.filename, "file_info": result})
            else:
                failed.append({"filename": file.filename, "error": result.get("error")})
        except Exception as e:
            failed.append({"filename": file.filename, "error": str(e)})
    return jsonify({
        "message": f"{len(uploaded)} file(s) uploaded successfully",
        "uploaded": uploaded,
        "failed": failed,
        "timestamp": datetime.now().isoformat(),
    }), (200 if uploaded else 400)

# ---------------------------------------------------------------------
# Data listing routes
# ---------------------------------------------------------------------
@app.get("/threads")
def list_threads():
    try:
        files = []
        for fname in os.listdir(UPLOAD_FOLDER):
            if allowed_file(fname):
                fpath = os.path.join(UPLOAD_FOLDER, fname)
                st = os.stat(fpath)
                files.append({
                    "filename": fname,
                    "size_kb": round(st.st_size / 1024, 2),
                    "uploaded": datetime.fromtimestamp(st.st_mtime).isoformat(),
                    "file_type": fname.split(".")[-1],
                })
        return jsonify({"threads": sorted(files, key=lambda x: x["uploaded"], reverse=True)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.get("/models")
def list_models():
    if model_dir_has_model(CANONICAL_MODEL_DIR):
        return jsonify({"models": [CANONICAL_MODEL_NAME], "default": CANONICAL_MODEL_NAME})
    return jsonify({"models": [], "default": None})

# ---------------------------------------------------------------------
# Analysis routes
# ---------------------------------------------------------------------
@app.post("/analyze-file")
def analyze_file():
    try:
        data = request.get_json(force=True)
        filename = data.get("filename")
        text_col = data.get("text_col", "txt")
        model_dir = data.get("model_dir")
        if not filename:
            return jsonify({"error": "Missing 'filename'"}), 400
        path = os.path.join(UPLOAD_FOLDER, filename)
        if not os.path.exists(path):
            return jsonify({"error": f"File not found: {filename}"}), 404
        if not filename.endswith(".csv"):
            return jsonify({"error": "Only CSV supported"}), 400
        tok, mdl = (load_spottoxbert(model_dir) if model_dir else (default_tok, default_mdl))
        df = pd.read_csv(path)
        if text_col not in df.columns:
            return jsonify({"error": f"Column '{text_col}' not found"}), 400
        texts = df[text_col].fillna("").astype(str).tolist()
        scores = score_texts(tok, mdl, texts)
        mean, p90, p95, maxv = map(float, [scores.mean(), np.quantile(scores, 0.9), np.quantile(scores, 0.95), scores.max()])
        bins = np.linspace(0, 1, 11)
        hist, edges = np.histogram(scores, bins=bins)
        top_k = np.argsort(-scores)[:10]
        flagged = [{"row": int(i), "text": texts[i][:200], "score": float(scores[i])} for i in top_k]
        return jsonify({
            "filename": filename,
            "summary": {"mean": mean, "p90": p90, "p95": p95, "max": maxv},
            "histogram": {"counts": hist.tolist(), "edges": edges.tolist()},
            "top_flagged": flagged,
            "timestamp": datetime.now().isoformat(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.post("/analyze-multiple")
def analyze_multiple():
    try:
        data = request.get_json(force=True)
        files = data.get("filenames", [])
        text_col = data.get("text_col", "txt")
        model_dir = data.get("model_dir")
        if not files:
            return jsonify({"error": "Missing 'filenames' list"}), 400
        tok, mdl = (load_spottoxbert(model_dir) if model_dir else (default_tok, default_mdl))
        results = []
        for fname in files:
            path = os.path.join(UPLOAD_FOLDER, fname)
            if not os.path.exists(path):
                results.append({"filename": fname, "error": "File not found"})
                continue
            try:
                df = pd.read_csv(path)
                if text_col not in df.columns:
                    results.append({"filename": fname, "error": f"Missing column '{text_col}'"})
                    continue
                texts = df[text_col].fillna("").astype(str).tolist()
                scores = score_texts(tok, mdl, texts)
                mean, p90, p95, maxv = map(float, [scores.mean(), np.quantile(scores, 0.9), np.quantile(scores, 0.95), scores.max()])
                bins = np.linspace(0, 1, 11)
                hist, edges = np.histogram(scores, bins=bins)
                results.append({
                    "filename": fname,
                    "rows": len(texts),
                    "summary": {"mean": mean, "p90": p90, "p95": p95, "max": maxv},
                    "histogram": {"counts": hist.tolist(), "edges": edges.tolist()},
                })
            except Exception as e:
                results.append({"filename": fname, "error": str(e)})
        return jsonify({"results": results, "timestamp": datetime.now().isoformat()}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.post("/analyze-text")
def analyze_text():
    try:
        data = request.get_json(force=True)
        text = (data.get("text") or "").strip()
        if not text:
            return jsonify({"error": "Missing 'text'"}), 400
        tok, mdl = (default_tok, default_mdl)
        scores = score_texts(tok, mdl, [text])
        return jsonify({"text": text, "toxicity_score": float(scores[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Starting SpotTox Backend Server")
    print("=" * 60)
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Allowed file types: {ALLOWED_EXTENSIONS}")
    print(f"Canonical model: {CANONICAL_MODEL_DIR}")
    print(f"Running on: http://localhost:5001")
    print(f"Health check: http://localhost:5001/health")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=5001)
