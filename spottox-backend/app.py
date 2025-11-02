# app.py - SpotTox Backend Server (BERT + RoBERTa multi-model)
from flask import Flask, request, jsonify
from flask_cors import CORS
import os, json
from datetime import datetime
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXTENSIONS = {"txt", "json", "csv"}

MODEL_PATHS = {
    "SpotToxBERT": os.path.join(BASE_DIR, "models/SpotToxBERT"),
    "SpotToxRoBERTa": os.path.join(BASE_DIR, "models/SpotToxRoBERTa"),
}

current_model_name = "SpotToxBERT"
current_tok, current_mdl = None, None


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model(name):
    global current_model_name, current_tok, current_mdl

    if name not in MODEL_PATHS:
        raise ValueError(f"Unknown model: {name}")

    model_dir = MODEL_PATHS[name]
    if not os.path.isdir(model_dir):
        raise ValueError(f"Model directory not found: {model_dir}")

    print(f"Switching to model: {name}")

    tok = AutoTokenizer.from_pretrained(model_dir)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_dir)
    mdl.to("cpu")
    mdl.eval()

    current_model_name = name
    current_tok = tok
    current_mdl = mdl
    print(f"Model loaded: {name}")


def score_texts(tok, mdl, texts, max_length=256, batch_size=32):
    device = torch.device("cpu")
    scores = []
    for i in range(0, len(texts), batch_size):
        enc = tok(
            texts[i:i + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = mdl(**enc).logits.squeeze(-1).cpu().numpy()
        scores.extend(np.clip(logits, 0, 1))
    return np.array(scores)


# ---------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------
app = Flask(__name__)
CORS(app)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load default model on startup
load_model(current_model_name)


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------
@app.get("/")
def home():
    return {"message": "SpotTox backend", "active_model": current_model_name}


@app.get("/models")
def list_models():
    return {"available": list(MODEL_PATHS.keys()), "active": current_model_name}


@app.post("/set_model")
def set_model():
    data = request.get_json()
    name = data.get("model")
    try:
        load_model(name)
        return {"status": "ok", "active_model": name}
    except Exception as e:
        return {"error": str(e)}, 400


@app.post("/upload")
def upload_file():
    if "file" not in request.files:
        return {"error": "no file"}, 400
    f = request.files["file"]
    if not f.filename or not allowed_file(f.filename):
        return {"error": "invalid file"}, 400

    path = os.path.join(UPLOAD_FOLDER, f.filename)
    f.save(path)
    return {"filename": f.filename}

@app.post("/upload-multiple")
def upload_multiple():
    files = request.files.getlist("files")
    uploaded = []
    for f in files:
        if not allowed_file(f.filename):
            continue
        path = os.path.join(UPLOAD_FOLDER, f.filename)
        f.save(path)
        uploaded.append({"filename": f.filename})
    return {"uploaded": uploaded}, 200


@app.post("/analyze-file")
def analyze_file():
    data = request.get_json()
    filename = data.get("filename")
    text_col = data.get("text_col", "txt")
    model_override = data.get("model")  #user-selected model

    if model_override and model_override != current_model_name:
        load_model(model_override)

    path = os.path.join(UPLOAD_FOLDER, filename)
    df = pd.read_csv(path)

    if text_col not in df.columns:
        return {"error": f"column {text_col} missing"}, 400

    texts = df[text_col].fillna("").astype(str).tolist()
    scores = score_texts(current_tok, current_mdl, texts)

    return {
        "model": current_model_name,
        "mean": float(scores.mean()),
        "p90": float(np.quantile(scores, .9)),
        "p95": float(np.quantile(scores, .95)),
        "max": float(scores.max()),
        "histogram": scores.tolist(),
        "top": [
            {"text": texts[i][:200], "score": float(scores[i])}
            for i in np.argsort(-scores)[:10]
        ]
    }


@app.post("/analyze-multiple")
def analyze_multiple():
    data = request.get_json()
    files = data.get("filenames", [])
    text_col = data.get("text_col", "txt")
    model_name = data.get("model")

    if model_name and model_name != current_model_name:
        load_model(model_name)

    results = []
    for f in files:
        path = os.path.join(UPLOAD_FOLDER, f)
        if not os.path.exists(path):
            results.append({"file": f, "error": "not found"})
            continue

        df = pd.read_csv(path)
        if text_col not in df.columns:
            results.append({"file": f, "error": f"column {text_col} missing"})
            continue

        texts = df[text_col].fillna("").astype(str).tolist()
        scores = score_texts(current_tok, current_mdl, texts)

        results.append({
            "file": f,
            "mean": float(scores.mean()),
            "p90": float(np.quantile(scores, .9)),
            "p95": float(np.quantile(scores, .95)),
            "max": float(scores.max()),
            "histogram": scores.tolist()
        })
    return {"model": current_model_name, "results": results}


if __name__ == "__main__":
    print("SpotTox backend live at http://localhost:5001")
    app.run(debug=True, host="0.0.0.0", port=5001)
