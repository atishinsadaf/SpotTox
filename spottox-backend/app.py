# app.py - SpotTox Backend Server
from flask import Flask, request, jsonify
from flask_cors import CORS
import os, json
from datetime import datetime
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import random

DEMO_MODE = False

# ---------------------------------------------------------------------
# Device (CPU or GPU)
# ---------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXTENSIONS = {"txt", "json", "csv"}

MODEL_PATHS = {
    "SpotToxBERT": os.path.join(BASE_DIR, "models/SpotToxBERT"),
    "SpotToxRoBERTa": os.path.join(BASE_DIR, "models/SpotToxRoBERTa"),
    "SpotToxLSTM": os.path.join(BASE_DIR, "models/SpotToxLSTM"),
    "SpotToxDistilBERT": os.path.join(BASE_DIR, "models/SpotToxDistilBERT"),

}

current_model_name = "SpotToxBERT"
current_tok, current_mdl = None, None


# ---------------------------------------------------------------------
# LSTM model architecture (6-label output)
# ---------------------------------------------------------------------
class SpotToxLSTM(nn.Module):
    def __init__(self, hidden_dim=256, num_labels=6):
        super().__init__()
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        self.lstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, num_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        with torch.no_grad():
            bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_out.last_hidden_state
        lstm_out, _ = self.lstm(sequence_output)
        pooled = torch.mean(lstm_out, dim=1)
        x = self.dropout(pooled)
        return self.sigmoid(self.fc(x))  # [batch, 6]


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

    if name == "SpotToxLSTM":
        tok = AutoTokenizer.from_pretrained(model_dir)
        mdl = SpotToxLSTM()
        mdl.load_state_dict(torch.load(os.path.join(model_dir, "SpotToxLSTM.pt"), map_location=device))
        mdl.to(device)
        mdl.eval()
    else:
        tok = AutoTokenizer.from_pretrained(model_dir)
        mdl = AutoModelForSequenceClassification.from_pretrained(model_dir)
        mdl.to(device)
        mdl.eval()

    current_model_name = name
    current_tok = tok
    current_mdl = mdl
    print(f"Model loaded: {name}")


# return full label set for LSTM
def score_texts(tok, mdl, texts, max_length=256, batch_size=32):
    results = []

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
            out = mdl(**enc)

        if hasattr(out, "logits"):
            logits = out.logits.squeeze(-1).cpu().numpy()

            if current_model_name == "SpotToxDistilBERT":
                # DistilBERT regression: offensiveness_score in [-1, 1] → map to [0, 1]
                scores = (logits + 1.0) / 2.0
                scores = np.clip(scores, 0, 1)
            else:
                # Other models (BERT/RoBERTa) – keep old behavior
                scores = np.clip(logits, 0, 1)

            results.extend(scores.tolist())
            continue

        # LSTM returns 6 logits
        logits = out.cpu().numpy()
        logits = np.clip(logits, 0, 1)

        for row in logits:
            results.append({
                "offensiveness": float(row[0]),
                "toxicity": float(row[1]),
                "severe_toxicity": float(row[2]),
                "profanity": float(row[3]),
                "threat": float(row[4]),
                "insult": float(row[5]),
            })

    return results


# ---------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------
app = Flask(__name__)
CORS(app)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load default model on startup
if DEMO_MODE:
    print(" Running in DEMO MODE - using simulated data")
else:
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
    global current_model_name

    data = request.get_json()
    name = data.get("model")

    if name not in MODEL_PATHS:
        return {"error": f"Unknown model: {name}"}, 400

    if DEMO_MODE:
        # In demo mode, just update the name without loading
        current_model_name = name
        print(f" Demo mode: Switched to {name} (simulated)")
        return {"status": "ok", "active_model": name}
    else:
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
    model_override = data.get("model")

    path = os.path.join(UPLOAD_FOLDER, filename)
    df = pd.read_csv(path)

    if text_col not in df.columns:
        return {"error": f"column {text_col} missing"}, 400

    texts = df[text_col].fillna("").astype(str).tolist()

    # DEMO MODE: Generate fake scores
    if DEMO_MODE:
        scores = np.array([random.uniform(0, 1) for _ in texts])
    else:
        if model_override and model_override != current_model_name:
            load_model(model_override)
        scores = score_texts(current_tok, current_mdl, texts)

        # LSTM returns dict of 6 labels per comment
        if isinstance(scores[0], dict):
            # Extract "toxicity" scores for histogram/stats
            toxicity_scores = np.array([s["toxicity"] for s in scores])

            return {
                "model": current_model_name,
                "mean": float(toxicity_scores.mean()),
                "p90": float(np.quantile(toxicity_scores, .9)),
                "p95": float(np.quantile(toxicity_scores, .95)),
                "max": float(toxicity_scores.max()),
                "histogram": toxicity_scores.tolist(),
                "detailed_scores": scores,  # Full 6-label data for advanced view
                "top": [
                    {"text": texts[i][:200], "score": float(toxicity_scores[i]), "details": scores[i]}
                    for i in np.argsort(-toxicity_scores)[:10]
                ]
            }

    # BERT/RoBERTa or DEMO MODE (single score per comment)
    scores = np.array(scores)
    return {
        "model": current_model_name if not DEMO_MODE else f"{current_model_name} (Demo)",
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

        # DEMO MODE: Generate fake scores
        if DEMO_MODE:
            scores = np.array([random.uniform(0, 1) for _ in texts])
            results.append({
                "file": f,
                "mean": float(scores.mean()),
                "p90": float(np.quantile(scores, .9)),
                "p95": float(np.quantile(scores, .95)),
                "max": float(scores.max()),
                "histogram": scores.tolist()
            })
        else:
            if model_name and model_name != current_model_name:
                load_model(model_name)
            scores = score_texts(current_tok, current_mdl, texts)

            # LSTM compute mean of "toxicity" for chart
            if isinstance(scores[0], dict):
                mean_tox = float(np.mean([s["toxicity"] for s in scores]))
                results.append({"file": f, "mean": mean_tox, "scores": scores})
                continue

            # BERT/RoBERTa
            scores = np.array(scores)
            results.append({
                "file": f,
                "mean": float(scores.mean()),
                "p90": float(np.quantile(scores, .9)),
                "p95": float(np.quantile(scores, .95)),
                "max": float(scores.max()),
                "histogram": scores.tolist()
            })

    return {"model": current_model_name if not DEMO_MODE else f"{current_model_name} (Demo)", "results": results}


if __name__ == "__main__":
    print("SpotTox backend live at http://localhost:5001")
    app.run(debug=True, host="0.0.0.0", port=5001)
