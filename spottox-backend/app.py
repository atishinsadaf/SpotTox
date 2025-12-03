# app.py - SpotTox Backend Server
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import os, json, re
from datetime import datetime
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import random

# OCR imports for image processing
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("⚠️ pytesseract or Pillow not installed. Image upload will be limited.")

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
ALLOWED_EXTENSIONS = {"txt", "json", "csv", "png", "jpg", "jpeg", "gif", "webp"}

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
        return self.sigmoid(self.fc(x))


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
                scores = (logits + 1.0) / 2.0
                scores = np.clip(scores, 0, 1)
            else:
                scores = np.clip(logits, 0, 1)

            results.extend(scores.tolist())
            continue

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
# Demo mode helper - generates fake scores
# ---------------------------------------------------------------------
def demo_score_texts(texts):
    """Generate random toxicity scores for demo mode"""
    results = []
    for text in texts:
        # Generate slightly realistic scores based on text length and some keywords
        base_score = random.uniform(0.05, 0.35)
        
        # Check for potentially toxic keywords ( basic simulation)
        toxic_words = ['stupid', 'hate', 'awful', 'terrible', 'idiot', 'dumb', 'worst', 'suck', 'bad']
        text_lower = text.lower()
        for word in toxic_words:
            if word in text_lower:
                base_score += random.uniform(0.15, 0.3)
        
        # Cap at 1.0
        score = min(base_score, 0.98)
        results.append(score)
    
    return results


# ---------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------
app = Flask(__name__)
CORS(app)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

if DEMO_MODE:
    print("🎮 Running in DEMO MODE - using simulated toxicity scores")
else:
    load_model(current_model_name)


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------
@app.get("/")
def home():
    return {"message": "SpotTox backend", "active_model": current_model_name, "demo_mode": DEMO_MODE}


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
        current_model_name = name
        return {"status": "ok", "active_model": name}

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

    if DEMO_MODE:
        scores = np.array(demo_score_texts(texts))
    else:
        if model_override and model_override != current_model_name:
            load_model(model_override)
        scores = score_texts(current_tok, current_mdl, texts)

        if isinstance(scores[0], dict):
            toxicity_scores = np.array([s["toxicity"] for s in scores])
            return {
                "model": current_model_name,
                "mean": float(toxicity_scores.mean()),
                "p90": float(np.quantile(toxicity_scores, .9)),
                "p95": float(np.quantile(toxicity_scores, .95)),
                "max": float(toxicity_scores.max()),
                "histogram": toxicity_scores.tolist(),
                "detailed_scores": scores,
                "top": [
                    {"text": texts[i][:200], "score": float(toxicity_scores[i]), "details": scores[i]}
                    for i in np.argsort(-toxicity_scores)[:10]
                ]
            }

    scores = np.array(scores)
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

        if DEMO_MODE:
            scores = np.array(demo_score_texts(texts))
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

            if isinstance(scores[0], dict):
                mean_tox = float(np.mean([s["toxicity"] for s in scores]))
                results.append({"file": f, "mean": mean_tox, "scores": scores})
                continue

            scores = np.array(scores)
            results.append({
                "file": f,
                "mean": float(scores.mean()),
                "p90": float(np.quantile(scores, .9)),
                "p95": float(np.quantile(scores, .95)),
                "max": float(scores.max()),
                "histogram": scores.tolist()
            })

    return {"model": current_model_name, "results": results}


@app.post("/analyze_chat")
def analyze_chat():
    data = request.get_json()
    text = data.get("text", "")
    model_override = data.get("model", current_model_name)

    if not text.strip():
        return {"error": "empty text"}, 400

    # Handle DEMO_MODE properly
    if DEMO_MODE:
        scores = demo_score_texts([text])
        score = float(scores[0])
        return {"model": current_model_name, "score": score}

    if model_override and model_override != current_model_name:
        load_model(model_override)

    scores = score_texts(current_tok, current_mdl, [text])

    if isinstance(scores[0], dict):
        toxicity = float(scores[0]["toxicity"])
        return {
            "model": current_model_name,
            "toxicity": toxicity,
            "details": scores[0]
        }

    score = float(scores[0])
    return {"model": current_model_name, "score": score}


@app.post("/search_thread")
def search_thread():
    data = request.get_json()
    thread_id = str(data.get("thread_id", "")).strip()

    if not thread_id:
        return jsonify({"found": False, "error": "No thread_id provided"}), 400

    # Find CSV files
    csv_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith(".csv")]
    if not csv_files:
        return jsonify({"found": False, "error": "No uploaded dataset"}), 400

    # choose the most recently uploaded CSV, not alphabetically
    latest_file = max(
        csv_files,
        key=lambda f: os.path.getmtime(os.path.join(UPLOAD_FOLDER, f))
    )

    print("SEARCHING THIS FILE:", latest_file)

    path = os.path.join(UPLOAD_FOLDER, latest_file)
    df = pd.read_csv(path)

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace("\ufeff", "")
    )

    print("COLUMNS:", df.columns.tolist())

    # Determine ID column
    if "thread_id" in df.columns:
        id_col = "thread_id"
    elif "post_id" in df.columns:
        id_col = "post_id"
    else:
        return jsonify({"found": False, "error": "No thread_id or post_id column"}), 400

    # Determine text column
    if "text" in df.columns:
        text_col = "text"
    elif "txt" in df.columns:
        text_col = "txt"
    else:
        return jsonify({"found": False, "error": "No text or txt column"}), 400

    rows = df[df[id_col].astype(str) == thread_id]

    if rows.empty:
        return jsonify({"found": False, "rows": []})

    texts = rows[text_col].fillna("").astype(str).tolist()

    # Handle DEMO_MODE
    if DEMO_MODE:
        toxicity = np.array(demo_score_texts(texts))
    else:
        scores = score_texts(current_tok, current_mdl, texts)
        if isinstance(scores[0], dict):
            toxicity = np.array([s["toxicity"] for s in scores])
        else:
            toxicity = np.array(scores)

    return jsonify({
        "found": True,
        "thread_id": thread_id,
        "count": len(toxicity),
        "mean": float(toxicity.mean()),
        "p90": float(np.quantile(toxicity, 0.9)),
        "p95": float(np.quantile(toxicity, 0.95)),
        "max": float(toxicity.max()),
    })

@app.post("/analyze_reddit")
def analyze_reddit():
    data = request.get_json()
    url = data.get("url", "").strip()
    model_override = data.get("model", current_model_name)

    if not url:
        return {"error": "No URL provided"}, 400

    # Extract Reddit thread ID
    try:
        parts = url.split("/comments/")
        thread_part = parts[1].split("/")[0]
        thread_id = thread_part.strip()
    except:
        return {"error": "Invalid Reddit URL"}, 400

    # Fetch Reddit JSON
    json_url = f"https://www.reddit.com/comments/{thread_id}.json"

    headers = {"User-agent": "SpotToxBot/1.0"}
    res = requests.get(json_url, headers=headers)

    if res.status_code != 200:
        return {"error": "Reddit API error"}, 400

    try:
        data = res.json()
    except:
        return {"error": "Could not parse Reddit JSON"}, 400

    # Extract all comments
    comments = []
    try:
        listing = data[1]["data"]["children"]
        for c in listing:
            if c["kind"] == "t1":  # normal comment
                txt = c["data"].get("body", "")
                if txt.strip():
                    comments.append(txt)
    except:
        return {"error": "Failed extracting comments"}, 400

    if not comments:
        return {"error": "No comments found in thread"}, 400

    # Handle DEMO_MODE
    if DEMO_MODE:
        toxicity = np.array(demo_score_texts(comments))
    else:
        # Switch model if needed
        if model_override != current_model_name:
            load_model(model_override)

        # Score them
        scores = score_texts(current_tok, current_mdl, comments)

        # Handle LSTM dict output
        if isinstance(scores[0], dict):
            toxicity = np.array([s["toxicity"] for s in scores])
        else:
            toxicity = np.array(scores)

    return {
        "model": current_model_name,
        "count": len(toxicity),
        "mean": float(toxicity.mean()),
        "p90": float(np.quantile(toxicity, 0.9)),
        "p95": float(np.quantile(toxicity, 0.95)),
        "max": float(toxicity.max()),
        "histogram": toxicity.tolist(),
        "top": [
            {"text": comments[i][:200], "score": float(toxicity[i])}
            for i in np.argsort(-toxicity)[:10]
        ]
    }


# ---------------------------------------------------------------------
# Image Upload & OCR Analysis
# ---------------------------------------------------------------------
def extract_text_from_image(image_path):
    """Extract text from an image using OCR"""
    if not OCR_AVAILABLE:
        return None, "OCR not available. Install pytesseract and Pillow."
    
    try:
        img = Image.open(image_path)
        # Use pytesseract to extract text
        text = pytesseract.image_to_string(img)
        return text, None
    except Exception as e:
        return None, str(e)


def parse_conversation_from_text(raw_text):
    """Parse extracted text into individual messages"""
    messages = []
    
    # Split by newlines and filter empty lines
    lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
    
    # Try to identify message patterns
    # Common patterns: "Username: message" or just separate lines
    for line in lines:
        # Skip very short lines (likely noise)
        if len(line) < 3:
            continue
        
        # Check for "username: message" pattern
        if ':' in line:
            parts = line.split(':', 1)
            if len(parts) == 2 and len(parts[0]) < 30:  # Username likely < 30 chars
                messages.append({
                    "user": parts[0].strip(),
                    "text": parts[1].strip()
                })
                continue
        
        # Otherwise treat whole line as a message
        messages.append({
            "user": "Unknown",
            "text": line
        })
    
    return messages


@app.post("/upload-image")
def upload_image():
    """Handle image upload for OCR processing"""
    if "file" not in request.files:
        return {"error": "No file provided"}, 400
    
    f = request.files["file"]
    if not f.filename:
        return {"error": "No filename"}, 400
    
    # Check if it's an image
    ext = f.filename.rsplit(".", 1)[-1].lower()
    if ext not in {"png", "jpg", "jpeg", "gif", "webp"}:
        return {"error": "Invalid image format. Use PNG, JPG, GIF, or WEBP."}, 400
    
    # Save the image
    path = os.path.join(UPLOAD_FOLDER, f.filename)
    f.save(path)
    
    return {"filename": f.filename, "type": "image"}


@app.post("/analyze-image")
def analyze_image():
    """Extract text from image and analyze toxicity"""
    data = request.get_json()
    filename = data.get("filename")
    model_override = data.get("model", current_model_name)
    
    if not filename:
        return {"error": "No filename provided"}, 400
    
    path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(path):
        return {"error": "File not found"}, 400
    
    # Check if OCR is available
    if not OCR_AVAILABLE:
        return {
            "error": "OCR not available. Please install pytesseract and Pillow.",
            "install_instructions": "Run: brew install tesseract && pip install pytesseract Pillow"
        }, 400
    
    # Extract text from image
    raw_text, error = extract_text_from_image(path)
    if error:
        return {"error": f"OCR failed: {error}"}, 400
    
    if not raw_text or not raw_text.strip():
        return {"error": "No text found in image"}, 400
    
    # Parse into messages
    messages = parse_conversation_from_text(raw_text)
    
    if not messages:
        return {"error": "Could not parse any messages from image"}, 400
    
    # Extract just the text for scoring
    texts = [m["text"] for m in messages if m["text"]]
    
    if not texts:
        return {"error": "No valid text to analyze"}, 400
    
    # Score the texts
    if DEMO_MODE:
        scores = demo_score_texts(texts)
    else:
        if model_override and model_override != current_model_name:
            load_model(model_override)
        scores = score_texts(current_tok, current_mdl, texts)
        
        # Handle LSTM dict output
        if isinstance(scores[0], dict):
            scores = [s["toxicity"] for s in scores]
    
    scores = np.array(scores)
    
    # Combine messages with scores
    analyzed_messages = []
    for i, msg in enumerate(messages):
        if i < len(scores):
            analyzed_messages.append({
                "user": msg["user"],
                "text": msg["text"],
                "score": float(scores[i])
            })
    
    # Sort by toxicity for top flagged
    sorted_msgs = sorted(analyzed_messages, key=lambda x: x["score"], reverse=True)
    
    return {
        "model": current_model_name,
        "message_count": len(analyzed_messages),
        "thread_count": 1,
        "raw_text": raw_text[:500],  # First 500 chars for preview
        "mean": float(scores.mean()),
        "p90": float(np.quantile(scores, 0.9)),
        "p95": float(np.quantile(scores, 0.95)),
        "max": float(scores.max()),
        "histogram": scores.tolist(),
        "messages": analyzed_messages,
        "top": sorted_msgs[:10]
    }


if __name__ == "__main__":
    print("🚀 SpotTox backend live at http://localhost:5001")
    if OCR_AVAILABLE:
        print("✅ OCR is available for image processing")
    else:
        print("⚠️ OCR not available - install with: brew install tesseract && pip install pytesseract Pillow")
    app.run(debug=True, host="0.0.0.0", port=5001)