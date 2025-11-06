# trainLSTM.py — SpotTox LSTM model (6 labels)
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, BertTokenizer

# ------------------------------------------------------------
# Tokenizer helper for backend scoring (also used in app.py)
# ------------------------------------------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def preprocess_text(text, max_length=256):
    enc = tokenizer.encode(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length
    )
    return np.array(enc)
# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "models", "SpotToxLSTM")
os.makedirs(SAVE_DIR, exist_ok=True)

MODEL_NAME = "bert-base-uncased"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
EPOCHS = 3
LR = 2e-5
MAX_LEN = 128

# ------------------------------------------------------------
# Load CSVs and merge
# ------------------------------------------------------------
files = [
    os.path.join("data", "ruddit_with_text_conv_0.csv"),
    os.path.join("data", "ruddit_with_text_conv_1.csv"),
    os.path.join("data", "ruddit_with_text_conv_2.csv"),
]
dfs = []
for f in files:
    df_temp = pd.read_csv(f)
    df_temp.columns = df_temp.columns.str.strip()
    dfs.append(df_temp)

df = pd.concat(dfs, ignore_index=True)

# Columns to use
text_col = "txt"
label_cols = [
    "offensiveness_score",
    "Toxicity",
    "Severe Toxicity",
    "Profanity",
    "Threat",
    "Insult",
]

# Filter and clean
df = df[[text_col] + label_cols].dropna()
df[label_cols] = df[label_cols].astype(float)

# Normalize all columns to 0–1 range safely
for c in label_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
    min_val, max_val = df[c].min(), df[c].max()
    if max_val > 1 or min_val < 0:
        df[c] = (df[c] - min_val) / (max_val - min_val)

print(f"Loaded {len(df)} samples with labels: {label_cols}")

# ------------------------------------------------------------
# Tokenizer and Dataset
# ------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class ToxicDataset(Dataset):
    def __init__(self, df):
        self.texts = df[text_col].tolist()
        self.labels = df[label_cols].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

dataset = ToxicDataset(df)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ------------------------------------------------------------
# Model Definition: BERT + LSTM
# ------------------------------------------------------------
class SpotToxLSTM(nn.Module):
    def __init__(self, hidden_dim=256, num_labels=len(label_cols)):
        super().__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
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

    def forward(self, input_ids, attention_mask, labels=None):
        # Freeze BERT for efficiency
        with torch.no_grad():
            bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_out.last_hidden_state
        lstm_out, _ = self.lstm(sequence_output)
        pooled = torch.mean(lstm_out, dim=1)
        x = self.dropout(pooled)
        out = self.fc(x)
        return self.sigmoid(out)

# ------------------------------------------------------------
# Training Loop
# ------------------------------------------------------------
model = SpotToxLSTM().to(DEVICE)
criterion = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")

# ------------------------------------------------------------
# Save Model and Tokenizer
# ------------------------------------------------------------
torch.save(model.state_dict(), os.path.join(SAVE_DIR, "SpotToxLSTM.pt"))
tokenizer.save_pretrained(SAVE_DIR)
print("Model and tokenizer saved to", SAVE_DIR)
print("Trained on labels:", label_cols)
