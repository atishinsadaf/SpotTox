import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)


# -----------------------------
# Dataset class
# -----------------------------
class CommentsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length: int = 128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = float(self.labels[idx])

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.float)
        return item


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_and_concat(csv_paths, text_col: str, label_col: str) -> pd.DataFrame:

    dfs = []
    print(" Loading CSV files...")
    for p in csv_paths:
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")

        print(f"  -> {path}")
        df = pd.read_csv(path)

        # Check that required columns exist
        if text_col not in df.columns:
            raise ValueError(f"Column '{text_col}' not found in {path}")
        if label_col not in df.columns:
            raise ValueError(f"Column '{label_col}' not found in {path}")

        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    print(f" Combined dataset shape: {df_all.shape}")

    # Keep only necessary columns
    df_all = df_all[[text_col, label_col]].copy()
    df_all = df_all.dropna()

    # Ensure label is numeric
    df_all[label_col] = pd.to_numeric(df_all[label_col], errors="coerce")
    df_all = df_all.dropna(subset=[label_col])

    print(f" After cleaning: {df_all.shape[0]} rows")
    return df_all


def create_dataloaders(df, text_col, label_col, tokenizer, max_length, batch_size, train_ratio=0.8, val_ratio=0.1):
    """
    Split dataframe into train / val / test and create DataLoaders.
    """
    texts = df[text_col].tolist()
    labels = df[label_col].tolist()

    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, train_size=train_ratio, random_state=42
    )

    # Split temp into val + test
    val_size = val_ratio / (1.0 - train_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, train_size=val_size, random_state=42
    )

    print(f" Split sizes -> train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

    train_dataset = CommentsDataset(X_train, y_train, tokenizer, max_length)
    val_dataset = CommentsDataset(X_val, y_val, tokenizer, max_length)
    test_dataset = CommentsDataset(X_test, y_test, tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def evaluate_regression(model, data_loader, device):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            logits = outputs.logits.squeeze(-1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(logits.cpu().numpy())

    all_labels = np.array(all_labels, dtype=float)
    all_preds = np.array(all_preds, dtype=float)

    mse = mean_squared_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    # Pearson correlation (if variance > 0)
    if np.std(all_labels) > 0 and np.std(all_preds) > 0:
        corr = np.corrcoef(all_labels, all_preds)[0, 1]
    else:
        corr = float("nan")

    return mse, mae, corr


# -----------------------------
# Training loop
# -----------------------------
def train(
    model,
    train_loader,
    val_loader,
    device,
    epochs: int,
    learning_rate: float,
    warmup_ratio: float,
    output_dir: Path,
):
    num_training_steps = epochs * len(train_loader)
    num_warmup_steps = int(warmup_ratio * num_training_steps)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    model.to(device)

    print(" Starting training...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            loss.backward()
            total_loss += loss.item()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if step % 100 == 0 or step == 1:
                avg_loss_so_far = total_loss / step
                print(f"  Epoch {epoch} | Step {step}/{len(train_loader)} | Loss: {avg_loss_so_far:.4f}")

        avg_train_loss = total_loss / len(train_loader)
        mse, mae, corr = evaluate_regression(model, val_loader, device)
        print(f"\n Epoch {epoch} done!")
        print(f"   Train loss: {avg_train_loss:.4f}")
        print(f"   Val MSE:    {mse:.4f}")
        print(f"   Val MAE:    {mae:.4f}")
        print(f"   Val Corr:   {corr:.4f}\n")

    # Save final model
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f" Saving model to {output_dir} ...")
    model.save_pretrained(output_dir)
    print(" Model saved.")

    return model


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Train DistilBERT regression model for SpotTox.")

    parser.add_argument(
        "--csv",
        nargs="+",
        required=True,
        help="List of CSV files to load. Example: --csv data/ruddit_with_text_conv_0.csv data/ruddit_with_text_conv_1.csv",
    )
    parser.add_argument(
        "--text_col",
        type=str,
        default="txt",  # matches your CSV
        help="Name of the text column in the CSV.",
    )
    parser.add_argument(
        "--label_col",
        type=str,
        default="offensiveness_score",  # matches your CSV (you can change to 'Toxicity' if you want)
        help="Name of the label column in the CSV.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="distilbert-base-uncased",
        help="HuggingFace model name.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/SpotToxDistilBERT",
        help="Where to save the trained model.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Max sequence length.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Warmup ratio for learning rate scheduler.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    csv_paths = args.csv
    text_col = args.text_col
    label_col = args.label_col

    df_all = load_and_concat(csv_paths, text_col, label_col)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Using device: {device}")

    # Tokenizer & model
    tokenizer = DistilBertTokenizerFast.from_pretrained(args.model_name)
    model = DistilBertForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=1,              # regression
        problem_type="regression", # tells HF to use MSE loss
    )

    # Dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        df_all,
        text_col=text_col,
        label_col=label_col,
        tokenizer=tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    output_dir = Path(args.output_dir)

    # Train
    model = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        output_dir=output_dir,
    )

    # Final test evaluation
    print("Evaluating on test set...")
    mse, mae, corr = evaluate_regression(model, test_loader, device)
    print(f"Test MSE:  {mse:.4f}")
    print(f"Test MAE:  {mae:.4f}")
    print(f"Test Corr: {corr:.4f}")

    # Save tokenizer + some metadata for backend to use
    tokenizer.save_pretrained(output_dir)
    meta = {
        "text_col": text_col,
        "label_col": label_col,
        "model_name": args.model_name,
        "max_length": args.max_length,
    }
    meta_path = output_dir / "spottox_metadata.json"
    import json
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f" Saved metadata to {meta_path}")


if __name__ == "__main__":
   main()
