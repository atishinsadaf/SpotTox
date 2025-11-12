# trainBERT.py — Train a single SpotToxBERT model

from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd, numpy as np, math, shutil, argparse, os
from pathlib import Path
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments
)
import warnings

OUT_DIR = Path("models/SpotToxBERT")
BASE_MODEL = "bert-base-uncased"
SEED = 42

def load_and_concat(csv_paths, text_col: str, label_col: str) -> pd.DataFrame:
    """Read multiple CSVs, clean, and concatenate into a single dataframe."""
    frames = []
    for p in csv_paths:
        csv_path = Path(p)
        assert csv_path.exists(), f"CSV not found: {csv_path}"
        df = pd.read_csv(csv_path)
        if text_col not in df.columns:
            raise ValueError(f"{csv_path} missing text_col '{text_col}'")
        if label_col not in df.columns:
            raise ValueError(f"{csv_path} missing label_col '{label_col}'")

        df = df.dropna(subset=[text_col, label_col]).copy()
        df["labels"] = pd.to_numeric(df[label_col], errors="coerce").clip(0, 1)
        df = df.dropna(subset=["labels"]).reset_index(drop=True)
        df["source"] = csv_path.stem  # optional debug/tracking
        frames.append(df[[text_col, "labels", "source"]])

        print(f"✓ {csv_path.name}: kept {len(df)} rows")

    if not frames:
        raise ValueError("No rows loaded from the provided CSVs.")
    all_df = pd.concat(frames, ignore_index=True)
    print(f"\n=== Total rows after concat: {len(all_df)} from {len(frames)} file(s)")
    return all_df

def make_datasets(df: pd.DataFrame, text_col: str,
                  test_size=0.10, val_size=0.10):
    """Split into train/val/test and build a HF DatasetDict."""
    # First split off test
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=SEED, shuffle=True
    )
    # Split train into train/val
    train_df, val_df = train_test_split(
        train_df, test_size=val_size, random_state=SEED, shuffle=True
    )

    dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df[[text_col, "labels"]], preserve_index=False),
        "validation": Dataset.from_pandas(val_df[[text_col, "labels"]], preserve_index=False),
        "test": Dataset.from_pandas(test_df[[text_col, "labels"]], preserve_index=False),
    })
    return dataset

def tokenize_dataset(dataset: DatasetDict, text_col: str, max_length: int):
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)

    def tok_fn(batch):
        return tok(batch[text_col], truncation=True, padding="max_length", max_length=max_length)

    dataset = dataset.map(tok_fn, batched=True, desc="Tokenizing")
    keep = ["input_ids", "token_type_ids", "attention_mask", "labels"]
    for split in dataset.keys():
        cols = dataset[split].column_names
        dataset[split] = dataset[split].remove_columns([c for c in cols if c not in keep])
    return dataset, tok

def make_training_args(epochs: int, batch: int, lr: float):
    """Compatible TrainingArguments across transformers versions."""
    try:
        return TrainingArguments(
            output_dir=str(OUT_DIR),
            learning_rate=lr,
            per_device_train_batch_size=batch,
            per_device_eval_batch_size=max(8, batch*2),
            num_train_epochs=epochs,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="rmse",
            greater_is_better=False,
            fp16=True,
            logging_steps=50,
            report_to="none",
            save_total_limit=1,
            seed=SEED,
            data_seed=SEED,
            overwrite_output_dir=True,
        )
    except TypeError:
        # For older versions
        warnings.warn("Falling back to legacy TrainingArguments")
        return TrainingArguments(
            output_dir=str(OUT_DIR),
            learning_rate=lr,
            per_device_train_batch_size=batch,
            per_device_eval_batch_size=max(8, batch*2),
            num_train_epochs=epochs,
            weight_decay=0.01,
            do_eval=True,
            logging_steps=50,
            save_steps=500,
            eval_steps=500,
            fp16=True,
            seed=SEED,
        )

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.squeeze(-1)
    rmse = math.sqrt(mean_squared_error(labels, preds))
    return {"rmse": rmse}

def train_single(df_all: pd.DataFrame, text_col: str,
                 epochs: int, batch: int, lr: float, max_length: int):
    # Build datasets
    dataset = make_datasets(df_all, text_col=text_col, test_size=0.10, val_size=0.10)
    # Tokenize
    dataset, tok = tokenize_dataset(dataset, text_col=text_col, max_length=max_length)
    # Model (regression head)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=1)
    model.config.problem_type = "regression"

    # Ensure clean single export
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    args = make_training_args(epochs=epochs, batch=batch, lr=lr)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tok,
        compute_metrics=compute_metrics,
    )

    print(f"\nTraining SpotToxBERT on {len(df_all)} total rows ...")
    trainer.train()

    print("\n Test evaluation:")
    print(trainer.evaluate(dataset["test"]))

    trainer.save_model(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    print(f"\n Saved model to ./{OUT_DIR}")

def main():
    ap = argparse.ArgumentParser(description="Train a single SpotToxBERT model from one or more CSVs.")
    ap.add_argument("--csv", nargs="+", required=True, help="CSV path(s). All are concatenated to train one model.")
    ap.add_argument("--text_col", default="txt")
    ap.add_argument("--label_col", default="offensiveness_score")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_length", type=int, default=256)
    args = ap.parse_args()

    # Load & concat
    df_all = load_and_concat(args.csv, args.text_col, args.label_col)

    # Train one model and export to ./SpotToxBERT
    train_single(
        df_all=df_all,
        text_col=args.text_col,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        max_length=args.max_length,
    )

if __name__ == "__main__":
    main()
