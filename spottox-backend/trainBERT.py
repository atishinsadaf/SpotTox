from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import pandas as pd, numpy as np, math
from pathlib import Path
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments
)
from sklearn.metrics import mean_squared_error
import argparse

def train_one(csv_path: Path, text_col: str, label_col: str,
              epochs: int, batch: int, lr: float, max_length: int):
    assert csv_path.exists(), f"CSV not found: {csv_path}"
    out_dir = f"SpotToxBERT_{csv_path.stem}"

    # load
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[text_col, label_col]).copy()
    df["labels"] = pd.to_numeric(df[label_col], errors="coerce").clip(0, 1)
    df = df.dropna(subset=["labels"]).reset_index(drop=True)
    print(f"\n=== [{csv_path.name}] rows: {len(df)} -> saving to: ./{out_dir}")

    # split
    train_df, test_df = train_test_split(df, test_size=0.10, random_state=42, shuffle=True)
    train_df, val_df  = train_test_split(train_df, test_size=0.1111, random_state=42, shuffle=True)

    # HF datasets
    dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df[[text_col, "labels"]], preserve_index=False),
        "validation": Dataset.from_pandas(val_df[[text_col, "labels"]], preserve_index=False),
        "test": Dataset.from_pandas(test_df[[text_col, "labels"]], preserve_index=False),
    })

    # tokenize
    model_name = "bert-base-uncased"
    tok = AutoTokenizer.from_pretrained(model_name)

    def tok_fn(batch):
        return tok(batch[text_col], truncation=True, padding="max_length", max_length=max_length)

    dataset = dataset.map(tok_fn, batched=True)
    keep = ["input_ids","token_type_ids","attention_mask","labels"]
    dataset = dataset.remove_columns([c for c in dataset["train"].column_names if c not in keep])

    # model (regression)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    model.config.problem_type = "regression"

    # args
    def make_args():
        try:
            return TrainingArguments(
                output_dir=out_dir,
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
                report_to="none"
            )
        except TypeError:
            return TrainingArguments(
                output_dir=out_dir,
                learning_rate=lr,
                per_device_train_batch_size=batch,
                per_device_eval_batch_size=max(8, batch*2),
                num_train_epochs=epochs,
                weight_decay=0.01,
                do_eval=True,
                logging_steps=50,
                save_steps=500,
                eval_steps=500,
                fp16=True
            )
    train_args = make_args()

    # metric
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.squeeze(-1)
        rmse = math.sqrt(mean_squared_error(labels, preds))
        return {"rmse": rmse}

    # train / eval / save
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tok,
        compute_metrics=compute_metrics
    )
    print(f"\nTraining SpotToxBERT on {csv_path.name} ...")
    trainer.train()

    print("📊 Test evaluation:")
    print(trainer.evaluate(dataset["test"]))

    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)
    print(f"🎉 Saved model to ./{out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train SpotToxBERT on multiple CSVs (separately).")
    ap.add_argument("--csv", nargs="+", required=True, help="List of CSV paths (each trained separately)")
    ap.add_argument("--text_col", default="txt")
    ap.add_argument("--label_col", default="offensiveness_score")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_length", type=int, default=256)
    args = ap.parse_args()

    for csv in args.csv:
        train_one(Path(csv), args.text_col, args.label_col,
                  args.epochs, args.batch, args.lr, args.max_length)
