import os
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer
)
import numpy as np, math, shutil, warnings
from pathlib import Path

SAVE_DIR = Path("models/SpotToxRoBERTa")
BASE_MODEL = "roberta-base"
SEED = 42

DATA_FILES = [
    "./data/ruddit_with_text_conv_0.csv",
    "./data/ruddit_with_text_conv_1.csv",
    "./data/ruddit_with_text_conv_2.csv",
]

pd.options.mode.chained_assignment = None

def load_data():
    dfs = []
    for f in DATA_FILES:
        df = pd.read_csv(f).dropna(subset=["txt", "offensiveness_score"])
        df["labels"] = pd.to_numeric(df["offensiveness_score"], errors="coerce")
        # normalize scores to 0-1
        df["labels"] = (df["labels"] - df["labels"].min()) / (df["labels"].max() - df["labels"].min())
        dfs.append(df[["txt", "labels"]])
        print(f"Loaded {f} with {len(df)} rows")
    full = pd.concat(dfs, ignore_index=True)
    print(f"Total rows: {len(full)}")
    return full.rename(columns={"txt": "text"})

def make_dataset(df):
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=SEED)
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=SEED)

    d = DatasetDict({
        "train": Dataset.from_pandas(train_df, preserve_index=False),
        "validation": Dataset.from_pandas(val_df, preserve_index=False),
        "test": Dataset.from_pandas(test_df, preserve_index=False),
    })
    tokenizer = RobertaTokenizer.from_pretrained(BASE_MODEL)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

    d = d.map(tok, batched=True)

    cols = ["input_ids", "attention_mask", "labels"]
    for split in d.keys():
        d[split] = d[split].remove_columns([c for c in d[split].column_names if c not in cols])

    return d, tokenizer

def make_args():
    try:
        return TrainingArguments(
            output_dir=str(SAVE_DIR),
            num_train_epochs=3,
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            fp16=False,
            logging_steps=50,
            seed=SEED,
        )
    except TypeError:
        warnings.warn("⚠️ Using legacy TrainingArguments fallback")
        return TrainingArguments(
            output_dir=str(SAVE_DIR),
            num_train_epochs=3,
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            logging_steps=50,
            save_steps=500,
        )

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.squeeze(-1)
    rmse = math.sqrt(((preds - labels) ** 2).mean())
    return {"rmse": rmse}

def train_roberta():
    df = load_data()
    dataset, tok = make_dataset(df)

    if SAVE_DIR.exists():
        shutil.rmtree(SAVE_DIR)

    model = RobertaForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=1)
    model.config.problem_type = "regression"

    args = make_args()

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tok,
        compute_metrics=compute_metrics,
    )

    print("\nTraining SpotTox RoBERTa...\n")
    trainer.train()

    print("\nFinal Test:")
    print(trainer.evaluate(dataset["test"]))

    trainer.save_model(SAVE_DIR)
    tok.save_pretrained(SAVE_DIR)
    print(f"\nModel saved to {SAVE_DIR}")

if __name__ == "__main__":
    train_roberta()


