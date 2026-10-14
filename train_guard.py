"""Fine-tune a small transformer as a prompt-injection classifier.

Usage:
    python train_guard.py                       # trains and saves to ./guard_model
    python train_guard.py --push user/my-guard  # also pushes to the Hugging Face Hub
"""
import argparse
import json

import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments,
                          set_seed)

ID2LABEL = {0: "SAFE", 1: "INJECTION"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0)
    return {"accuracy": accuracy_score(labels, preds),
            "precision": prec, "recall": rec, "f1": f1}


def main():
    p = argparse.ArgumentParser()
    # The dataset contains some non-English examples, so default to a multilingual model.
    p.add_argument("--model", default="distilbert-base-multilingual-cased")
    p.add_argument("--out", default="guard_model")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--push", default=None, help="HF repo id, e.g. username/prompt-guard")
    a = p.parse_args()
    set_seed(42)

    ds = load_dataset("deepset/prompt-injections")  # columns: text, label (1 = injection)
    # Hold out a validation split from train for model selection, so the
    # official test split is only used once, for the final numbers.
    split = ds["train"].train_test_split(test_size=0.15, seed=42, stratify_by_column="label")
    train_ds, val_ds, test_ds = split["train"], split["test"], ds["test"]

    tok = AutoTokenizer.from_pretrained(a.model)
    enc = lambda d: d.map(lambda b: tok(b["text"], truncation=True, max_length=128), batched=True)
    train_ds, val_ds, test_ds = enc(train_ds), enc(val_ds), enc(test_ds)

    model = AutoModelForSequenceClassification.from_pretrained(
        a.model, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID)

    args = TrainingArguments(
        output_dir=f"{a.out}_ckpt", num_train_epochs=a.epochs, learning_rate=a.lr,
        per_device_train_batch_size=a.batch, per_device_eval_batch_size=32,
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="f1", save_total_limit=1, report_to="none", seed=42)

    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=DataCollatorWithPadding(tok),
                      compute_metrics=compute_metrics)
    trainer.train()

    test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
    print(json.dumps(test_metrics, indent=2))

    trainer.save_model(a.out)
    tok.save_pretrained(a.out)
    with open(f"{a.out}/test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    if a.push:
        model.push_to_hub(a.push)
        tok.push_to_hub(a.push)


if __name__ == "__main__":
    main()
