"""
Model Training + Conformal Calibration
======================================
Fine-tunes `distilbert-base-uncased` on the synthetic dispute texts with a
plain PyTorch training loop, then computes split-conformal calibration
scores on a held-out calibration set.

Outputs:
    model/                  - fine-tuned weights + tokenizer
    model/conformal.json    - calibration nonconformity scores + label map + test metrics

Splits (stratified by class):
    60% train / 20% calibration / 20% test

The calibration set is NEVER seen during training - that independence is
what gives split conformal prediction its finite-sample coverage guarantee.

Usage:
    python train_model.py [--epochs 3] [--batch-size 16] [--seed 42]
"""

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

DB_PATH = "fintech.db"
MODEL_NAME = "distilbert-base-uncased"
MODEL_DIR = Path("model")
MAX_LENGTH = 96
LABELS = ["Fraud", "Merchant Error", "Policy Violation"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}


class DisputeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


def load_disputes():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT dispute_text, true_category FROM disputes ORDER BY dispute_id"
    ).fetchall()
    conn.close()
    texts = [r[0] for r in rows]
    labels = [LABEL2ID[r[1]] for r in rows]
    return texts, labels


@torch.no_grad()
def predict_proba(model, loader, device):
    model.eval()
    probs, labels = [], []
    for batch in loader:
        out = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        probs.append(torch.softmax(out.logits, dim=-1).cpu())
        labels.append(batch["labels"])
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    texts, labels = load_disputes()
    print(f"Loaded {len(texts)} labelled disputes from {DB_PATH}")

    # --- stratified 60/20/20 split: train / calibration / test ---
    X_train, X_rest, y_train, y_rest = train_test_split(
        texts, labels, test_size=0.4, stratify=labels, random_state=args.seed
    )
    X_calib, X_test, y_calib, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=args.seed
    )
    print(f"Split: train={len(X_train)} calib={len(X_calib)} test={len(X_test)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label={i: l for l, i in LABEL2ID.items()},
        label2id=LABEL2ID,
    ).to(device)

    train_loader = DataLoader(
        DisputeDataset(X_train, y_train, tokenizer),
        batch_size=args.batch_size,
        shuffle=True,
    )
    calib_loader = DataLoader(
        DisputeDataset(X_calib, y_calib, tokenizer), batch_size=32
    )
    test_loader = DataLoader(
        DisputeDataset(X_test, y_test, tokenizer), batch_size=32
    )

    # --- training loop (plain PyTorch) ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps
    )

    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += out.loss.item()
            if step % 5 == 0 or step == len(train_loader):
                print(
                    f"epoch {epoch + 1}/{args.epochs} "
                    f"step {step}/{len(train_loader)} "
                    f"loss {running_loss / step:.4f}",
                    flush=True,
                )

    # --- conformal calibration ---
    # Nonconformity score: 1 - softmax probability of the TRUE class.
    # A well-classified calibration example gets a score near 0.
    calib_probs, calib_labels = predict_proba(model, calib_loader, device)
    calib_scores = 1.0 - calib_probs[np.arange(len(calib_labels)), calib_labels]

    # --- held-out test metrics ---
    test_probs, test_labels = predict_proba(model, test_loader, device)
    test_preds = test_probs.argmax(axis=1)
    test_acc = float((test_preds == test_labels).mean())
    print(f"Held-out test accuracy: {test_acc:.3f} (n={len(test_labels)})")

    # Empirical conformal coverage check on the test set at 95%:
    # prediction set = {classes whose p-value > 0.05}; coverage = how often
    # the true class is inside the set. Should be >= ~0.95.
    sorted_calib = np.sort(calib_scores)
    n = len(sorted_calib)

    def p_value(score: float) -> float:
        # (# calib scores >= score + 1) / (n + 1)
        return (np.sum(sorted_calib >= score) + 1) / (n + 1)

    covered = 0
    set_sizes = []
    for i in range(len(test_labels)):
        pvals = np.array([p_value(1.0 - test_probs[i, k]) for k in range(len(LABELS))])
        pred_set = set(np.where(pvals > 0.05)[0])
        set_sizes.append(len(pred_set))
        covered += int(test_labels[i] in pred_set)
    coverage = covered / len(test_labels)
    avg_set_size = float(np.mean(set_sizes))
    print(f"Conformal coverage @95%: {coverage:.3f} | avg set size: {avg_set_size:.2f}")

    # --- persist everything ---
    MODEL_DIR.mkdir(exist_ok=True)
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    with open(MODEL_DIR / "conformal.json", "w") as f:
        json.dump(
            {
                "labels": LABELS,
                "calibration_scores": calib_scores.tolist(),
                "test_accuracy": test_acc,
                "conformal_coverage_at_95": coverage,
                "avg_prediction_set_size": avg_set_size,
                "n_train": len(X_train),
                "n_calibration": len(X_calib),
                "n_test": len(X_test),
            },
            f,
            indent=2,
        )
    print(f"Saved fine-tuned model + conformal calibration to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
