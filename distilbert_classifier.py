"""COSC 4371 project 7: classifier 2, fine-tuned DistilBERT (the transformer).

Uses the SAME split as the baseline: it reads split.csv written by smishing_baseline.py, gets the
message text from the two original files, fine-tunes on the train rows, picks the threshold on the
validation rows (lowest threshold with validation false-positive rate <= 5%, same rule as the
baseline) and scores every test row once. It writes scores_distilbert.csv in the checker's format.

Run it on a GPU (free Colab T4: Runtime > Change runtime type > T4 GPU):

  pip install torch transformers pandas numpy scikit-learn scipy
  python distilbert_classifier.py --uci data/SMSSpamCollection.csv --mendeley data/Dataset_5971.csv \
      --split split.csv --out scores_distilbert.csv --seed 42
  python check_checkpoint.py --split split.csv --scores scores_distilbert.csv --threshold <printed value>

Put smishing_baseline.py in the same folder (this script reuses its file loaders and threshold rule).
--dry-run skips the model and writes random scores, only to test the file plumbing.
"""

import argparse
import random
import sys
import time

import numpy as np
import pandas as pd

from smishing_baseline import choose_threshold, load_mendeley, load_uci, prepare, run_checker

MODEL = "distilbert-base-uncased"
MAX_LEN = 128
EPOCHS = 3
BATCH = 32
LR = 2e-5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uci", required=True)
    ap.add_argument("--mendeley", required=True)
    ap.add_argument("--split", default="split.csv", help="split.csv written by smishing_baseline.py")
    ap.add_argument("--out", default="scores_distilbert.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-checker", action="store_true",
                    help="do not run check_checkpoint.py automatically at the end")
    ap.add_argument("--dry-run", action="store_true", help="random scores, no model (plumbing test)")
    args = ap.parse_args()

    # message text for every row id, using the same loaders and row ids as the baseline
    text = {}
    for name, loader, path in [("uci", load_uci, args.uci), ("mendeley", load_mendeley, args.mendeley)]:
        df = prepare(loader(path), name)
        text.update(dict(zip(df["row_id"], df["text"])))

    split = pd.read_csv(args.split, dtype={"row_id": str, "dataset": str, "split": str})
    split["text"] = split["row_id"].map(text)
    if split["text"].isna().any():
        sys.exit("some row_id values in split.csv are not in the data files; use the same files as the baseline run")

    train = split[split["split"] == "train"]
    val = split[split["split"] == "validation"]
    test = split[split["split"] == "test"]
    train_ds = train["dataset"].iloc[0]
    print(f"train {len(train)}, validation {len(val)}, test {len(test)} rows; training dataset: {train_ds}")

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.dry_run:
        rng = np.random.default_rng(args.seed)

        def score(df):
            return rng.random(len(df))

        train_seconds = 0.0
        device = "none (dry run)"
    else:
        import torch
        from torch.utils.data import DataLoader
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.manual_seed(args.seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("WARNING: no GPU found; fine-tuning on CPU will be slow (use a Colab T4).")
        tok = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

        def batches(texts, labels=None, shuffle=False):
            idx = list(range(len(texts)))
            if shuffle:
                random.shuffle(idx)
            for i in range(0, len(idx), BATCH):
                b = idx[i:i + BATCH]
                enc = tok([texts[j] for j in b], truncation=True, max_length=MAX_LEN,
                          padding=True, return_tensors="pt").to(device)
                if labels is not None:
                    enc["labels"] = torch.tensor([labels[j] for j in b]).to(device)
                yield enc

        t0 = time.perf_counter()
        xs, ys = train["text"].tolist(), train["label"].tolist()
        model.train()
        for epoch in range(EPOCHS):
            total = 0.0
            for enc in batches(xs, ys, shuffle=True):
                loss = model(**enc).loss
                loss.backward()
                opt.step()
                opt.zero_grad()
                total += loss.item()
            print(f"epoch {epoch + 1}/{EPOCHS}: mean batch loss {total / (len(xs) / BATCH):.4f}")
        train_seconds = time.perf_counter() - t0
        model.eval()

        def score(df):
            out = []
            with torch.no_grad():
                for enc in batches(df["text"].tolist()):
                    out.append(torch.softmax(model(**enc).logits, dim=-1)[:, 1].cpu().numpy())
            return np.concatenate(out)

    thr = choose_threshold(score(val), val["label"].to_numpy())

    parts = []
    for ds, g in test.groupby("dataset", sort=False):
        name = f"{ds}_holdout" if ds == train_ds else f"{ds}_all"
        parts.append(pd.DataFrame({"row_id": g["row_id"].values, "test_set": name,
                                   "label": g["label"].values, "score": score(g)}))
    pd.concat(parts).to_csv(args.out, index=False)
    with open("threshold_distilbert.txt", "w") as f:
        f.write(repr(thr) + "\n")

    print(f"\nmodel {MODEL}; max length {MAX_LEN}; {EPOCHS} epochs; batch {BATCH}; learning rate {LR}; seed {args.seed}")
    print(f"device {device}; fine-tuning time {train_seconds / 60:.1f} minutes ({train_seconds:.0f} s)")
    print(f"threshold (lowest validation score with validation FPR <= 5%): {thr!r}")
    print(f"\nNext: python check_checkpoint.py --split {args.split} --scores {args.out} --threshold {thr!r}")

    # Run the instructor's checker (check_checkpoint.py in this folder) and show its report.
    if not args.no_checker:
        run_checker(args.split, args.out, thr, "report_distilbert.txt")


if __name__ == "__main__":
    main()
