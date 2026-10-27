"""COSC 4371 project 7 (smishing): audit, split, leakage check and TF-IDF baseline.

One command trains the baseline and writes split.csv, scores.csv, threshold.txt and audit.txt:

  python smishing_baseline.py --uci data/SMSSpamCollection.csv --mendeley data/Dataset_5971.csv \
      --train uci --seed 42

--train names the dataset the baseline trains on (the other dataset is a test set only).
Positive class = 1 = unwanted message (UCI "spam"; Mendeley "spam" or "smishing"). Negative = "ham".
Needs Python 3.9+ with pandas, numpy, scipy, scikit-learn (matplotlib only for the figure).

Overall flow (each step is marked in main()):
  1. Load both datasets and give every message a unique id and a 0/1 label.
  2. Audit: count rows, classes, missing values and duplicates (numbers for section 2(b)).
  3. Remove exact duplicates inside each dataset, then remove Mendeley messages that are also in UCI.
  4. Group look-alike messages (near-duplicates, shared URL or phone number) so a group is never split.
  5. Split UCI into train / validation / hold-out; all remaining Mendeley messages are test only.
  6. Leakage check: no training message may also appear in validation or test.
  7. Train TF-IDF + logistic regression, choose the threshold on validation, score the test sets.
  8. Write split.csv, scores.csv, threshold.txt, audit.txt (and optionally the ROC figure).
"""

import argparse
import os
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve

# ---- settings you may need to quote in the report ----
NEAR_DUP_COS = 0.90   # two messages are near-duplicates if their TF-IDF cosine similarity is >= 0.90
FPR_TARGET = 0.05     # threshold rule: validation false-positive rate must be at most 5%
FRACTIONS = (0.80, 0.10, 0.10)  # train / validation / in-domain hold-out (primary dataset only)

# Patterns that find web links and phone numbers inside a message's text.
# They are used to spot "repeated entities" (the same link or number on several messages).
URL_RE = re.compile(r"(?:https?://|www\.)\S+|\b[\w-]+\.(?:com|net|org|co\.uk|info|biz|ly|me)\b\S*", re.I)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\s\-]{6,}\d)(?!\d)")

report = []  # every line printed by say() is also kept here and saved to audit.txt


def say(msg=""):
    """Print a line and remember it, so audit.txt contains exactly what was printed."""
    print(msg)
    report.append(msg)


def norm(text):
    """Exact-duplicate key: lower-cased text with whitespace collapsed.
    'Free  Entry!' and 'free entry!' get the same key, so they count as the same message."""
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def load_uci(path):
    """UCI SMS Spam Collection. Accepts the CSV version (a header row such as label,text or
    v1,v2 or Category,Message) or the original tab-separated file with no header."""
    # Peek at the first line to decide which of the file formats we were given.
    # "utf-8-sig" silently removes the invisible byte-order mark some files start with.
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        first = f.readline().replace('"', "").strip().lower()
    if first.startswith(("label,", "category,", "v1,", "class,", "type,")):
        # CSV version with a header row. dtype=str keeps everything as text;
        # keep_default_na=False stops the word "NA" in a message turning into a missing value.
        df = pd.read_csv(path, encoding="utf-8-sig", encoding_errors="replace", dtype=str,
                         keep_default_na=False)
        # Use the first two columns whatever their names are (label first, message text second).
        df = df.iloc[:, :2]
        df.columns = ["label_raw", "text"]
    else:
        # Original UCI format: label<TAB>text, no header. quoting=3 means "treat quotes as normal characters".
        df = pd.read_csv(path, sep="\t", header=None, names=["label_raw", "text"], quoting=3,
                         encoding="utf-8-sig", encoding_errors="replace", dtype=str, keep_default_na=False)
    return df


def load_mendeley(path):
    """Mendeley SMS Phishing file: columns LABEL, TEXT, URL, EMAIL, PHONE (only LABEL and TEXT are used)."""
    df = pd.read_csv(path, encoding="utf-8-sig", encoding_errors="replace", dtype=str, keep_default_na=False)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={"label": "label_raw"})
    if "text" not in df.columns or "label_raw" not in df.columns:
        sys.exit(f"Mendeley file needs LABEL and TEXT columns; found {list(df.columns)}")
    return df


def prepare(df, name):
    """Add the columns every later step needs: dataset name, unique row_id, 0/1 label, duplicate key."""
    df = df.copy()
    df["dataset"] = name
    df["orig_index"] = np.arange(len(df))  # position in the original file
    df["row_id"] = [f"{name}:{i}" for i in df["orig_index"]]  # e.g. "uci:17"; unique across both datasets
    lab = df["label_raw"].str.strip().str.lower()
    df["label"] = (lab != "ham").astype(int)  # 1 = unwanted (spam or smishing), 0 = ham
    df.loc[~lab.isin(["ham", "spam", "smishing"]), "label"] = -1  # unknown label: flagged in main()
    df["key"] = df["text"].map(norm)  # normalised text used to find exact duplicates
    return df


def exact_audit(df, name, raw_rows, raw_cols):
    """Print the audit numbers for section 2(b): classes, missing values, exact duplicates."""
    say(f"\n[{name}] rows {raw_rows}, columns {raw_cols}")
    counts = df["label_raw"].str.strip().str.lower().value_counts()
    for k, v in counts.items():
        say(f"  label '{k}': {v} ({100 * v / len(df):.2f}%)")
    pos = int((df["label"] == 1).sum())
    say(f"  positive (unwanted) {pos} ({100 * pos / len(df):.2f}%), negative {int((df['label'] == 0).sum())}")
    say(f"  missing/unrecognised target: {int((df['label'] == -1).sum())}")
    say(f"  missing text (key feature): {int((df['key'] == '').sum())}")
    # duplicated() marks every copy after the first, so this counts the *extra* copies.
    dup = int(df["key"].duplicated().sum())
    # A text is "conflicting" if identical copies carry different labels (e.g. one ham, one spam).
    conflict = int((df.groupby("key")["label"].nunique() > 1).sum())
    say(f"  exact duplicates (extra copies of the same normalised text): {dup}; "
        f"texts with conflicting labels: {conflict}")
    return dup, conflict


def dedupe(df, name):
    """Keep one copy of each normalised text; drop every copy of a text whose copies disagree on label."""
    conflict_keys = set(df.groupby("key")["label"].nunique().loc[lambda s: s > 1].index)
    n0 = len(df)
    keep = df[~df["key"].isin(conflict_keys)].drop_duplicates("key", keep="first")
    keep = keep[keep["key"] != ""].reset_index(drop=True)  # also drop empty messages
    say(f"[{name}] kept {len(keep)} of {n0} rows: removed {n0 - len(keep)} "
        f"(extra exact copies, plus all copies of label-conflicting texts). "
        f"Protects the train/validation/test boundaries inside the dataset.")
    return keep


def entity_counts(df, name):
    """Audit check 'repeated entities': how many URLs / phone numbers appear on more than one message."""
    urls = df["text"].map(lambda t: [u.lower().rstrip(".,)") for u in URL_RE.findall(str(t))])
    phones = df["text"].map(lambda t: [re.sub(r"\D", "", p) for p in PHONE_RE.findall(str(t))])
    for label, series in [("URL", urls), ("phone number", phones)]:
        flat = pd.Series([x for xs in series for x in xs])  # one long list of every URL (or number) found
        rep = flat.value_counts()
        rep = rep[rep > 1]  # keep only those seen more than once
        say(f"  repeated {label}s: {len(rep)} distinct values appear on more than one row "
            f"({int(rep.sum())} mentions); most repeated: {int(rep.iloc[0]) if len(rep) else 0}x")


def entities(text):
    """URLs and phone numbers in a message (the repeated entities of an SMS dataset)."""
    out = {"u:" + u.lower().rstrip(".,)") for u in URL_RE.findall(str(text))}
    out |= {"p:" + re.sub(r"\D", "", p) for p in PHONE_RE.findall(str(text))}
    return out


def build_groups(df):
    """Link rows that are near-duplicates (cosine >= NEAR_DUP_COS) or share a URL or phone number.
    A linked set is one group and is never split across train/validation/hold-out.
    Returns (near-duplicate components, near-duplicate + entity components)."""
    from scipy.sparse import coo_matrix
    # Turn every message into a TF-IDF vector (word 1- and 2-grams, length-normalised).
    vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, norm="l2")
    X = vec.fit_transform(df["key"])
    # X @ X.T gives the cosine similarity of every pair of messages (vectors are length-normalised).
    S = (X @ X.T).tocsr()
    S.data = (S.data >= NEAR_DUP_COS).astype(np.int8)  # 1 if the pair is similar enough, otherwise 0
    S.eliminate_zeros()
    # Treat "similar" as a link between two messages; connected messages form one component (group).
    _, near = connected_components(S, directed=False)
    # Also link messages that share a URL or phone number: remember the first message that had
    # each entity and link every later message with that entity back to it.
    seen, rows, cols = {}, [], []
    for i, t in enumerate(df["text"]):
        for e in entities(t):
            if e in seen:
                rows.append(seen[e])
                cols.append(i)
            else:
                seen[e] = i
    E = coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=S.shape)
    _, both = connected_components(S + E + E.T, directed=False)  # near-duplicate links + entity links
    return near, both


def grouped_split(groups, seed):
    """Assign whole groups (never single rows) to train / validation / test with a seeded shuffle."""
    rng = np.random.default_rng(seed)  # seeded so the split is the same every run
    ids = np.unique(groups)
    rng.shuffle(ids)  # random order of groups
    sizes = pd.Series(groups).value_counts()
    # Walk through the shuffled groups; the running share of rows decides which split each group joins.
    cum = np.cumsum([sizes[g] for g in ids]) / len(groups)
    assign = {}
    for g, c in zip(ids, cum):
        assign[g] = "train" if c <= FRACTIONS[0] else ("validation" if c <= FRACTIONS[0] + FRACTIONS[1] else "test")
    return np.array([assign[g] for g in groups])


def choose_threshold(val_scores, val_labels):
    """Lowest threshold whose false-positive rate on the validation set is at most FPR_TARGET.
    False-positive rate = share of normal (label 0) messages that score at or above the threshold."""
    neg = val_scores[val_labels == 0]  # scores of the normal messages only
    # Candidates: every distinct normal-message score, plus one value just above the highest score.
    cands = np.append(np.unique(neg), np.nextafter(neg.max(), np.inf))
    for t in cands:  # ascending: lowest threshold meeting the rule
        if (neg >= t).mean() <= FPR_TARGET:
            return float(t)
    return float(cands[-1])


def run_checker(split_path, scores_path, threshold, report_path):
    """Run check_checkpoint.py (kept in the same folder as this script) on the files just written,
    print its report, and save it to report_path. The threshold is passed exactly as stored in
    threshold.txt, which is what the checkpoint asks for."""
    checker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "check_checkpoint.py")
    if not os.path.exists(checker):
        print("\ncheck_checkpoint.py not found next to this script; run it yourself with the command above.")
        return
    result = subprocess.run([sys.executable, checker, "--split", split_path, "--scores", scores_path,
                             "--threshold", repr(threshold)], capture_output=True, text=True)
    text = result.stdout + result.stderr
    print("\n==== check_checkpoint.py report ====")
    print(text)
    with open(report_path, "w") as f:
        f.write(text)
    print(f"(saved to {report_path})")


def main():
    # ---- command-line options ----
    ap = argparse.ArgumentParser()
    ap.add_argument("--uci", required=True, help="UCI SMS Spam Collection file (CSV with header, or tab separated)")
    ap.add_argument("--mendeley", required=True, help="Mendeley Dataset_5971.csv")
    ap.add_argument("--train", choices=["uci", "mendeley"], default="uci")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overlap-from", choices=["test", "train"], default="test",
                    help="which side loses rows whose text is in both datasets "
                         "(default: the test dataset, so the training data stays whole)")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--no-checker", action="store_true",
                    help="do not run check_checkpoint.py automatically at the end")
    ap.add_argument("--figure", help="optional PNG path: ROC curve (log FPR axis) on the other dataset")
    args = ap.parse_args()

    # ---- step 1: load both files and add ids / labels ----
    raw = {"uci": load_uci(args.uci), "mendeley": load_mendeley(args.mendeley)}
    data = {k: prepare(v, k) for k, v in raw.items()}
    for k in data:
        if (data[k]["label"] == -1).any():
            bad = data[k].loc[data[k]["label"] == -1, "label_raw"].str.strip().str.lower()
            print(f"{k}: {len(bad)} rows have a label other than ham/spam/smishing. "
                  f"Most common values found: {bad.value_counts().head(5).to_dict()}")
            print(f"First rows of the {k} file as loaded:\n{raw[k].head(3).to_string()}")
            sys.exit(f"{k}: stopping. Check that you passed the right file and that it is unedited.")

    # ---- step 2: audit (numbers for section 2(b)) ----
    say("== Audit (code-measured, on the files loaded) ==")
    for k in data:
        exact_audit(data[k], k, len(raw[k]), raw[k].shape[1])
    say("\nTime range: unavailable (neither file has a timestamp column).")

    # A = the dataset we train on, B = the other dataset (test only)
    A_name = args.train
    B_name = "mendeley" if A_name == "uci" else "uci"

    say("\n== Repeated entities in the primary dataset (" + A_name + ") ==")
    entity_counts(data[A_name], A_name)

    # ---- step 3: removals ----
    say("\n== Removals ==")
    A = dedupe(data[A_name], A_name)  # exact duplicates inside each dataset
    B = dedupe(data[B_name], B_name)

    # Messages that appear in BOTH datasets are removed from one side so that training and
    # testing never share a message. By default the test side (B) loses them, so A stays whole.
    if args.overlap_from == "test":
        cross = B["key"].isin(set(A["key"]))
        say(f"[{B_name}] removed {int(cross.sum())} of {len(B)} rows whose text exactly matches a "
            f"{A_name} row. Protects the {A_name}->{B_name} dataset boundary; {A_name} training data stays whole.")
        B = B[~cross].reset_index(drop=True)
    else:
        cross = A["key"].isin(set(B["key"]))
        say(f"[{A_name}] removed {int(cross.sum())} of {len(A)} rows whose text exactly matches a "
            f"{B_name} row. Protects the {A_name}->{B_name} dataset boundary.")
        A = A[~cross].reset_index(drop=True)

    # ---- step 4: group look-alike messages, then step 5: split A by group ----
    near, comp = build_groups(A)
    sz = pd.Series(near).value_counts()  # size of each near-duplicate group
    say(f"\n[{A_name}] near-duplicates (cosine >= {NEAR_DUP_COS} on word 1-2gram TF-IDF): "
        f"{int((sz[sz > 1] - 1).sum())} rows beyond the first of {int((sz > 1).sum())} groups.")
    gs = pd.Series(comp).value_counts()  # size of each near-duplicate-or-shared-entity group
    say(f"[{A_name}] entity groups (near-duplicates plus rows sharing a URL or phone number): "
        f"{int((gs > 1).sum())} groups of 2+ rows covering {int(gs[gs > 1].sum())} rows; largest "
        f"{int(gs.iloc[0])} rows. Each group goes whole to one split (protects train/validation/hold-out).")
    A["split"] = grouped_split(comp, args.seed)

    # extra information: near-duplicate overlap with the other dataset (reported, not removed)
    vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), norm="l2").fit(pd.concat([A["key"], B["key"]]))
    sim = (vec.transform(A["key"]) @ vec.transform(B["key"]).T).tocsr()  # similarity of every A-B pair
    near_cross = int((sim.max(axis=0).toarray().ravel() >= NEAR_DUP_COS).sum())  # B rows with a close match in A
    say(f"[{B_name}] test rows with a near-duplicate (cosine >= {NEAR_DUP_COS}) in {A_name}: {near_cross} "
        f"of {len(B)} (reported only; not removed).")

    B["split"] = "test"  # every remaining B message is test only
    A_tests = f"{A_name}_holdout"  # name of the in-domain test set (held-out part of A)
    B_tests = f"{B_name}_all"      # name of the cross-dataset test set (what is left of B)

    # ---- step 6: leakage check ----
    tr = A[A["split"] == "train"]
    va = A[A["split"] == "validation"]
    te_keys = set(A.loc[A["split"] == "test", "key"]) | set(B["key"])  # texts of every test message
    say("\n== Leakage check (exact normalised text) ==")
    say(f"  training rows that exactly duplicate a row in any test set: {int(tr['key'].isin(te_keys).sum())}")
    say(f"  training rows that exactly duplicate a validation row: {int(tr['key'].isin(set(va['key'])).sum())}")

    # ---- step 7: baseline = TF-IDF + logistic regression ----
    t0 = time.perf_counter()  # start the training-time clock
    # TF-IDF turns each message into numbers: how often each word or word pair appears,
    # down-weighted for common words. Fit on TRAINING rows only so nothing leaks from test.
    #   ngram_range=(1,2): single words and word pairs; min_df=2: ignore terms seen in only 1 message;
    #   sublinear_tf: use 1+log(count) so a repeated word does not dominate.
    vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Xtr = vec.fit_transform(tr["text"])
    # Logistic regression learns a weight per term and outputs a score between 0 and 1.
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=args.seed)
    clf.fit(Xtr, tr["label"])
    train_seconds = time.perf_counter() - t0

    def score(df):
        """Probability that each message is unwanted (higher = more likely unwanted)."""
        return clf.predict_proba(vec.transform(df["text"]))[:, 1]

    # The threshold is chosen on VALIDATION data only (never on test data).
    thr = choose_threshold(score(va), va["label"].to_numpy())

    # ---- step 8: write the files the checker reads ----
    # split.csv: one row per kept message, saying which split it is in.
    out = pd.concat([A, B])[["row_id", "dataset", "split", "label"]]
    out.to_csv(f"{args.outdir}/split.csv", index=False)

    # scores.csv: one row per test message, scored once, with the name of its test set.
    sc = []
    for name, df in [(A_tests, A[A["split"] == "test"]), (B_tests, B)]:
        s = score(df)
        sc.append(pd.DataFrame({"row_id": df["row_id"].values, "test_set": name,
                                "label": df["label"].values, "score": s}))
    pd.concat(sc).to_csv(f"{args.outdir}/scores.csv", index=False)
    # repr() writes the number with every digit, so the checker gets exactly the threshold used here.
    with open(f"{args.outdir}/threshold.txt", "w") as f:
        f.write(repr(thr) + "\n")

    say("\n== Baseline ==")
    say(f"  train rows {len(tr)}, validation rows {len(va)}, hold-out rows {int((A['split'] == 'test').sum())}, "
        f"{B_name} test rows {len(B)}")
    say(f"  TF-IDF vocabulary size {Xtr.shape[1]}; training time {train_seconds:.2f} s (fit TF-IDF + classifier)")
    say(f"  threshold (lowest validation score with validation FPR <= {FPR_TARGET:.0%}): {thr!r}")
    say(f"  python {sys.version.split()[0]}, pandas {pd.__version__}, numpy {np.__version__}, "
        f"scipy {scipy.__version__}, scikit-learn {sklearn.__version__}, seed {args.seed}")
    say(f"\nNext: python check_checkpoint.py --split {args.outdir}/split.csv "
        f"--scores {args.outdir}/scores.csv --threshold {thr!r}")

    with open(f"{args.outdir}/audit.txt", "w") as f:
        f.write("\n".join(report) + "\n")

    # Run the instructor's checker on the two files just written and show its report.
    if not args.no_checker:
        run_checker(f"{args.outdir}/split.csv", f"{args.outdir}/scores.csv", thr,
                    f"{args.outdir}/report.txt")

    # ---- optional figure for section 4: ROC curve with a logarithmic false-positive axis ----
    if args.figure:
        import matplotlib
        matplotlib.use("Agg")  # draw to a file, no window needed
        import matplotlib.pyplot as plt
        s_b = sc[1]  # scores on the cross-dataset test set
        fpr, tpr, _ = roc_curve(s_b["label"], s_b["score"])
        plt.figure(figsize=(5, 4))
        plt.plot(fpr, tpr)
        plt.xscale("log")
        plt.xlim(1e-3, 1)
        plt.xlabel("false-positive rate (log)")
        plt.ylabel("recall")
        plt.title(f"{B_tests}: n={len(s_b)}, {100 * s_b['label'].mean():.1f}% positive")
        plt.tight_layout()
        plt.savefig(args.figure, dpi=150)


if __name__ == "__main__":
    main()
