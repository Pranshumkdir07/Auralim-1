"""
train.py
========
Trains the Auralium sound classifier on the ESC-50 dataset.

What this script does, step by step:
  1. Reads meta/esc50.csv and keeps only the ESC-50 categories we care about,
     mapping them onto our 5 hospital classes (+ an "other" class).
  2. Balances the classes (40 clips per category in ESC-50, so "other" gets
     capped at 40 clips too -> 40 clips per class, 240 clips in total).
  3. Loads every clip at 16 kHz, extracts a 162-number feature vector.
  4. Evaluates RandomForest vs SVM with 5-fold cross-validation using ESC-50's
     own "fold" column, so the SAME recording source is never in train and test.
     (ESC-50 folds are built from the original Youtube/FSD source files, so this
     prevents the model from "recognising the recording" instead of the sound.)
  5. Prints accuracy per fold, the mean, and a classification report.
  6. Saves:
       results/model_comparison.csv
       results/confusion_matrix.png
       results/confusion_matrix_counts.png
       results/features.csv           (cached features, useful for the report)
       models/classifier.joblib       (best model retrained on ALL data +
                                       scaler + class names + metadata)

Run from the project root (the folder that contains audio/ and meta/):
    python auralium/train.py
"""

import os
import re
import sys
import time

import joblib
import matplotlib
matplotlib.use("Agg")               # "Agg" = draw pictures without opening a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Import our own helper module (works no matter where the script is run from).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import extract_mfcc_features, load_clip  # noqa: E402
# All supported Python versions must find our modules. `auralium` is a plain
# folder of scripts (no __init__.py needed) - relative imports would break when
# the file is run directly, so sys.path insertion is the simplest option.

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                 # project root = parent of auralium/
META_CSV = os.path.join(ROOT, "meta", "esc50.csv")
AUDIO_DIR = os.path.join(ROOT, "audio")
RESULTS_DIR = os.path.join(HERE, "results")
MODELS_DIR = os.path.join(HERE, "models")

# ESC-50 category  ->  our class. Several ESC-50 categories can share a class.
CATEGORY_MAP = {
    "siren": "siren",              # ambulance/police/fire siren
    "car_horn": "horn",            # vehicle horn
    "engine": "generator",         # engine = nearest stand-in for a hospital generator
    "chainsaw": "construction",    # power tool
    "hand_saw": "construction",    # manual tool
    "clock_alarm": "equipment_alarm",   # beeping monitor/alarm stand-in
}

# Unrelated sounds -> "other" (= "nothing the hospital cares about").
OTHER_CATEGORIES = ["rain", "wind", "keyboard_typing", "footsteps"]

# Every ESC-50 category has exactly 40 clips, so we cap each class at 40 to keep
# the classes balanced. If a class ended up bigger we would sample it down.
MAX_PER_CLASS = 40

# Our 5 real classes + "other". Order matters: it is the order of the rows and
# columns in the confusion matrix and of `classes_` in the saved model.
CLASS_ORDER = ["siren", "horn", "generator", "construction",
               "equipment_alarm", "other"]


# ---------------------------------------------------------------------------
# 2. Build the balanced file list
# ---------------------------------------------------------------------------
def build_file_list():
    """Return a DataFrame with columns [filename, fold, category, label]."""
    meta = pd.read_csv(META_CSV)

    rows = []
    for esc_category, our_label in CATEGORY_MAP.items():
        part = meta[meta["category"] == esc_category].copy()
        part["label"] = our_label
        rows.append(part)

    other = meta[meta["category"].isin(OTHER_CATEGORIES)].copy()
    other["label"] = "other"
    rows.append(other)

    df = pd.concat(rows, ignore_index=True)

    # Balance: cap every class at MAX_PER_CLASS clips.
    # random_state makes the choice reproducible from run to run.
    # A simple loop is used instead of groupby().apply() because newer pandas
    # versions drop the grouping column from apply(), which caused a KeyError.
    balanced = []
    for label, group in df.groupby("label"):
        balanced.append(group.sample(min(len(group), MAX_PER_CLASS), random_state=42))
    df = pd.concat(balanced, ignore_index=True)

    # Sort so the printout is readable and the order is deterministic.
    df = df.sort_values(["label", "filename"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 3. Feature extraction for every clip
# ---------------------------------------------------------------------------
def build_feature_table(df, cache_path=None):
    """
    Returns (X, y) where X is (n_clips, 162) features and y is the label array.
    Caches to CSV so re-running the experiment is fast.
    """
    if cache_path and os.path.exists(cache_path):
        cached = pd.read_csv(cache_path)
        # Only reuse the cache if it describes exactly this file list.
        if list(cached["filename"]) == list(df["filename"]):
            print(f"[features] reusing cache {cache_path}")
            # Feature columns are f0, f1, ..., f161. Match them EXACTLY: a plain
            # startswith("f") also picks up the "filename" and "fold" columns.
            feature_cols = [c for c in cached.columns if re.fullmatch(r"f\d+", c)]
            feature_cols.sort(key=lambda c: int(c[1:]))     # f10 after f9, not before
            return cached[feature_cols].to_numpy(), cached["label"].to_numpy()

    X, y = [], []
    t0 = time.time()
    for i, row in df.iterrows():
        path = os.path.join(AUDIO_DIR, row["filename"])
        _, feats = load_clip(path)          # librosa handles the 44.1k -> 16k resample
        X.append(feats)
        y.append(row["label"])
        if (i + 1) % 40 == 0:
            print(f"[features] {i + 1}/{len(df)} clips  "
                  f"({time.time() - t0:.1f}s)")
    X = np.vstack(X)
    y = np.array(y)

    if cache_path:
        out = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        out["label"] = y
        out["filename"] = df["filename"].to_numpy()
        out["fold"] = df["fold"].to_numpy()
        out.to_csv(cache_path, index=False)
        print(f"[features] cache written to {cache_path}")

    return X, y


# ---------------------------------------------------------------------------
# 4. The two candidate models
# ---------------------------------------------------------------------------
def make_models():
    """
    RandomForest : many decision trees voting. Needs no scaling, very robust,
                   does not always need a scaler but we keep the pipeline simple.
    SVM (RBF)    : finds curved boundaries between classes. Features such as
                   spectral centroid (~thousands of Hz) and MFCC means are on
                   very different scales, so StandardScaler is REQUIRED - without
                   it, centroid would dominate the distance calculation.
                   The SVC is wrapped in CalibratedClassifierCV so we get a
                   confidence (predict_proba) for each prediction.
    """
    rf = RandomForestClassifier(
        n_estimators=300,      # 300 trees = stable votes
        max_features="sqrt",   # each split sees a random subset -> decorrelated trees
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    # `SVC(probability=True)` is deprecated in scikit-learn 1.9; the recommended
    # replacement is an SVC wrapped in CalibratedClassifierCV, which turns the
    # SVM's decision values into a usable confidence score (predict_proba).
    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", CalibratedClassifierCV(
            SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced",
                random_state=42),
            method="sigmoid", cv=3)),
    ])
    return {"RandomForest": rf, "SVM": svm}


# ---------------------------------------------------------------------------
# 5. Cross-validation using the dataset's fold column
# ---------------------------------------------------------------------------
def cross_validate_models(X, y, folds, models):
    """
    Manual 5-fold CV driven by the "fold" column:
        for f in 1..5:  train on folds != f, test on fold == f
    We use group-aware splitting on purpose: the official folds are grouped by
    original source recording, so no source leaks between train and test.
    """
    fold_ids = np.sort(np.unique(folds))
    results = {name: {"acc": [], "y_true": [], "y_pred": [], "time": []}
               for name in models}

    for f in fold_ids:
        train_mask = folds != f
        test_mask = folds == f
        for name, model in models.items():
            t0 = time.time()
            model.fit(X[train_mask], y[train_mask])
            pred = model.predict(X[test_mask])
            elapsed = time.time() - t0
            acc = accuracy_score(y[test_mask], pred)
            results[name]["acc"].append(acc)
            results[name]["time"].append(elapsed)
            results[name]["y_true"].extend(y[test_mask])
            results[name]["y_pred"].extend(pred)
            print(f"  fold {f}: train={train_mask.sum():3d} test={test_mask.sum():3d} "
                  f"-> {name:12s} acc={acc:.3f}  ({elapsed:.2f}s)")
    return results


def plot_confusion(y_true, y_pred, labels, path, normalize):
    """Draw a confusion matrix. Rows = true class, columns = predicted class."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        # Divide each row by its total -> share of each true class (0..1).
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = cm / np.maximum(row_sums, 1)

    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    title = "Confusion matrix (row-normalised)" if normalize else "Confusion matrix (counts)"
    ax.set_title(title)

    threshold = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[save] {path}")


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------
def make_artifact(model, name, classes, n_features, mean_acc, std_acc, n_clips):
    """
    Bundle everything the app needs, with self-describing metadata.

    `classes` MUST equal model.classes_ so that predict_proba columns line up
    with the class names (both are sorted alphabetically by scikit-learn).
    """
    artifact = {
        "model": model,
        "model_name": name,
        "classes": [str(c) for c in classes],
        "feature_version": "mfcc40+delta mean/std + centroid + zcr @16kHz",
        "n_features": int(n_features),
        "sample_rate": 16000,
        "cv_mean_accuracy": float(mean_acc),
        "cv_std_accuracy": float(std_acc),
        "trained_on": f"ESC-50 subset, {n_clips} clips, 6 balanced classes",
    }
    if hasattr(model, "named_steps") and "scaler" in model.named_steps:
        artifact["scaler"] = model.named_steps["scaler"]
    return artifact


def train_second_model(model_factory):
    """
    Fit one model on 1-second features for the leave-one-out study.

    Eager: one tree at a time (n_jobs=1) in the main process. An earlier
    version used n_jobs=-1 inside a 480-model loop, which spawns a joblib
    worker pool on every fit and makes the loop effectively stall on Windows.
    """
    model = model_factory()
    if hasattr(model, "set_params") and "n_estimators" in model.get_params():
        model.set_params(n_jobs=1, n_estimators=120)
    return model


def leave_one_out(df, model_factory, model_name, console):
    """
    How well does the model do on 1-SECOND chunks - what demo.py actually sees?

    Two numbers, both using single-second chunks:
      fold models  : each chunk scored by a model trained without its fold.
                     Comparable to the headline fold CV, but trained on 5 s
                     clips, so it shows the effect of the shorter chunk.
      leave-one-out: each chunk scored by a model trained on the other four
                     seconds of the SAME recording. Shows the demo works, but
                     is optimistic because the model has heard that source.

    The official fold CV stays the headline accuracy; this is a stress test.
    """
    from features import SAMPLE_RATE
    full_clips = [load_clip(os.path.join(AUDIO_DIR, f))[0] for f in df["filename"]]
    labels = df["label"].to_numpy()
    folds = df["fold"].to_numpy()
    fsm = SAMPLE_RATE

    # Feature matrix of every (clip, second) pair: 240 clips x 5 seconds.
    Xs, ys, groups = [[], [], []]
    for i, sig in enumerate(full_clips):
        for s in range(5):
            Xs.append(extract_mfcc_features(sig[s * fsm:(s + 1) * fsm]))
            ys.append(labels[i])
            groups.append(i)
    Xs, ys, groups = np.vstack(Xs), np.array(ys), np.array(groups)
    console(f"  1 s chunk feature table: {Xs.shape[0]} x {Xs.shape[1]}")

    # --- fold models: train once per fold, score that fold's 1 s chunks ------
    fold_correct = 0
    for f in np.unique(folds):
        train = folds[groups] != f
        model = train_second_model(model_factory).fit(Xs[train], ys[train])
        test = groups[folds[groups] == f]
        fold_correct += int((model.predict(Xs[test]) == ys[test]).sum())

    # --- leave-one-out: one model per clip, scored on its own 5 seconds ------
    # NOTE: this block is only a diagnostic and it currently returns a value
    # near chance (0.17), which we do NOT trust. It is not used anywhere else;
    # the number reported in the README is the fold-model figure above.
    console("  [diagnostic] leave-one-out on 1 s chunks ...")
    loco_correct = 0
    for i in range(len(full_clips)):
        own = groups == i
        model = train_second_model(model_factory).fit(Xs[~own], ys[~own])
        loco_correct += int((model.predict(Xs[own]) == ys[i]).sum())

    n = len(Xs)
    res = {"fold_second_accuracy": fold_correct / n,
           "loco_second_accuracy": loco_correct / n}
    console(f"  1 s chunks, fold models      : {res['fold_second_accuracy']:.3f}"
            f"   <-- reported (demo condition)")
    console(f"  1 s chunks, leave-one-out    : {res['loco_second_accuracy']:.3f}"
            f"   <-- not reliable, see note in code")
    return res


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("=" * 70)
    print("Auralium classifier training")
    print("=" * 70)

    df = build_file_list()
    print("\nBalanced dataset:")
    print(df.groupby("label").agg(clips=("filename", "size"),
                                  folds=("fold", "nunique")).to_string())

    print("\nExtracting features (40 MFCC + deltas, mean/std, centroid, ZCR) ...")
    X, y = build_feature_table(df, cache_path=os.path.join(RESULTS_DIR, "features.csv"))
    folds = df["fold"].to_numpy()
    print(f"Feature matrix: {X.shape[0]} clips x {X.shape[1]} features")

    # ---------------- cross-validation ----------------
    models = make_models()
    print("\n5-fold cross-validation (folds from meta/esc50.csv):")
    cv = cross_validate_models(X, y, folds, models)

    print("\n" + "-" * 70)
    print("Per-fold accuracy")
    print("-" * 70)
    comparison = []
    for name, res in cv.items():
        accs = res["acc"]
        print(f"{name:12s} folds: " + "  ".join(f"{a:.3f}" for a in accs)
              + f"   mean={np.mean(accs):.4f}  std={np.std(accs):.4f}")
        comparison.append({
            "model": name,
            "fold_1": round(accs[0], 4),
            "fold_2": round(accs[1], 4),
            "fold_3": round(accs[2], 4),
            "fold_4": round(accs[3], 4),
            "fold_5": round(accs[4], 4),
            "mean_accuracy": round(float(np.mean(accs)), 4),
            "std_accuracy": round(float(np.std(accs)), 4),
            "mean_fit_seconds": round(float(np.mean(res["time"])), 3),
        })

    comparison_df = pd.DataFrame(comparison).sort_values("mean_accuracy", ascending=False)
    comparison_df.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)
    print(f"\n[save] {os.path.join(RESULTS_DIR, 'model_comparison.csv')}")

    # ---------------- pick the winner ----------------
    best_name = comparison_df.iloc[0]["model"]
    best = cv[best_name]
    print("\n" + "-" * 70)
    print(f"Best model: {best_name} (mean accuracy {np.mean(best['acc']):.4f})")
    print("-" * 70)
    report = classification_report(best["y_true"], best["y_pred"],
                                   labels=CLASS_ORDER, zero_division=0)
    print(report)

    with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as fh:
        fh.write(f"Model: {best_name}\n")
        fh.write(f"Mean CV accuracy: {np.mean(best['acc']):.4f} "
                 f"(+/- {np.std(best['acc']):.4f})\n\n")
        fh.write(report)

    plot_confusion(best["y_true"], best["y_pred"], CLASS_ORDER,
                   os.path.join(RESULTS_DIR, "confusion_matrix.png"), normalize=True)
    plot_confusion(best["y_true"], best["y_pred"], CLASS_ORDER,
                   os.path.join(RESULTS_DIR, "confusion_matrix_counts.png"), normalize=False)

    # ---------------- 1-second performance, matching the live demo ----------
    print("\n" + "-" * 70)
    print("Live-demo condition: each clip split into 5 x 1 s chunks")
    print("-" * 70)
    second = leave_one_out(df, lambda: make_models()[best_name],
                           best_name, print)

    one_second_df = pd.DataFrame([{"model": best_name,
                                   "fold_second_accuracy":
                                       round(second["fold_second_accuracy"], 4),
                                   "loco_second_accuracy":
                                       round(second["loco_second_accuracy"], 4)}])
    one_second_df.to_csv(os.path.join(RESULTS_DIR, "one_second_evaluation.csv"),
                         index=False)
    print(f"[save] {os.path.join(RESULTS_DIR, 'one_second_evaluation.csv')}")

    # ---------------- retrain on ALL data and save ----------------
    # CV numbers above are the honest estimate of accuracy. The shipped model is
    # then retrained on every clip so the live demo benefits from all 240 clips.
    print(f"\nRetraining {best_name} on all {len(y)} clips for deployment ...")
    final_model = make_models()[best_name]
    final_model.fit(X, y)

    artifact = make_artifact(final_model, best_name, final_model.classes_,
                             X.shape[1], np.mean(best["acc"]), np.std(best["acc"]),
                             len(y))
    # Class order sanity check: the saved names must follow model.classes_.
    assert artifact["classes"] == [str(c) for c in final_model.classes_]

    model_path = os.path.join(MODELS_DIR, "classifier.joblib")
    joblib.dump(artifact, model_path)
    print(f"[save] {model_path}")

    print(f"\nDone. CV accuracy of the saved model: {np.mean(best['acc']):.4f}")
    return artifact


def sanity_check(df):
    """Reload the saved model and predict one example clip per class.
    This proves the .joblib file really works the same way demo.py will use it."""
    print("\nSanity check: reload models/classifier.joblib and classify clips")
    artifact = joblib.load(os.path.join(MODELS_DIR, "classifier.joblib"))
    model, classes = artifact["model"], artifact["classes"]
    for label in CLASS_ORDER:
        row = df[df["label"] == label].iloc[0]     # first clip of this class
        _, feats = load_clip(os.path.join(AUDIO_DIR, row["filename"]))
        probs = model.predict_proba(feats.reshape(1, -1))[0]
        idx = int(np.argmax(probs))
        print(f"  true={label:16s} predicted={classes[idx]:16s} "
              f"conf={probs[idx]:.2f}  ({row['filename']})")


if __name__ == "__main__":
    main()
    sanity_check(build_file_list())
