"""
Trains Model A — the Condition classifier (11-class), per ADR-001 in docs/DECISIONS.md.

Model selection, not a single-model script: several model families are trained on the
same data and the champion is chosen by validation MACRO F1 (ADR-018) rather than
assumed. Macro F1, not accuracy — on imbalanced data accuracy rewards neglecting the
rare classes, which in a triage system means quietly missing rare diseases.

LogisticRegression is the BASELINE to beat (ADR-017). On the first ambiguous dataset it
outscored tuned gradient boosting by 6.7 points with a train/val gap of 0.025 vs 0.476 —
the trees had capacity to spare and spent it memorising. Any tree ensemble added here has
to earn its place against that.

Every model is logged to MLflow as its own TOP-LEVEL run (not nested), so the runs table
compares them directly and each run's Metrics tab has its own charts.

Test.csv is never touched here — it stays sealed until a final single evaluation.

Run: python -m pettriage.models.train_model_a
"""

from __future__ import annotations

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from tqdm import tqdm

from pettriage.config import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
)
from pettriage.data.preprocess import load_fitted

CONDITION_LABEL_COL = "Disease_Category_Label"
URGENCY_LABEL_COL = "Urgency_Label"
NON_FEATURE_COLS = ["Disease_Category", "Urgency", CONDITION_LABEL_COL, URGENCY_LABEL_COL]

HISTGB_MAX_ITER = 300
HISTGB_STEP = 10
HISTGB_PATIENCE = 4  # stop once val loss hasn't improved for this many chunks

# The champion is chosen on macro F1, NOT accuracy (ADR-018). With ~5x class imbalance,
# accuracy rewards a model for getting the common classes (Skin, GI) right while
# neglecting rare ones like Viral Systemic entirely. Macro F1 averages the per-class
# F1 scores with equal weight, so ignoring a rare disease actually costs the model —
# which is the behaviour a triage system needs.
CHAMPION_METRIC = "macro_f1"


# Loads the three saved splits. All three are loaded so shapes can be sanity-checked,
# but only train/val are used — test.csv is reserved for a single final evaluation.
def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    val = pd.read_csv(PROCESSED_DATA_DIR / "val.csv")
    test = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")
    print(f"Loaded splits: train={train.shape} val={val.shape} test={test.shape}")
    return train, val, test


# Every column except the two targets (text + numeric-label versions of each).
def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


# Share of rows where the true class is among the model's 3 highest-probability guesses.
# This is the metric ADR-015 cares about: when the top guess is wrong, is the right
# answer still worth showing the user as an alternative?
def top_k_accuracy(proba: np.ndarray, y_true: np.ndarray, k: int = 3) -> float:
    top_k = np.argsort(proba, axis=1)[:, ::-1][:, :k]
    return float(np.mean([y_true[i] in top_k[i] for i in range(len(y_true))]))


# Grows a HistGradientBoostingClassifier in chunks via warm_start, logging train and val
# log-loss after every chunk so MLflow renders a real loss curve, and stopping early once
# val loss stops improving (the model has started overfitting past that point).
def fit_histgb_with_curve(model, X_train, y_train, X_val, y_val, desc: str):
    best_val_loss = float("inf")
    rounds_without_improvement = 0
    labels = np.unique(y_train)

    pbar = tqdm(range(HISTGB_MAX_ITER // HISTGB_STEP), desc=desc)
    for i in pbar:
        model.max_iter = (i + 1) * HISTGB_STEP
        model.fit(X_train, y_train)

        train_loss = log_loss(y_train, model.predict_proba(X_train), labels=labels)
        val_loss = log_loss(y_val, model.predict_proba(X_val), labels=labels)
        mlflow.log_metric("train_loss", train_loss, step=i)
        mlflow.log_metric("val_loss", val_loss, step=i)
        pbar.set_postfix(
            trees=model.max_iter, train=f"{train_loss:.4f}", val=f"{val_loss:.4f}"
        )

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            rounds_without_improvement = 0
        else:
            rounds_without_improvement += 1
            if rounds_without_improvement >= HISTGB_PATIENCE:
                pbar.set_description(f"{desc} [early stop @ {best_val_loss:.4f}]")
                break
    return model


# Trains one candidate, logs everything MLflow needs to be useful (params, headline
# metrics, per-class F1, and the confusion matrix + full report as artifacts), and
# returns the scores for the champion comparison.
def train_candidate(
    name, model, X_train, y_train, X_val, y_val, label_names, use_curve=False
) -> dict:
    with mlflow.start_run(run_name=name):
        mlflow.log_param("model_type", type(model).__name__)
        mlflow.log_param("candidate", name)
        for k, v in model.get_params().items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                mlflow.log_param(f"hp_{k}", v)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_train_rows", len(X_train))

        if use_curve:
            model = fit_histgb_with_curve(model, X_train, y_train, X_val, y_val, name)
        else:
            print(f"  training {name} ...")
            model.fit(X_train, y_train)

        train_acc = float((model.predict(X_train) == y_train).mean())
        val_pred = model.predict(X_val)
        val_acc = float((val_pred == y_val).mean())
        proba = model.predict_proba(X_val)
        top3 = top_k_accuracy(proba, y_val.to_numpy(), k=3)
        macro_f1 = float(f1_score(y_val, val_pred, average="macro"))

        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("val_accuracy", val_acc)
        mlflow.log_metric("overfit_gap", train_acc - val_acc)
        mlflow.log_metric("val_top3_accuracy", top3)
        mlflow.log_metric("val_macro_f1", macro_f1)
        mlflow.log_metric(
            "val_log_loss", float(log_loss(y_val, proba, labels=np.unique(y_train)))
        )
        mlflow.log_metric("mean_top_probability", float(proba.max(axis=1).mean()))
        mlflow.log_metric("frac_uncertain_below_0.6", float((proba.max(axis=1) < 0.6).mean()))

        per_class = f1_score(y_val, val_pred, average=None, labels=range(len(label_names)))
        for cls, score in zip(label_names, per_class):
            mlflow.log_metric(f"f1_{cls.replace(' / ', '_').replace(' ', '_')}", float(score))

        report = classification_report(
            y_val, val_pred, target_names=label_names, digits=3, zero_division=0
        )
        cm = pd.DataFrame(
            confusion_matrix(y_val, val_pred, labels=range(len(label_names))),
            index=label_names, columns=label_names,
        )
        mlflow.log_text(report, "classification_report.txt")
        mlflow.log_text(cm.to_csv(), "confusion_matrix.csv")
        mlflow.sklearn.log_model(model, name=name)

        print(f"  {name:34s} train={train_acc:.3f} val={val_acc:.3f} "
              f"gap={train_acc - val_acc:.3f} top3={top3:.3f} macroF1={macro_f1:.3f}")

    return {
        "name": name, "model": model, "train_acc": train_acc, "val_acc": val_acc,
        "gap": train_acc - val_acc, "top3": top3, "macro_f1": macro_f1,
        "report": report, "cm": cm,
    }


# The candidate line-up. LogisticRegression is the baseline; the tree ensembles have to
# beat it to justify their complexity. class_weight="balanced" matters now that the data
# uses real-world prevalence — without it, rare classes like Viral Systemic get ignored
# in favour of always guessing a common one.
def build_candidates() -> list[tuple[str, object, bool]]:
    return [
        ("baseline_logreg", LogisticRegression(
            C=0.3, max_iter=3000, random_state=RANDOM_STATE), False),
        ("baseline_logreg_balanced", LogisticRegression(
            C=0.3, max_iter=3000, class_weight="balanced",
            random_state=RANDOM_STATE), False),
        ("random_forest", RandomForestClassifier(
            n_estimators=400, min_samples_leaf=3, class_weight="balanced_subsample",
            random_state=RANDOM_STATE, n_jobs=-1), False),
        ("hist_gradient_boosting", HistGradientBoostingClassifier(
            learning_rate=0.05, max_leaf_nodes=16, min_samples_leaf=25,
            l2_regularization=1.0, warm_start=True, max_iter=HISTGB_STEP,
            random_state=RANDOM_STATE), True),
    ]


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    train, val, test = load_splits()
    X_cols = feature_cols(train)
    fitted = load_fitted()
    label_names = list(fitted.cat_encoder.classes_)

    print(f"{len(X_cols)} feature columns, {len(label_names)} classes")
    counts = train["Disease_Category"].value_counts()
    print(f"class balance in train: largest={counts.max()} smallest={counts.min()} "
          f"(ratio {counts.max() / counts.min():.1f}x)\n")

    X_train, y_train = train[X_cols], train[CONDITION_LABEL_COL]
    X_val, y_val = val[X_cols], val[CONDITION_LABEL_COL]

    results = [
        train_candidate(name, model, X_train, y_train, X_val, y_val, label_names, curve)
        for name, model, curve in build_candidates()
    ]

    print("\n" + "=" * 78)
    print(f"MODEL COMPARISON (sorted by {CHAMPION_METRIC} — the selection criterion)")
    print("=" * 78)
    table = pd.DataFrame([
        {k: r[k] for k in ("name", "train_acc", "val_acc", "gap", "top3", "macro_f1")}
        for r in results
    ]).sort_values(CHAMPION_METRIC, ascending=False)
    print(table.to_string(index=False))

    champion = max(results, key=lambda r: r[CHAMPION_METRIC])
    baseline = next(r for r in results if r["name"] == "baseline_logreg")
    print(f"\nCHAMPION: {champion['name']} "
          f"({CHAMPION_METRIC} {champion[CHAMPION_METRIC]:.3f}, "
          f"accuracy {champion['val_acc']:.3f})")

    # Worth seeing when the two criteria disagree: on imbalanced data the
    # highest-accuracy model is often one that quietly neglects the rare classes.
    by_accuracy = max(results, key=lambda r: r["val_acc"])
    if by_accuracy["name"] != champion["name"]:
        print(f"  NOTE: '{by_accuracy['name']}' scored higher on raw accuracy "
              f"({by_accuracy['val_acc']:.3f} vs {champion['val_acc']:.3f}) but lower on "
              f"{CHAMPION_METRIC} ({by_accuracy['macro_f1']:.3f} vs "
              f"{champion['macro_f1']:.3f}) — it wins on the common classes while doing "
              "worse on the rare ones. Macro F1 is the criterion here (ADR-018).")

    if champion["name"].startswith("baseline"):
        print("  The baseline won — the tree ensembles did not justify their complexity.")
    else:
        lift = champion[CHAMPION_METRIC] - baseline[CHAMPION_METRIC]
        print(f"  Beat the LogisticRegression baseline by {lift:+.3f} {CHAMPION_METRIC}.")

    print(f"\n--- {champion['name']} — Validation ---")
    print(champion["report"])
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(champion["cm"])

    joblib.dump(champion["model"], MODELS_DIR / "model_a.joblib")
    print(f"\nSaved champion Model A -> {MODELS_DIR / 'model_a.joblib'}")
    print("\nBrowse runs:  mlflow ui --backend-store-uri sqlite:///mlflow.db")
    print("Then open http://localhost:5000 and select the 'pettriage' experiment.")


if __name__ == "__main__":
    main()
