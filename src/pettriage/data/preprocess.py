"""
Turns raw rows from training_data.csv (or a single live request) into the model-ready
numeric format. Ports the recipe from notebooks/04_feature_engineering.ipynb into
reusable code, so training, evaluation, and the future API all apply identical
transforms — see ADR-011 through ADR-014 in docs/DECISIONS.md for the reasoning
behind each step.

Run: python -m pettriage.data.preprocess
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from pettriage.config import MODELS_DIR, PROCESSED_DATA_DIR, RANDOM_STATE

SYMPTOM_COLS = ["Symptom_1", "Symptom_2", "Symptom_3", "Symptom_4"]

# Legacy Yes/No flag columns in training_data.csv, mapped to the symptom string(s)
# they correspond to. Used only to drop the redundant columns (ADR-012).
FLAG_TO_SYMPTOMS = {
    "Vomiting": ["Vomiting"],
    "Diarrhea": ["Diarrhea"],
    "Coughing": ["Coughing"],
    "Labored_Breathing": ["Labored Breathing"],
    "Lameness": ["Limping"],
    "Skin_Lesions": ["Skin Lesions", "Excessive Scratching"],
    "Nasal_Discharge": ["Nasal Discharge"],
    "Eye_Discharge": ["Eye Discharge"],
    "Appetite_Loss": ["Loss of Appetite"],
}

CAT_COLS = ["Animal_Type", "Gender"]
DROP_COLS = ["Breed"]
TARGET_COLS = ["Disease_Category", "Urgency"]

# ── Missing-value policy ───────────────────────────────────────────────────────
# An empty cell means one of three completely different things, and each demands a
# different response. Deciding this per column — rather than blanket-filling every
# gap — is the difference between handling missing data and corrupting it.
#
# 1. IMPUTABLE — "we failed to measure it." The true value exists; we just don't
#    know it. An owner with no thermometer leaves Body_Temperature blank, but the
#    animal certainly has a temperature. Filling with a sensible estimate (median,
#    fit on train only) is a reasonable guess.
IMPUTABLE_NUMERIC_COLS = [
    "Age",
    "Weight",
    "Body_Temperature",
    "Heart_Rate",
    "Duration_Days",
]

# 2. STRUCTURAL — "there is nothing to measure." An animal with 3 symptoms has an
#    empty Symptom_4 because it HAS no 4th symptom. Nothing is missing; the absence
#    IS the information. Imputing here would invent a symptom the animal does not
#    have. These are handled by multi-hot encoding instead (absent -> 0), and
#    Symptom_Count records the true number reported.
STRUCTURAL_ABSENCE_COLS = ["Symptom_2", "Symptom_3", "Symptom_4"]

# 3. REQUIRED — "this row is broken." A record with no Animal_Type or no
#    Disease_Category is a data-quality failure. Silently filling it would hide a
#    bug upstream, so validation fails loudly instead.
REQUIRED_COLS = [
    "Animal_Type",
    "Gender",
    "Symptom_1",
    "Duration",
    "Disease_Category",
    "Urgency",
]

# Symptom_Count is derived (a sum of multi-hot columns), so it can never be missing
# and is deliberately absent from every list above — but it still gets scaled.
DERIVED_NUMERIC_COLS = ["Symptom_Count"]

# Everything that gets scaled. Built from the lists above so the two can never drift
# apart: only IMPUTABLE_NUMERIC_COLS are passed to the imputer, but all of these are
# passed to the scaler.
NUMERIC_COLS = IMPUTABLE_NUMERIC_COLS + DERIVED_NUMERIC_COLS

DEFAULT_PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"


# "Labored Breathing" -> "Symptom_Labored_Breathing"
def symptom_col(name: str) -> str:
    return "Symptom_" + name.replace(" ", "_")


# '3 weeks' -> 21.0, 'A few hours' -> 0.25, 'Today' -> 0.5. Returns None if unparseable.
# Deterministic text parsing, not a statistical estimate — so it does NOT need to be
# fit on train only (ADR-013).
def parse_duration_to_days(value) -> float | None:
    s = str(value).strip().lower()
    if s == "today":
        return 0.5
    if "few hours" in s:
        return 0.25
    m = re.match(r"(\d+)\s*(day|week|month|year)", s)
    if not m:
        return None
    n, unit = int(m.group(1)), m.group(2)
    mult = {"day": 1, "week": 7, "month": 30, "year": 365}[unit]
    return float(n * mult)


# Classifies every column that has missing values into its policy bucket, so a data
# audit reports WHY each gap exists and what will be done about it — rather than just
# counting nulls. Returns one row per column that has at least one missing value.
# Used by notebook 03; the `verdict` column is what a reviewer actually reads.
def audit_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        if n_missing == 0:
            continue
        if col in REQUIRED_COLS:
            bucket, verdict = "REQUIRED", "FAIL - broken row, fix upstream"
        elif col in STRUCTURAL_ABSENCE_COLS:
            bucket, verdict = "STRUCTURAL", "OK - absence is information, never impute"
        elif col in IMPUTABLE_NUMERIC_COLS:
            bucket, verdict = "IMPUTABLE", "OK - fill with train median"
        else:
            bucket, verdict = "UNCLASSIFIED", "REVIEW - not in any policy list"
        rows.append({
            "column": col,
            "n_missing": n_missing,
            "pct_missing": round(100 * n_missing / len(df), 2),
            "bucket": bucket,
            "verdict": verdict,
        })
    return pd.DataFrame(
        rows, columns=["column", "n_missing", "pct_missing", "bucket", "verdict"]
    )


# Hard gate on the REQUIRED columns. Raises rather than warns: a row with no
# Animal_Type or no Disease_Category cannot be silently repaired, and letting it
# through would bury an upstream bug inside the model's training data.
def validate_required_columns(df: pd.DataFrame) -> None:
    present = [c for c in REQUIRED_COLS if c in df.columns]
    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Required columns absent from the dataset: {missing_cols}")

    offenders = {c: int(df[c].isna().sum()) for c in present if df[c].isna().any()}
    if offenders:
        raise ValueError(
            f"Required columns contain missing values: {offenders}. "
            "These cannot be imputed — fix the data source."
        )


# Builds the stratification key, collapsing combinations too small to split three ways.
# A stratum with 1 member cannot be spread across train/val/test at all, and sklearn
# refuses outright. Rather than dropping those rows, rare combinations fall back to
# stratifying on Urgency alone, then to a single shared bucket — so a genuinely rare
# case (Skin/Fungal that IS an emergency) still lands somewhere sensible.
def _stratify_key(df: pd.DataFrame, min_stratum: int) -> pd.Series:
    key = df["Disease_Category"] + "_" + df["Urgency"]

    rare = key.value_counts()[lambda c: c < min_stratum].index
    key = key.where(~key.isin(rare), "RARE_" + df["Urgency"])

    still_rare = key.value_counts()[lambda c: c < min_stratum].index
    key = key.where(~key.isin(still_rare), "RARE")

    if (key.value_counts() < 2).any():
        raise ValueError(
            "Cannot stratify: some group still has fewer than 2 rows after collapsing. "
            "The dataset is too small or too skewed for a 70/15/15 stratified split."
        )
    return key


# Stratified 70/15/15 split on Disease_Category + Urgency combined (ADR-011), so every
# split keeps a proportional share of even the rarest category/urgency combinations.
def split_data(
    df: pd.DataFrame, random_state: int = RANDOM_STATE, min_stratum: int = 6
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    strat_key = _stratify_key(df, min_stratum)
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=strat_key, random_state=random_state
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=strat_key.loc[temp_df.index],
        random_state=random_state,
    )
    return train_df.copy(), val_df.copy(), test_df.copy()


# Replaces the 4 positional Symptom_1-4 columns and the 9 legacy flag columns with one
# binary column per known symptom, plus Symptom_Count (ADR-012). Position-invariant:
# "Vomiting" in slot 1 or slot 3 produces the same result.
def _consolidate_symptoms(df: pd.DataFrame, symptom_vocab: list[str]) -> pd.DataFrame:
    df = df.copy()
    for s in symptom_vocab:
        df[symptom_col(s)] = df[SYMPTOM_COLS].eq(s).any(axis=1).astype(int)
    multihot_cols = [symptom_col(s) for s in symptom_vocab]
    df["Symptom_Count"] = df[multihot_cols].sum(axis=1)
    old_flag_cols = [c for c in FLAG_TO_SYMPTOMS if c in df.columns]
    return df.drop(columns=SYMPTOM_COLS + old_flag_cols, errors="ignore")


# Swaps the free-text Duration column for the numeric Duration_Days.
def _parse_duration(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Duration_Days"] = df["Duration"].apply(parse_duration_to_days)
    return df.drop(columns=["Duration"])


# Fills only the imputable numeric columns, then returns the full numeric block in
# NUMERIC_COLS order ready for the scaler. Derived columns (Symptom_Count) pass
# through untouched — they are never missing, so there is nothing to fill.
def _impute_numeric(df: pd.DataFrame, imputer: SimpleImputer | None):
    numeric = df[NUMERIC_COLS].copy()
    if imputer is not None:
        numeric[IMPUTABLE_NUMERIC_COLS] = imputer.transform(df[IMPUTABLE_NUMERIC_COLS])
    return numeric


# Everything learned from train data only. Save/load this, never recompute it from val,
# test, or live request data (see the 'fit on train only' discussion).
@dataclass
class FittedPreprocessor:
    symptom_vocab: list[str]
    ohe: OneHotEncoder
    scaler: StandardScaler
    cat_encoder: LabelEncoder
    urg_encoder: LabelEncoder
    imputer: SimpleImputer | None = None


# Fits the symptom vocabulary, one-hot encoder, scaler, and label encoders on train_df,
# and returns train_df transformed alongside the fitted objects. Nothing here may look
# at val/test/live data — that's the whole point of this function existing separately
# from transform().
def fit_transform(train_df: pd.DataFrame) -> tuple[pd.DataFrame, FittedPreprocessor]:
    symptom_vocab = sorted(
        s for s in pd.unique(train_df[SYMPTOM_COLS].values.ravel()) if pd.notna(s)
    )

    df = _consolidate_symptoms(train_df, symptom_vocab)
    df = _parse_duration(df)
    df = df.drop(columns=DROP_COLS, errors="ignore")

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    ohe.fit(df[CAT_COLS])

    # Only IMPUTABLE_NUMERIC_COLS are filled — never the structural columns, whose
    # emptiness is real information (see the missing-value policy at the top).
    # Median, not mean: vitals have occasional extreme values (a 42C heatstroke case)
    # that would drag a mean-filled value somewhere no real animal sits.
    imputer = SimpleImputer(strategy="median")
    imputer.fit(df[IMPUTABLE_NUMERIC_COLS])

    scaler = StandardScaler()
    scaler.fit(_impute_numeric(df, imputer))

    cat_encoder = LabelEncoder().fit(df["Disease_Category"])
    urg_encoder = LabelEncoder().fit(df["Urgency"])

    fitted = FittedPreprocessor(
        symptom_vocab, ohe, scaler, cat_encoder, urg_encoder, imputer
    )
    return _apply_fitted(df, fitted), fitted


# Applies an already-fitted preprocessor to new data — val, test, or a single live
# request. Works whether or not Disease_Category/Urgency are present: label columns are
# only added when the source data actually has them, since a live prediction request has
# no ground truth to encode.
def transform(df: pd.DataFrame, fitted: FittedPreprocessor) -> pd.DataFrame:
    df = _consolidate_symptoms(df, fitted.symptom_vocab)
    df = _parse_duration(df)
    df = df.drop(columns=DROP_COLS, errors="ignore")
    return _apply_fitted(df, fitted)


# The shared tail of fit_transform() and transform(): one-hot encode, scale, and (when
# the targets are present) label-encode. Kept separate so fit and transform paths can
# never drift apart.
def _apply_fitted(df: pd.DataFrame, fitted: FittedPreprocessor) -> pd.DataFrame:
    df = df.copy()

    encoded = pd.DataFrame(
        fitted.ohe.transform(df[CAT_COLS]),
        columns=fitted.ohe.get_feature_names_out(CAT_COLS),
        index=df.index,
    )
    df = pd.concat([df.drop(columns=CAT_COLS), encoded], axis=1)

    df[NUMERIC_COLS] = fitted.scaler.transform(_impute_numeric(df, fitted.imputer))

    if all(col in df.columns for col in TARGET_COLS):
        cat_labels = fitted.cat_encoder.transform(df["Disease_Category"])
        df["Disease_Category_Label"] = cat_labels
        df["Urgency_Label"] = fitted.urg_encoder.transform(df["Urgency"])

    return df


# Saves the fitted objects as a plain dict, NOT as a pickled FittedPreprocessor object.
# Pickle records the module a class was defined in; a class pickled while this file ran
# as "__main__" (python -m pettriage.data.preprocess) becomes unloadable from any other
# script, which is exactly what broke train_model_a.py. A dict of sklearn objects has no
# such problem, since their classes always live in sklearn's own modules.
def save_fitted(
    fitted: FittedPreprocessor, path: Path = DEFAULT_PREPROCESSOR_PATH
) -> Path:
    payload = {
        "symptom_vocab": fitted.symptom_vocab,
        "ohe": fitted.ohe,
        "scaler": fitted.scaler,
        "cat_encoder": fitted.cat_encoder,
        "urg_encoder": fitted.urg_encoder,
        "imputer": fitted.imputer,
    }
    joblib.dump(payload, path)
    return path


# Rebuilds a FittedPreprocessor from the saved dict. Raises a clear error if the file
# predates the dict format above, since the old pickle simply cannot be read back.
def load_fitted(path: Path = DEFAULT_PREPROCESSOR_PATH) -> FittedPreprocessor:
    stale_msg = (
        f"{path} is in the old pickled-object format and can't be loaded. "
        "Re-run `python -m pettriage.data.preprocess` to regenerate it."
    )
    try:
        payload = joblib.load(path)
    except AttributeError as exc:  # old file pickled the class itself, by module name
        raise TypeError(stale_msg) from exc
    if not isinstance(payload, dict):
        raise TypeError(stale_msg)
    return FittedPreprocessor(**payload)


# End-to-end reproduction of notebook 04: load -> stratified split -> fit on train ->
# transform val/test -> save train/val/test.csv + the fitted preprocessor.
def run(
    training_csv: Path = PROCESSED_DATA_DIR / "training_data.csv",
    out_dir: Path = PROCESSED_DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, FittedPreprocessor]:
    df = pd.read_csv(training_csv)

    audit = audit_missing_values(df)
    if audit.empty:
        print("Missing values: none")
    else:
        print("Missing-value audit:")
        print(audit.to_string(index=False))
    validate_required_columns(df)

    train_df, val_df, test_df = split_data(df)

    train_ready, fitted = fit_transform(train_df)
    val_ready = transform(val_df, fitted)
    test_ready = transform(test_df, fitted)

    ready = [("train", train_ready), ("val", val_ready), ("test", test_ready)]
    for name, part in ready:
        part.to_csv(out_dir / f"{name}.csv", index=False)
        print(f"Saved {name}: {part.shape} -> {out_dir / f'{name}.csv'}")

    saved_path = save_fitted(fitted)
    print(f"Saved fitted preprocessor -> {saved_path}")

    return train_ready, val_ready, test_ready, fitted


if __name__ == "__main__":
    run()
