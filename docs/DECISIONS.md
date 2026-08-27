# PetTriage — Architecture Decision Records (ADRs)

> Decisions are logged here as they are made — not pre-planned.
> This answers: "Why did we build it this way?"

---

## ADR-001: Two Separate Classifiers
**Date:** 2026-08-25
**Status:** Accepted

**Decision:**
Two classifiers — one for Condition, one for Urgency — instead of a single multi-output model.

**Reason:**
- Urgency can be correct even when Condition is wrong
- Feedback can be collected and used separately for each
- Easier to retrain, version, and monitor independently

---

## ADR-002: Soft Cascade Architecture
**Date:** 2026-08-25
**Status:** Accepted

**Decision:**
Urgency classifier receives Model A's **full probability distribution** (soft labels), not the top predicted condition (hard label).

**Reason:**
Hard labels cause error propagation. If Model A is 40% UTI / 35% Kidney Stones / 25% other, that uncertainty is meaningful signal. Passing only "UTI" discards it.

**Example:**
```
Hard label approach → Model A: "Stomach Upset" → Model B: "Monitor" ← WRONG, dangerous
Soft label approach → Model A: {Internal Bleeding: 0.35, ...} → Model B: "Emergency" ← correct
```

---

## ADR-003: scikit-learn for Phase 1
**Date:** 2026-08-25
**Status:** Accepted

**Decision:**
Use scikit-learn (RandomForest / LogisticRegression) for initial model, not deep learning.

**Reason:**
- Dataset will be small initially
- Deep learning needs volume to outperform classical ML
- scikit-learn models are interpretable — important for health applications
- Easy to swap out later without touching the rest of the system

---

## ADR-004: Feedback Trust Asymmetry
**Date:** 2026-08-25
**Status:** Accepted

**Decision:**
Urgency feedback = high trust. Condition feedback = low trust.

**Reason:**
Owners know after the fact if it was an emergency. They do not reliably know the medical condition — that requires a vet. Using owner-supplied condition labels as ground truth introduces label noise.

---

## ADR-005: MLflow for Experiment Tracking
**Date:** 2026-08-25
**Status:** Accepted

**Decision:**
MLflow running locally for Phase 1.

**Reason:**
Free, open source, no account needed. Tracks params, metrics, and model artifacts. Migrating to a remote server later requires zero code changes.

---

## ADR-006: uv as Package Manager
**Date:** 2026-08-25
**Status:** Accepted

**Decision:**
`uv` instead of `pip` for all dependency management.

**Reason:**
Faster, modern, compatible with standard `requirements.txt`. Already installed.

---

## ADR-007: Synthetic Data for Phase 1
**Date:** 2026-08-26
**Status:** Accepted

**Decision:**
Use LLM-generated synthetic data as the sole training dataset for Phase 1. Real Kaggle data used as seed/reference only.

**Reason:**
- Real datasets lack Urgency labels
- Real datasets are thin (<5 rows for many categories)
- Synthetic data lets us control class balance and label quality
- MLOps pipeline is identical regardless of data source
- Phase 2 will retrain with real data + user feedback

---

## ADR-008: Seed Data Not Merged into Training
**Date:** 2026-08-28
**Status:** Accepted

**Decision:**
The 147-row seed dataset (`dogs_cats_labeled.csv`) is NOT merged into `training_data.csv`.

**Reason:**
The seed data has no `Urgency` column — the primary label for Model B. Merging it would require either dropping it (losing rows) or imputing urgency labels (introducing noise). Synthetic data already provides well-labeled, balanced training examples.

---

## ADR-009: 11 Disease Categories
**Date:** 2026-08-26
**Status:** Accepted

**Decision:**
Group all conditions into 11 clinical categories instead of predicting specific disease names.

**Categories:**
Respiratory, Gastrointestinal, Viral Systemic, Bacterial/Parasitic,
Skin/Fungal, Musculoskeletal, Eye/Ear, Renal/Urinary,
Endocrine/Metabolic, Dental/Oral, Trauma/Poisoning

**Reason:**
- 139 specific diseases → 3 rows per disease average (not learnable)
- Categories are more actionable for pet owners ("Respiratory" vs "Feline Herpesvirus")
- 11 categories with 200 rows each = learnable, balanced training set
- LLM generation is cleaner per category than per specific disease

---

## ADR-010: No PetMD Scraping
**Date:** 2026-08-28
**Status:** Accepted

**Decision:**
Do not scrape PetMD or similar veterinary websites.

**Reason:**
- ToS prohibits automated scraping for commercial purposes
- LLMs (Claude, Gemini) have already ingested PetMD content during training
- Generating via LLM is legally equivalent, faster, and produces cleaner labeled data
- The knowledge is the same; only the access method differs

---

## ADR-011: Stratified Split on Combined Category+Urgency Key
**Date:** 2026-08-28
**Status:** Accepted

**Decision:**
Split `training_data.csv` into train/val/test (70/15/15) stratified on a combined
`Disease_Category + "_" + Urgency` key, not a plain random split or single-column stratify.

**Reason:**
Category and Urgency are entangled (e.g. Trauma/Poisoning is 65% Emergency, Skin/Fungal is
5%). Stratifying on Urgency alone could still cluster all of one category into a single
split. Some combined groups are as small as 10 rows out of 2,200 — a random or
single-column-stratified split risked near-zero representation of rare combinations in
val/test.

---

## ADR-012: Multi-Hot Symptom Encoding Replaces Positional Slots + Legacy Flags
**Date:** 2026-08-28
**Status:** Accepted

**Decision:**
Replace the 4 positional `Symptom_1-4` columns and the 9 legacy Yes/No flag columns
(`Vomiting`, `Diarrhea`, ...) with a single set of 31 binary `Symptom_<name>` columns (one
per distinct symptom string found in the data), set to 1 if that symptom appears anywhere in
a row's 4 slots.

**Reason:**
- The 9 legacy flags covered only 9 of 31 distinct symptom strings actually present in
  `Symptom_1-4` — symptoms like `Fever`, `Lethargy`, `Seizures`, `Weight Loss` had no column.
- Positional slots implied a false ordering (no real "1st" vs "4th" symptom).
- Cross-checked derived columns against the 9 legacy flags: 0 mismatches, confirming they're
  a redundant subset — keeping both would add duplicated, collinear signal with no benefit.
- Added `Symptom_Count` (total symptoms reported) as a candidate severity feature.

---

## ADR-013: Duration Parsed to Continuous Days, Not Ordinal Categories
**Date:** 2026-08-28
**Status:** Accepted

**Decision:**
Parse the free-text `Duration` column (`"2 days"`, `"3 weeks"`, `"A few hours"`, ...) into a
single continuous `Duration_Days` numeric feature instead of ordinal/categorical encoding of
the 18 distinct strings.

**Reason:**
Ordinal/categorical encoding would treat each of the 18 strings as a label with no
inherent magnitude relationship. Parsing to days preserves the true "more time elapsed =
larger number" relationship, letting a model learn a smooth risk-vs-duration signal instead
of 18 unrelated categories.

---

## ADR-014: Breed Dropped from Feature Set; Gender Kept
**Date:** 2026-08-29
**Status:** Accepted

**Decision:**
Drop `Breed` (25 distinct values) from the model features entirely — not one-hot encoded, not
kept in any form. `Gender` is kept and one-hot encoded alongside `Animal_Type`.

**Reason:**
- Tested with Cramér's V (association strength, 0-1): `Breed` scored ~0.12 against both
  Disease_Category and Urgency — barely above negligible. `Animal_Type`, with only 2 possible
  values vs. Breed's 25, scored *higher* (~0.17) against Disease_Category — a cleaner signal
  with far less complexity. The p-values for Breed were technically "significant" but with
  2,200 rows spread across 25 categories that's expected from noise alone; effect size
  (Cramér's V) is what matters, not just detectability.
- Structural generalization problem independent of the weak signal: the 25 breeds present are
  only what this synthetic batch happened to generate. Real input will include breeds never
  seen in training; even with `handle_unknown='ignore'` preventing a crash, an unseen breed
  becomes an all-zero one-hot row — meaning Breed could only ever help for these exact 25
  breeds, never generalizing further.
- `Gender` kept despite not being tested the same way — engineer's call based on direct
  clinical reasoning (e.g. ovarian conditions only apply to female animals), independent of
  what association tests on this synthetic sample show. Only 2 possible values, so no
  generalization concern like Breed.

**Effect:** `train.csv`/`val.csv`/`test.csv` reduced from 70 to 45 columns.
