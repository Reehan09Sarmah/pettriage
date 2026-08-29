# PetTriage — Session Context

> **Update this file at the end of every session.**
> When starting a new conversation, paste this file to the agent.

---

## Project in One Line
A production ML system that triages pet health symptoms (dogs and cats), learns from user feedback, and auto-retrains weekly.

## Current Phase
**Phase 1 — Foundation**
Goal: Working model + API + CI running locally. No cloud yet.

## What's Built

### Environment & Setup
- [x] `.venv` created via `uv venv`, activated
- [x] `requirements.txt` and `requirements-dev.txt` written and installed
- [x] `pyproject.toml` — package metadata, enables `pip install -e .`
- [x] Editable install done: `uv pip install -e .`
- [x] `src/pettriage/` package structure with `__init__.py` files
- [x] `tests/`, `notebooks/`, `data/raw/`, `data/processed/` folders created
- [x] `src/pettriage/config.py` — central config with all paths

### Project Governance
- [x] `.agents/rules/pettriage.md` — agent harness (auto-loads in Antigravity)
- [x] `docs/CONTEXT.md`, `docs/DECISIONS.md`, `docs/PROGRESS.md`
- [x] `docs/LLM_GENERATION_PROMPT.md` — prompt used to generate synthetic data

### Git & CI
- [x] Git repo initialized inside `pettriage/`
- [x] `.gitignore` created
- [x] GitHub public repo created and pushed
- [x] `.github/workflows/ci.yml` — runs ruff + pytest on every push
- [x] `tests/test_placeholder.py` — placeholder test (CI passes)
- [x] `README.md` with architecture overview

### Data
- [x] 3 Kaggle datasets downloaded to `data/raw/`
- [x] `notebooks/01_data_exploration.ipynb` — EDA on all 3 datasets
- [x] `notebooks/02_data_preparation.ipynb` — filter to dogs/cats, map 48 diseases → 9 categories, clean temperature, save seed data
- [x] `data/processed/dogs_cats_labeled.csv` — 147 seed rows, 9 categories, no Urgency label
- [x] `src/pettriage/data/generate_synthetic.py` — seeded programmatic generator, replaces the
      manual LLM-prompt workflow (ADR-016, ADR-019). Knobs: `--total-rows` (default 30,000),
      `--ambiguity`, `--missing-rate` (0.08), `--label-noise` (0.0), `--balanced`, `--seed`
- [x] `data/raw/synthetic/` — 11 category CSVs, ~30,000 rows, split by real-world prevalence
      (Skin/GI ~4,500 each down to Viral Systemic ~900), 8% missing vitals, no label noise
- [x] `notebooks/03_data_merge_validation.ipynb` — 5 checks; Check 5 now classifies every
      missing value by *cause* (imputable / structural / required) using the shared policy
      imported from `preprocess.py`
- [x] `data/processed/training_data.csv` — ~30,000 rows, 11 categories, prevalence-weighted
- [x] `notebooks/04_feature_engineering.ipynb` — the exploratory version of the recipe now
      ported into `preprocess.py` (kept as the plain-language explanation of each step)
- [x] `src/pettriage/data/preprocess.py` — the reusable port. `fit_transform()` (train only),
      `transform()` (val/test/live), `save_fitted()`/`load_fitted()`, plus the three-way
      missing-value policy and `audit_missing_values()`
- [x] `data/processed/train.csv` / `val.csv` / `test.csv` — 21,000 / 4,500 / 4,500 rows,
      45 columns, stratified on combined Category+Urgency key
- [x] `models/preprocessor.joblib` — fitted encoders, scaler, imputer, symptom vocabulary
- [x] `src/pettriage/models/train_model_a.py` — model *selection*: trains 4 candidates,
      picks the champion on macro F1, logs each as a top-level MLflow run with per-class F1,
      confusion matrix and classification report as artifacts
- [x] `models/model_a.joblib` — current champion (HistGradientBoosting, macro F1 0.647)

## Current Model A Results (validation, 4,500 rows)
| Model | train | val | gap | top-3 | macro F1 |
|---|---|---|---|---|---|
| hist_gradient_boosting **(champion)** | 0.769 | 0.702 | 0.067 | 0.909 | **0.647** |
| random_forest | 0.856 | 0.686 | 0.169 | 0.901 | 0.640 |
| baseline_logreg | 0.707 | 0.694 | 0.012 | 0.910 | 0.639 |
| baseline_logreg_balanced | 0.688 | 0.672 | 0.015 | 0.902 | 0.636 |

All four within 0.011 — a statistical tie, so "HistGB wins" is provisional. Weakest classes:
Viral Systemic (F1 0.312, top-3 recall 0.711) and Endocrine/Metabolic (0.492).

## What's Not Built Yet
- [ ] Hyperparameter tuning — `RandomizedSearchCV`, `f1_macro`, CV on train (ADR-021) **← next**
- [ ] `src/pettriage/models/train_model_b.py` — Urgency classifier; probability-extraction
      method still undecided (ADR-022)
- [ ] `src/pettriage/models/predict.py` — load artifacts, run inference, stage-2 decision rule
      (ADR-020) + probability calibration check
- [ ] `src/pettriage/api/app.py` — FastAPI `/predict` and `/feedback` endpoints
- [ ] Real tests in `tests/` (beyond placeholder)
- [ ] Docker (install Docker Desktop first)
- [ ] Cloud deployment (GCP Cloud Run — Phase 2)

## Active Decisions (see DECISIONS.md for full reasoning)
- Two classifiers: Condition (11-class) + Urgency (3-class soft cascade)
- Urgency classifier receives Model A's probability distribution, NOT hard label
- Phase 1 ML: scikit-learn only
- Experiment tracking: MLflow (local)
- 11 disease categories: Respiratory, Gastrointestinal, Viral Systemic, Bacterial/Parasitic, Skin/Fungal, Musculoskeletal, Eye/Ear, Renal/Urinary, Endocrine/Metabolic, Dental/Oral, Trauma/Poisoning
- Synthetic data only for Phase 1 training; seed data (no Urgency labels) not merged
- No PetMD scraping — LLM generates equivalent data legally
- MCP server deferred to Phase 2+
- Train/val/test split (70/15/15) stratified on combined Category+Urgency key
- Symptoms represented as 31-column multi-hot (replaces positional `Symptom_1-4` + 9 legacy flags)
- `Duration` parsed to continuous `Duration_Days`, not ordinal categories
- All statistical transforms (scaler, one-hot encoder) fit on train split only — never on val/test
- `Breed` dropped from features (weak Cramér's V signal, doesn't generalize past the 25 breeds seen); `Gender` kept on clinical grounds
- Prediction output shows top category + only genuinely close alternatives (margin-based, not fixed top-N) — ADR-015, applies once `predict.py`/API is built
- Synthetic data is generated programmatically, not by LLM prompts — seeded, reproducible, tunable (ADR-016)
- Categories split by real-world prevalence, not evenly — real triage queues are imbalanced (ADR-017)
- Missing values classified by cause into three lists in `preprocess.py`: imputable (fill with train median) / structural (never fill — an animal with 3 symptoms has no 4th) / required (fail loudly) — ADR-018
- Model A champion selected on **macro F1**, not accuracy — accuracy rewards neglecting rare classes (ADR-018)
- Label noise generation OFF by default — nothing in the pipeline handles it, so it was pure handicap (ADR-019)
- Two-stage prediction: model outputs probabilities, a separate decision rule sets the precision/recall tradeoff — no retraining needed to change it (ADR-020)
- Hyperparameter tuning uses cross-validation on **train**, never tuning against val (ADR-021)

## Stack
| Layer | Tool |
|---|---|
| Language | Python 3.12 |
| ML | scikit-learn |
| API | FastAPI + uvicorn |
| Tracking | MLflow (local) |
| Validation | pydantic |
| Lint | ruff |
| Tests | pytest + httpx |
| CI | GitHub Actions (running) |
| Cloud | GCP Cloud Run (Phase 2) |

## Working Style Notes
- Agent implements/edits code; engineer runs notebooks themselves rather than the agent executing and narrating output.
- Notebook markdown cells should carry full plain-language explanations (not just terse technical notes) — notebooks are a learning reference, not just code.
- Explain concepts in simple terms/analogies first, technical vocabulary second.

## How to Reproduce Everything
```
python -m pettriage.data.generate_synthetic     # -> data/raw/synthetic/*.csv
# then run notebooks/03_data_merge_validation.ipynb   -> data/processed/training_data.csv
python -m pettriage.data.preprocess              # -> train/val/test.csv + preprocessor.joblib
python -m pettriage.models.train_model_a         # -> model_a.joblib + MLflow runs
```
Browse experiments with **`mlflow ui --backend-store-uri sqlite:///mlflow.db`** — the
`--backend-store-uri` flag is required; plain `mlflow ui` reads an empty `./mlruns` and shows
nothing. Then open http://localhost:5000 and pick the `pettriage` experiment.

## Immediate Next Steps
1. Hyperparameter tuning for Model A (ADR-021) — `RandomizedSearchCV` with `f1_macro` and
   cross-validation on the train split; add `class_weight` to the HistGB search space (it is
   currently the only candidate not setting it); log best params/scores to MLflow
2. Then Model B — first settle the probability-extraction question (ADR-022): in-sample vs
   cross-fitted out-of-fold probabilities as Model B's cascade features
3. Then `predict.py` with the stage-2 decision rule (ADR-020) and a calibration check

## Open Threads / Reading
- A long explanation of precision vs recall vs macro F1, why `class_weight` raised recall but
  lowered macro F1, and how two-stage decisions work in real systems (mammography AI, fraud
  detection, sepsis alerts) was written out in the session transcript — engineer flagged it to
  read later. The conclusions are captured in ADR-020.
- `macro recall` and `balanced accuracy` are the **same metric** (sklearn's
  `balanced_accuracy_score` is defined as mean per-class recall).
