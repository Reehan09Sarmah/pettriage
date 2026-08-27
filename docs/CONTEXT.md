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
- [x] `data/raw/synthetic/` — 11 LLM-generated CSVs, 200 rows each (2,200 total)
- [x] `notebooks/03_data_merge_validation.ipynb` — run. All validation checks passed clean (0 schema issues, 0 invalid values, 0 flag inconsistencies)
- [x] `data/processed/training_data.csv` — 2,200 rows, 11 categories perfectly balanced (200 each), Urgency: Monitor 47.3% / Okay 26.8% / Emergency 25.9%
- [x] `notebooks/04_feature_engineering.ipynb` — run, with plain-language markdown explanations for every step (stratified split, multi-hot symptoms, duration parsing, Breed value-check, one-hot/scaling, fit-on-train-only discipline)
- [x] `data/processed/train.csv` / `val.csv` / `test.csv` — 1,540 / 330 / 330 rows, 45 columns each, stratified on combined Category+Urgency key, `Breed` dropped (weak signal + doesn't generalize — ADR-014), `Gender` kept (clinical reasoning)

## What's Not Built Yet
- [ ] `src/pettriage/data/preprocess.py` — port notebook 04's recipe into reusable importable code
- [ ] `src/pettriage/models/train.py` — train both classifiers, log to MLflow
- [ ] `src/pettriage/models/predict.py` — load model artifact, run inference
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

## Immediate Next Steps
1. Explain the concept behind `src/pettriage/data/preprocess.py` (porting notebook 04's recipe into reusable, importable code — split logic, symptom vocabulary, duration parser, fitted encoders/scaler)
2. Get engineer confirmation on the approach
3. Implement `preprocess.py`
