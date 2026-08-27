# PetTriage — Progress Tracker

> Only what's actually done goes here. Updated at end of every session.

---

## Environment
- [x] Python 3.12 confirmed
- [x] `uv` installed
- [x] `.venv` created and activated
- [x] `requirements.txt` written and installed
- [x] `requirements-dev.txt` written and installed
- [x] `pyproject.toml` created — enables `pip install -e .`
- [x] Editable install: `uv pip install -e .`

## Project Structure
- [x] `src/pettriage/` package with submodules (data, models, api)
- [x] `tests/`, `notebooks/`, `data/raw/`, `data/processed/` created
- [x] `src/pettriage/config.py` — central paths and settings
- [x] `docs/` folder (CONTEXT, DECISIONS, PROGRESS)
- [x] `.agents/rules/pettriage.md` harness

## Git & GitHub
- [x] `git init` in `pettriage/`
- [x] `.gitignore` created
- [x] `README.md` created
- [x] GitHub public repo created and pushed
- [x] `tests/test_placeholder.py` placeholder test

## CI
- [x] `.github/workflows/ci.yml` — lint + test on every push
- [x] CI running on GitHub Actions

## Data
- [x] EDA notebook (`01_data_exploration.ipynb`) — run and analyzed
- [x] Data prep notebook (`02_data_preparation.ipynb`) — run, output saved
- [x] `data/processed/dogs_cats_labeled.csv` — 147 seed rows
- [x] `data/raw/synthetic/` — 11 LLM-generated CSVs (2,200 rows total)
- [x] `docs/LLM_GENERATION_PROMPT.md` — prompt template for synthetic generation
- [x] `notebooks/03_data_merge_validation.ipynb` — created and run
- [x] `data/processed/training_data.csv` — 2,200 rows, validated (schema/value/flag-consistency checks all clean, 0 issues found)
- [x] `notebooks/04_feature_engineering.ipynb` — created and run, with in-notebook plain-language explanations for every step
- [x] `data/processed/train.csv` / `val.csv` / `test.csv` — 1,540 / 330 / 330 rows, 45 model-ready columns each, stratified split (Urgency proportions preserved within ~1% across splits); `Breed` dropped after a Cramér's V check showed weak signal + poor real-world generalization, `Gender` kept on clinical grounds
