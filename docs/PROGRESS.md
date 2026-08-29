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
- [x] `Breed` dropped after a Cramér's V check showed weak signal + poor real-world generalization; `Gender` kept on clinical grounds
- [x] **Diagnosed the first dataset as fake** — Model A's 96.4% accuracy was an artifact: 79.5% of symptom combinations mapped to exactly one category (a lookup table). Ablation: symptoms alone 90.9%, vitals alone 26.4%. Probabilities were effectively one-hot (89% of rows above 0.99 confidence), making the soft cascade untestable
- [x] `src/pettriage/data/generate_synthetic.py` — seeded programmatic generator replacing the manual LLM-prompt workflow; overlapping per-category symptom profiles, tunable ambiguity/missingness/label-noise
- [x] Second diagnosis + fix — cardinal symptoms had been given *lower* weights than the shared generic ones, so Viral Systemic ↔ Bacterial/Parasitic hit 0.914 cosine similarity and F1 0.209. Rewrote all 11 profiles so defining signs dominate; similarity fell to 0.639 and every class improved (macro F1 0.554 → 0.636)
- [x] `data/raw/synthetic/` — regenerated: ~30,000 rows, real-world prevalence imbalance (~4.5x), 8% missing vitals, label noise off
- [x] `src/pettriage/data/preprocess.py` — reusable port of notebook 04; fit-on-train-only discipline, median imputation, three-way missing-value policy, `audit_missing_values()`, stratification robust to rare strata
- [x] `notebooks/03_data_merge_validation.ipynb` — Check 5 rewritten to classify missing values by cause instead of flagging structural absence as a defect
- [x] `data/processed/train.csv` / `val.csv` / `test.csv` — 21,000 / 4,500 / 4,500 rows, 45 columns each

## Models
- [x] `src/pettriage/models/train_model_a.py` — model selection across 4 candidates, champion chosen on macro F1, each logged as a top-level MLflow run with per-class F1 + confusion matrix + classification report artifacts
- [x] Learning-curve diagnostic established that 53% accuracy was *not* the problem's ceiling — the error was reducible variance, not irreducible class overlap
- [x] Model A trained: champion HistGradientBoosting — macro F1 0.647, accuracy 0.702, top-3 0.909
- [x] `models/model_a.joblib` + `models/preprocessor.joblib` saved (gitignored)

## MLflow
- [x] SQLite backend configured (`sqlite:///mlflow.db`) — the file-store backend is in maintenance mode in MLflow 3.x
- [x] Fixed Windows tracking-URI bug (`.as_uri()`, not a bare `E:\...` path)
- [x] Runs restructured flat (not nested) so the runs table compares them directly and every run has its own metric charts
- [x] Verified: `mlflow ui --backend-store-uri sqlite:///mlflow.db`

## Not Done Yet
- [ ] Hyperparameter tuning (ADR-021) — next session
- [ ] Model B / Urgency classifier (ADR-022 — extraction method undecided)
- [ ] `predict.py` with stage-2 decision rule (ADR-020) + calibration check
- [ ] FastAPI app, Docker, cloud deployment
