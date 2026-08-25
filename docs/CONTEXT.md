# PetTriage — Session Context

> **Update this file at the end of every session.**
> When starting a new conversation, paste this file to the agent.

---

## Project in One Line
A production ML system that triages pet health symptoms, learns from user feedback, and auto-retrains weekly.

## Current Phase
**Phase 1 — Foundation**
Goal: Working model + API + CI running locally. No cloud yet.

## What's Built
- [x] `.venv` created via `uv venv`
- [x] `requirements.txt` and `requirements-dev.txt` written
- [x] Dependencies installed via `uv pip install`
- [x] `docs/` and `.agents/rules/` folder structure created

## What's Not Built Yet
- [ ] Project folder structure (`src/`, `tests/`, `notebooks/`, `data/`)
- [ ] Git init + GitHub repo + first push
- [ ] `config.py` — central config file
- [ ] Data collection (Kaggle search or synthetic generation)
- [ ] Baseline condition classifier
- [ ] Urgency classifier (soft cascade)
- [ ] MLflow experiment tracking setup
- [ ] FastAPI `/predict` endpoint
- [ ] Docker setup
- [ ] GitHub Actions CI

## Active Decisions (see DECISIONS.md for full reasoning)
- Two separate classifiers: Condition (multi-class) + Urgency (3-class soft cascade)
- Urgency classifier receives Model A's **probability distribution**, NOT hard label
- Phase 1 ML: scikit-learn only. Deep learning deferred.
- Experiment tracking: MLflow (local, open source)
- Package manager: `uv`
- User feedback: thumbs up/down. If wrong → user selects correct urgency from list.
- Condition feedback treated as noisy; urgency feedback treated as high-trust signal.

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
| CI | GitHub Actions (not set up yet) |
| Cloud | GCP Cloud Run (Phase 2) |

## Last Session Summary
- Discussed project concept, domain selection (Pet Healthcare), architecture
- Designed feedback loop and soft cascade architecture
- Created requirements files and installed dependencies
- Decided on harness + context management system
- Set up docs/ structure
