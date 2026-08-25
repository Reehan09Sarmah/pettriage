# PetTriage 🐾

> A production ML system that triages pet health symptoms and learns from user feedback.

![CI](https://github.com/YOUR_USERNAME/pettriage/actions/workflows/ci.yml/badge.svg)

## What It Does

Users input their pet's symptoms, breed, age, and type.  
The system returns a likely condition and urgency level — **Emergency / Monitor / Okay**.  
User feedback is collected and the model retrains weekly, auto-deploying only if it improves.

## Architecture

```
[Symptoms + Pet Info]
        ↓
[Condition Classifier]     → probability distribution over conditions
        ↓
[Urgency Classifier]       → Emergency / Monitor / Okay
        ↓
[User Feedback]            → weekly retraining pipeline
```

## Setup

```bash
# Create and activate virtual environment
uv venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Mac/Linux

# Install dependencies
uv pip install -r requirements.txt
uv pip install -r requirements-dev.txt

# Install the package in editable mode
uv pip install -e .
```

## Running Tests

```bash
pytest tests/ -v
```

## Linting

```bash
ruff check src/ tests/
```

## Project Structure

```
pettriage/
├── src/pettriage/      # source package
│   ├── config.py       # central config — paths, settings
│   ├── data/           # preprocessing logic
│   ├── models/         # training + inference
│   └── api/            # FastAPI endpoints
├── tests/              # pytest test suite
├── notebooks/          # exploration only
├── data/               # raw + processed (gitignored)
└── docs/               # context, decisions, progress
```
