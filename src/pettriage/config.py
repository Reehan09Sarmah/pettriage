"""
Central configuration for PetTriage.
All paths and settings live here. Import from this module everywhere else.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
# __file__ is src/pettriage/config.py
# parents[0] = src/pettriage/
# parents[1] = src/
# parents[2] = project root (pettriage/)
ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)  # create on import if missing

# ── Model settings ─────────────────────────────────────────────────────────────
# These will grow as we build. Change here, changes everywhere.
RANDOM_STATE = 42
TEST_SIZE = 0.2  # 20% of data held out for evaluation

# ── MLflow ─────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = str(ROOT_DIR / "mlruns")
MLFLOW_EXPERIMENT_NAME = "pettriage"
