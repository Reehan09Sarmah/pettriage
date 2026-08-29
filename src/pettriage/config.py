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
# SQLite backend, not a plain mlruns/ folder: MLflow 3.x put the folder-based
# ("file://") store into maintenance mode and recommends a database backend even
# for single-user local setups. This is still just one local file (mlflow.db),
# no server/account needed — same spirit as the original folder-based plan.
MLFLOW_TRACKING_URI = "sqlite:///" + str(ROOT_DIR / "mlflow.db")
MLFLOW_EXPERIMENT_NAME = "pettriage"
