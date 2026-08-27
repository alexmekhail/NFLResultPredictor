import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from yaml import safe_load
import nfl_data_py as nfl

load_dotenv()

# The season folder (Season26/) — resolved from this file's location, not the
# process cwd, so scripts behave identically whether run from inside the season
# folder or via `python Season26/scripts/x.py` from the repo root.
SEASON_DIR = Path(__file__).resolve().parents[1]


def load_config(path: str = "config.yaml") -> dict:
    """Load config.yaml and resolve every entry under `paths:` to an absolute
    path anchored at the season folder (SEASON_DIR), regardless of cwd."""
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = SEASON_DIR / cfg_path

    with open(cfg_path, "r") as f:
        cfg = safe_load(f)

    paths = cfg.get("paths", {})
    resolved = {}
    for key, value in paths.items():
        p = Path(value)
        resolved[key] = str(p if p.is_absolute() else (SEASON_DIR / p))
    cfg["paths"] = resolved
    return cfg

def ensure_dirs(paths: dict):
    for p in paths.values():
        Path(p).mkdir(parents=True, exist_ok=True)

def get_season_schedule(season: int) -> pd.DataFrame:
    # Includes game_id, week, teams, scores, spreads, etc.
    sched = nfl.import_schedules([season])
    return sched

def get_schedules(seasons) -> pd.DataFrame:
    seasons = [int(s) for s in seasons]
    return nfl.import_schedules(seasons)

def save_csv(df: pd.DataFrame, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def download_raw(season: int, paths: dict) -> dict:
    schedule = get_season_schedule(season)
    save_csv(schedule, f"{paths['raw']}/schedule_{season}.csv")
    return {"schedule": schedule}

def download_training_raw(seasons, paths: dict) -> dict:
    seasons = [int(s) for s in seasons]
    schedule = get_schedules(seasons)
    tag = "_".join(str(s) for s in seasons)
    save_csv(schedule, f"{paths['raw']}/schedule_train_{tag}.csv")
    return {"schedule": schedule}
