import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from yaml import safe_load
import nfl_data_py as nfl

load_dotenv()

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return safe_load(f)

def ensure_dirs(paths: dict):
    for p in paths.values():
        Path(p).mkdir(parents=True, exist_ok=True)

def get_season_schedule(season: int) -> pd.DataFrame:
    # Includes game_id, week, teams, scores, spreads, etc.
    sched = nfl.import_schedules([season])
    return sched

def save_csv(df: pd.DataFrame, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def download_raw(season: int, paths: dict) -> dict:
    schedule = get_season_schedule(season)
    save_csv(schedule, f"{paths['raw']}/schedule_{season}.csv")
    return {"schedule": schedule}
