# Ensure the season folder is importable (so `src` resolves) regardless of
# whether this is run from inside Season26/ or as `python Season26/scripts/get_week.py`.
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
from src.predict import main as predict_main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    args = parser.parse_args()
    predict_main(args.season, args.week)
