# Ensure the season folder is importable (so `src` resolves) regardless of
# whether this is run from inside Season26/ or as `python Season26/scripts/backtest.py`.
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from src.data import load_config, get_schedules
from src.features import basic_game_features

DEFAULT_SEASONS = [2023, 2024, 2025]


def score_season(model, season: int, threshold: float) -> dict:
    """Run the trained model over one completed season and compare its picks
    against the games that were actually played.

    Accuracy = (# games where the predicted winner matched the actual winner)
               / (# completed games).

    A "predicted winner" is the home team when P(home win) >= threshold, else
    the away team. Because the label is `home_win`, that is identical to
    (pred_home_win == home_win), which is what we compute here.
    """
    sched = get_schedules([season])
    feats = basic_game_features(sched)

    # Only completed games have a real outcome to check against.
    feats = feats[feats["home_win"].notna()].copy()
    feats["home_win"] = feats["home_win"].astype(int)

    X = feats[["spread_home", "is_home"]].values
    prob_home = model.predict_proba(X)[:, 1]          # P(home team wins)
    pred_home_win = (prob_home >= threshold).astype(int)

    correct = pred_home_win == feats["home_win"].values
    n = len(feats)

    # Baselines for context.
    always_home = accuracy_score(feats["home_win"], np.ones(n, dtype=int))
    favored_home = (feats["spread_home"].values < 0).astype(int)   # neg spread_home => home favored
    not_pushed = feats["spread_home"].values != 0
    pick_favorite = accuracy_score(
        feats["home_win"].values[not_pushed], favored_home[not_pushed]
    )

    return {
        "season": season,
        "games": n,
        "accuracy": float(correct.mean()),
        "roc_auc": float(roc_auc_score(feats["home_win"], prob_home)),
        "correct": int(correct.sum()),
        "always_pick_home": float(always_home),
        "pick_favorite": float(pick_favorite),
    }


def main(seasons, threshold: float, write_report: bool):
    cfg = load_config()
    model_path = f"{cfg['paths']['artifacts']}/baseline_logreg.pkl"
    model = joblib.load(model_path)

    rows = [score_season(model, s, threshold) for s in seasons]
    df = pd.DataFrame(rows)

    # Pooled accuracy across every season (games weighted equally).
    total_games = int(df["games"].sum())
    total_correct = int(df["correct"].sum())
    pooled_acc = total_correct / total_games

    lines = []
    lines.append(f"Model:     {model_path}")
    lines.append(f"Threshold: {threshold}")
    lines.append(f"Accuracy = correct winner picks / completed games\n")
    lines.append(
        df.assign(
            accuracy=df["accuracy"].map("{:.3f}".format),
            roc_auc=df["roc_auc"].map("{:.3f}".format),
            always_pick_home=df["always_pick_home"].map("{:.3f}".format),
            pick_favorite=df["pick_favorite"].map("{:.3f}".format),
        ).to_string(index=False)
    )
    lines.append("")
    lines.append(
        f"POOLED {min(seasons)}-{max(seasons)}: "
        f"{total_correct}/{total_games} = {pooled_acc:.3f} winner accuracy"
    )
    report = "\n".join(lines)
    print(report)

    if write_report:
        out = f"{cfg['paths']['reports']}/backtest.txt"
        with open(out, "w") as f:
            f.write(report + "\n")
        print(f"\nSaved -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest the trained model against completed seasons.")
    parser.add_argument("--seasons", type=int, nargs="+", default=DEFAULT_SEASONS,
                        help=f"Seasons to score (default: {DEFAULT_SEASONS})")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="P(home win) cutoff for calling the home team the winner")
    parser.add_argument("--no-report", action="store_true",
                        help="Print only; do not write models/reports/backtest.txt")
    args = parser.parse_args()
    main(args.seasons, args.threshold, write_report=not args.no_report)
