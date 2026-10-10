# Allow running directly: `python src/predict.py ...`
if __name__ == "__main__" and __package__ is None:
    import sys, pathlib
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
    __package__ = "src"

import argparse
import joblib
import pandas as pd
from src.data import load_config, download_raw
from src.features import basic_game_features, project_spreads

def main(season: int, week: int, threshold: float | None = None):
    cfg = load_config()
    paths = cfg["paths"]
    prob_thresh = threshold if threshold is not None else float(cfg["output"].get("prob_threshold", 0.5))

    raw = download_raw(season, paths)
    # Weeks whose betting lines aren't out yet get a spread projected from the
    # lines posted in earlier weeks, then go through the same model.
    schedule = project_spreads(raw["schedule"], before_week=week)
    feats = basic_game_features(schedule)

    # Filter to requested week
    upcoming = feats[feats["week" ] == week].copy()
    if upcoming.empty:
        print(f"No games found for season={season}, week={week}.")
        return

    # Predict home win probability
    X = upcoming[["spread_home", "is_home"]].values
    model = joblib.load(f"{paths['artifacts']}/baseline_logreg.pkl")
    proba = model.predict_proba(X)[:, 1]  # P(home wins)

    # Add preds + winner
    upcoming["home_win_prob"] = proba
    upcoming["pred_home_win"] = (upcoming["home_win_prob"] >= prob_thresh)
    upcoming["predicted_winner"] = upcoming.apply(
        lambda r: r["home_team"] if r["pred_home_win"] else r["away_team"], axis=1
    )
    upcoming["confidence"] = (upcoming["home_win_prob"] - (1 - upcoming["home_win_prob"])).abs()  # distance from 0.5

    # Sort by confidence (strongest picks first)
    upcoming.sort_values("confidence", ascending=False, inplace=True)

    # Save & display
    out_path = f"{paths['processed']}/predictions_{season}_wk{week}.csv"
    cols = ["week", "home_team", "away_team", "spread_home", "spread_source", "home_win_prob", "predicted_winner", "confidence"]
    upcoming.to_csv(out_path, index=False)

    # Pretty print
    view = upcoming[cols].copy()
    view["home_win_prob"] = view["home_win_prob"].round(3)
    view["confidence"] = view["confidence"].round(3)
    n_proj = int((upcoming["spread_source"] == "projected").sum())
    if n_proj:
        print(f"Note: {n_proj}/{len(upcoming)} games use a projected spread (no posted line yet).")
    print(f"Threshold = {prob_thresh}. Saved predictions to {out_path}\n")
    print(view.to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--threshold", type=float, default=None, help="Override config prob threshold (e.g., 0.55)")
    args = parser.parse_args()
    main(args.season, args.week, threshold=args.threshold)
