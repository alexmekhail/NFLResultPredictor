import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from yaml import safe_load
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score
from src.data import load_config, download_training_raw
from src.features import basic_game_features

import joblib

def main():
    load_dotenv()
    cfg = load_config()
    paths = cfg["paths"]
    predict_season = int(cfg["season"])
    train_seasons = [int(s) for s in (cfg.get("train_seasons") or [predict_season])]

    Path(paths["artifacts"]).mkdir(parents=True, exist_ok=True)
    Path(paths["reports"]).mkdir(parents=True, exist_ok=True)

    raw = download_training_raw(train_seasons, paths)
    feats = basic_game_features(raw["schedule"])
    # Only completed games carry a usable label.
    feats = feats[feats["home_win"].notna()].copy()
    if feats.empty:
        raise SystemExit(
            f"No completed games found for train_seasons={train_seasons}. "
            "Set train_seasons in config.yaml to a season that has been played."
        )

    X = feats[["spread_home", "is_home"]].values
    y = feats["home_win"].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["model"]["test_size"], random_state=cfg["model"]["random_state"], stratify=y
    )

    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(max_iter=1000, class_weight=cfg["model"]["class_weight"]))
    ])
    pipe.fit(X_train, y_train)

    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= cfg["output"]["prob_threshold"]).astype(int)

    report = {
        "train_seasons": train_seasons,
        "predict_season": predict_season,
        "n_train": int(len(y_train)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "n_test": int(len(y_test))
    }

    with open(f"{paths['reports']}/baseline_metrics.txt", "w") as f:
        f.write(str(report))

    joblib.dump(pipe, f"{paths['artifacts']}/baseline_logreg.pkl")
    print("Training complete. Metrics:", report)

if __name__ == "__main__":
    main()
