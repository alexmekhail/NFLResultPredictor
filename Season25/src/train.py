import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from yaml import safe_load
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score
from src.data import load_config, download_raw
from src.features import basic_game_features

import joblib

def main():
    load_dotenv()
    cfg = load_config()
    paths = cfg["paths"]
    season = int(cfg["season"])

    Path(paths["artifacts"]).mkdir(parents=True, exist_ok=True)
    Path(paths["reports"]).mkdir(parents=True, exist_ok=True)

    raw = download_raw(season, paths)
    feats = basic_game_features(raw["schedule"])

    X = feats[["spread_home", "is_home"]].values
    y = feats["home_win"].values

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
