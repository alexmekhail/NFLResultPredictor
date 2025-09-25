import pandas as pd

def basic_game_features(schedule: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal baseline:
      - Home team indicator
      - Spread if available
      - Outcome label
    """
    df = schedule.copy()

    # Label: 1 = home win, 0 = away win
    if "home_score" in df.columns and "away_score" in df.columns:
        df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)

    # Feature: spread (fallback = 0 if not present)
    if "spread_line" in df.columns:
        df["spread_home"] = -df["spread_line"]  # convert to home perspective
    else:
        df["spread_home"] = 0.0

    df["is_home"] = 1  # since we’re predicting home perspective
    return df[["game_id", "week", "home_team", "away_team", "home_win", "spread_home", "is_home"]].dropna()
