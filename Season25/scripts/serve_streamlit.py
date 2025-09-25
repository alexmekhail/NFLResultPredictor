import glob
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="NFL Result Predictor", layout="wide")
st.title("🏈 NFL Result Predictor — Weekly Picks")

# Find prediction files
files = sorted(glob.glob("data/processed/predictions_*_wk*.csv"))
if not files:
    st.info("No predictions found. Run `python src/predict.py --season 2025 --week <n>` first.")
    st.stop()

# Pick file
fn = st.selectbox("Pick a predictions file", files, index=len(files) - 1)
df = pd.read_csv(fn)

# If predicted_winner is missing, derive it on the fly using a threshold
threshold = st.sidebar.slider("Pick threshold for a winner (if needed)", 0.40, 0.60, 0.50, 0.01)

if "predicted_winner" not in df.columns:
    df["predicted_winner"] = np.where(df["home_win_prob"] >= threshold, df["home_team"], df["away_team"])

# Compute probability for the predicted winner (handles both home/away winners)
df["predicted_win_prob"] = np.where(
    df["predicted_winner"] == df.get("home_team", ""),
    df["home_win_prob"],
    1 - df["home_win_prob"]
)

# Confidence (distance from 0.5) if not already present
if "confidence" not in df.columns:
    df["confidence"] = (df["predicted_win_prob"] - 0.5).abs() * 2  # scale 0..1 (0.5→0, 1.0→1)

# Week filter (if multiple weeks exist in the file)
if "week" in df.columns:
    weeks = sorted(df["week"].dropna().unique().tolist())
    sel_week = st.sidebar.selectbox("Filter by week", options=weeks, index=0)
    df = df[df["week"] == sel_week].copy()

# Make a nice matchup label
df["matchup"] = df["away_team"].astype(str) + " @ " + df["home_team"].astype(str)

# Sorting options
sort_by = st.sidebar.radio("Sort by", options=["confidence (desc)", "predicted win prob (desc)", "home win prob (desc)"])
if sort_by == "confidence (desc)":
    df = df.sort_values("confidence", ascending=False)
elif sort_by == "predicted win prob (desc)":
    df = df.sort_values("predicted_win_prob", ascending=False)
else:
    df = df.sort_values("home_win_prob", ascending=False)

# Main table
display_cols = [c for c in ["week", "matchup", "home_team", "away_team",
                            "home_win_prob", "predicted_winner", "predicted_win_prob", "confidence"] if c in df.columns]
st.subheader("Predictions")
st.dataframe(
    df[display_cols].assign(
        home_win_prob=lambda x: (x["home_win_prob"]).round(3),
        predicted_win_prob=lambda x: (x["predicted_win_prob"]).round(3),
        confidence=lambda x: (x["confidence"]).round(3),
    ),
    use_container_width=True
)

# Bar chart: Predicted winner probability per game
st.subheader("Predicted Winner Probability (by matchup)")
fig = px.bar(
    df,
    x="predicted_win_prob",
    y="matchup",
    orientation="h",
    hover_data={"predicted_winner": True, "home_win_prob": True, "confidence": True},
    text=df["predicted_winner"],
)
fig.update_layout(xaxis_title="Predicted Winner Probability", yaxis_title="Matchup", bargap=0.25)
fig.update_traces(textposition="outside", cliponaxis=False)
st.plotly_chart(fig, use_container_width=True)

# Optional: Summary table of picks
if "predicted_winner" in df.columns:
    st.subheader("Summary of Predicted Winners")
    summary = df["predicted_winner"].value_counts().reset_index()
    summary.columns = ["Team", "Predicted Wins (this selection)"]
    st.table(summary)

# Download button for the filtered view
csv_bytes = df[display_cols].to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download this table as CSV", data=csv_bytes, file_name="predictions_view.csv", mime="text/csv")
