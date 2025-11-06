from pathlib import Path
import re
import glob

# Determine base data path (works both locally and on Streamlit Cloud)
BASE_DIR = Path(__file__).resolve().parents[1]  # -> Season25
DATA_DIR = BASE_DIR / "data" / "processed"

# Find prediction files
raw_files = glob.glob(str(DATA_DIR / "predictions_*_wk*.csv"))

if not raw_files:
    st.info("No predictions found. Make sure CSVs are committed under Season25/data/processed/")
    st.stop()

# Parse and sort by week number
pairs = []
for f in raw_files:
    m = re.search(r'_wk(\d+)\.csv$', f)
    if m:
        pairs.append((int(m.group(1)), f))
pairs.sort(key=lambda x: x[0])

weeks = [w for w, _ in pairs]
labels = [f"Week {w}" for w in weeks]
default_idx = len(labels) - 1

week_choice = st.selectbox("Pick a week:", labels, index=default_idx)
fn = dict(zip(labels, [p for _, p in pairs]))[week_choice]
df = pd.read_csv(fn)

st.caption(f"📄 Loaded: {fn}")



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
df["Matchup"] = df["away_team"].astype(str) + " @ " + df["home_team"].astype(str)

# Sorting options
sort_by = st.sidebar.radio("Sort by", options=["confidence (desc)", "predicted win prob (desc)", "home win prob (desc)"])
if sort_by == "confidence (desc)":
    df = df.sort_values("confidence", ascending=False)
elif sort_by == "predicted win prob (desc)":
    df = df.sort_values("predicted_win_prob", ascending=False)
else:
    df = df.sort_values("home_win_prob", ascending=False)

# Convert probabilities to percentage strings rounded to one decimal
def fmt_percent(x):
    return (x * 100).round(1)

# Display table
st.subheader("Predictions")

# Format percentages
df_display = df.copy()
df_display["Confidence %"] = (df_display["confidence"] * 100).round(1).astype(str) + "%"

# NEW: combine winner + win prob into one column
df_display["Predicted Winner — Win Probability %"] = (
    df_display["predicted_winner"].astype(str)
    + " — "
    + (df_display["predicted_win_prob"] * 100).round(1).astype(str)
    + "%"
)

# Build final table view
st.dataframe(
    df_display[["Matchup", "Predicted Winner — Win Probability %", "Confidence %"]],
    use_container_width=True
)

# Bar chart: Predicted winner probability per game
st.subheader("Predicted Winner Probability (by matchup)")

df_vis = df.assign(
    predicted_win_prob_pct=(df["predicted_win_prob"] * 100).round(1)
)

hover_cols = {
    "Predicted Winner": df_vis["predicted_winner"],
    "Home Win %": (df_vis["home_win_prob"] * 100).round(1).astype(str) + "%",
    "Predicted Win %": (df_vis["predicted_win_prob"] * 100).round(1).astype(str) + "%",
    "Confidence %": (df_vis["confidence"] * 100).round(1).astype(str) + "%",
}

fig = px.bar(
    df_vis,
    x="predicted_win_prob_pct",
    y="Matchup",
    orientation="h",
    hover_data=hover_cols,
    text="predicted_winner",
)

fig.update_layout(
    xaxis_title="Predicted Winner Probability (%)",
    yaxis_title="Matchup",
    bargap=0.25,
)

fig.update_traces(textposition="outside", cliponaxis=False)

st.plotly_chart(fig, use_container_width=True)

# Download button for filtered view
csv_bytes = df_display[["Matchup", "Predicted Winner — Win Probability %", "Confidence %"]].to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download this table as CSV",
    data=csv_bytes,
    file_name="predictions_view.csv",
    mime="text/csv"
)
