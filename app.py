import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, brier_score_loss
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Sports Over/Under ML Predictor",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------- Data Loaders ----------------------

@st.cache_data(ttl=3600*6)
def load_nfl_data(min_season=2021):
    """Load NFL schedules + scores + closing totals from public nflverse source."""
    url = "http://www.habitatring.com/games.csv"
    df = pd.read_csv(url)
    df = df[df["season"] >= min_season].copy()
    # Keep regular + postseason for completeness
    df["total_score"] = df["away_score"] + df["home_score"]
    df["over_hit"] = (df["total_score"] > df["total_line"]).astype(float)
    # Handle pushes (total == line) as 0.5 or drop later
    df.loc[df["total_score"] == df["total_line"], "over_hit"] = 0.5
    df["under_hit"] = 1 - df["over_hit"]
    df["league"] = "NFL"
    # Clean
    df = df.rename(columns={
        "gameday": "date",
        "away_team": "away",
        "home_team": "home",
        "away_score": "away_pts",
        "home_score": "home_pts",
        "total_line": "ou_line",
        "total": "actual_total"
    })
    cols = ["game_id", "season", "week", "date", "away", "home", "away_pts", "home_pts",
            "actual_total", "ou_line", "over_hit", "under_hit", "spread_line", "league", "game_type"]
    return df[[c for c in cols if c in df.columns]]


@st.cache_data(ttl=3600*12)
def load_nba_sample(min_year=2021):
    """Lightweight NBA scores via nba_api (limited to keep free tier / rate limits happy).
    OU lines are estimated from historical averages for demo purposes.
    """
    try:
        from nba_api.stats.endpoints import leaguegamefinder
        # One call for recent seasons (api is season based)
        # To avoid many calls we fetch a couple seasons and note limitation
        frames = []
        for yr in range(min_year, min(min_year + 3, 2026)):  # limit API calls
            season_str = f"{yr}-{str(yr+1)[-2:]}"
            try:
                finder = leaguegamefinder.LeagueGameFinder(
                    season_nullable=season_str,
                    league_id_nullable="00",
                    season_type_nullable="Regular Season"
                )
                gdf = finder.get_data_frames()[0]
                gdf["season"] = yr
                frames.append(gdf)
            except Exception:
                continue
        if not frames:
            return pd.DataFrame()
        raw = pd.concat(frames, ignore_index=True)
        # Aggregate to game level (nba_api returns one row per team)
        games = []
        for gid, grp in raw.groupby("GAME_ID"):
            if len(grp) != 2:
                continue
            home = grp[grp["MATCHUP"].str.contains("vs.")].iloc[0]
            away = grp[grp["MATCHUP"].str.contains("@")].iloc[0]
            total = home["PTS"] + away["PTS"]
            # Estimate OU line ~ historical NBA average ~ 220-230 recently
            est_line = 225.0 + np.random.normal(0, 5)  # demo noise
            over = 1.0 if total > est_line else (0.5 if total == est_line else 0.0)
            games.append({
                "game_id": gid,
                "season": home["season"],
                "date": home["GAME_DATE"],
                "away": away["TEAM_ABBREVIATION"],
                "home": home["TEAM_ABBREVIATION"],
                "away_pts": away["PTS"],
                "home_pts": home["PTS"],
                "actual_total": total,
                "ou_line": round(est_line, 1),
                "over_hit": over,
                "under_hit": 1 - over,
                "league": "NBA",
                "week": None,
                "game_type": "REG"
            })
        return pd.DataFrame(games)
    except Exception as e:
        st.warning(f"NBA live fetch limited: {e}. Using fallback sample.")
        return pd.DataFrame()


@st.cache_data(ttl=3600*12)
def load_mlb_sample(min_year=2021):
    """MLB scores via pybaseball. OU estimated (typical MLB total ~8.5-9)."""
    try:
        from pybaseball import schedule_and_record
        # Limited teams / years to stay light
        teams = ["NYY", "BOS", "LAD", "HOU", "ATL"]  # sample
        frames = []
        for team in teams:
            for yr in range(min_year, min(min_year + 2, 2026)):
                try:
                    sched = schedule_and_record(yr, team)
                    sched["season"] = yr
                    sched["team"] = team
                    frames.append(sched)
                except Exception:
                    continue
        if not frames:
            return pd.DataFrame()
        raw = pd.concat(frames, ignore_index=True)
        # This is team-centric; simplify to unique games roughly
        # For demo we create synthetic game rows from the records
        games = []
        for _, row in raw.iterrows():
            if pd.isna(row.get("R")) or pd.isna(row.get("RA")):
                continue
            total = row["R"] + row["RA"]
            est_line = 8.7 + np.random.normal(0, 0.8)
            over = 1.0 if total > est_line else (0.5 if abs(total - est_line) < 0.1 else 0.0)
            games.append({
                "game_id": f"MLB_{row.get('Date','')}_{row['team']}",
                "season": row["season"],
                "date": str(row.get("Date", "")),
                "away": "OPP",
                "home": row["team"],
                "away_pts": row["RA"],
                "home_pts": row["R"],
                "actual_total": total,
                "ou_line": round(est_line, 1),
                "over_hit": over,
                "under_hit": 1 - over,
                "league": "MLB",
                "week": None,
                "game_type": "REG"
            })
        return pd.DataFrame(games).drop_duplicates(subset=["game_id"]).head(800)
    except Exception as e:
        st.warning(f"MLB fetch limited: {e}")
        return pd.DataFrame()


def get_data(league: str, min_season: int = 2021):
    if league == "NFL":
        return load_nfl_data(min_season)
    elif league == "NBA":
        return load_nba_sample(min_season)
    elif league == "MLB":
        return load_mlb_sample(min_season)
    return pd.DataFrame()


# ---------------------- ML / Probability ----------------------

def compute_historical_over_rate(df: pd.DataFrame) -> float:
    completed = df.dropna(subset=["actual_total", "ou_line"])
    if completed.empty:
        return 0.5
    # Treat pushes as half
    return completed["over_hit"].mean()


def simple_ml_over_prob(df: pd.DataFrame, features_for_next=None):
    """
    Train a lightweight classifier on historical over/under outcomes.
    Returns model accuracy, Brier score, and a next-year style probability.
    """
    data = df.dropna(subset=["actual_total", "ou_line", "over_hit"]).copy()
    data = data[data["over_hit"].isin([0.0, 1.0])]  # drop pure pushes for classification
    if len(data) < 50:
        base = compute_historical_over_rate(df)
        return {
            "model": None,
            "accuracy": None,
            "brier": None,
            "base_over_rate": base,
            "next_year_over_prob": base,
            "message": "Insufficient data for full ML – using historical rate."
        }

    # Simple features
    data["home_adv"] = 1  # placeholder
    data["line_centered"] = data["ou_line"] - data["ou_line"].mean()
    data["season_num"] = data["season"] - data["season"].min()

    X = data[["ou_line", "line_centered", "season_num"]].fillna(0)
    y = data["over_hit"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, preds)
    brier = brier_score_loss(y_test, probs)

    # "Next year" probability: average predicted prob on recent lines, or historical + small trend
    recent = data[data["season"] >= data["season"].max() - 1]
    if len(recent) > 10:
        X_recent = recent[["ou_line", "line_centered", "season_num"]].fillna(0)
        next_prob = model.predict_proba(X_recent)[:, 1].mean()
    else:
        next_prob = y.mean()

    # Slight mean-reversion toward 50% for realism of future
    next_prob = 0.7 * next_prob + 0.3 * 0.5

    return {
        "model": model,
        "accuracy": acc,
        "brier": brier,
        "base_over_rate": y.mean(),
        "next_year_over_prob": float(np.clip(next_prob, 0.35, 0.65)),
        "message": "Logistic regression trained on historical closing lines + season trend."
    }


def projected_total_distribution(df: pd.DataFrame):
    """Return mean and std of actual totals for Gaussian probability calc."""
    completed = df.dropna(subset=["actual_total"])
    if completed.empty:
        return 45.0, 12.0  # NFL-ish defaults
    return completed["actual_total"].mean(), completed["actual_total"].std()


# ---------------------- UI ----------------------

st.title("🏈🏀⚾ Sports Over/Under Machine Learning App")
st.markdown("""
**Last 5 years of game scores + closing over/under lines**  
Predicts the probability that the **over** hits in the next season / upcoming games.  
Data sources: public nflverse (NFL – full scores + real closing totals), nba_api & pybaseball (NBA/MLB scores; OU lines estimated for demo).  
**Free to host** on [Streamlit Community Cloud](https://streamlit.io/cloud) or Hugging Face Spaces.
""")

with st.sidebar:
    st.header("Controls")
    league = st.selectbox("League", ["NFL", "NBA", "MLB"], index=0)
    min_season = st.slider("From season", 2020, 2025, 2021)
    st.caption("NFL has the richest free data (real closing totals). NBA/MLB use live APIs + estimated lines for the ML demo.")
    show_raw = st.checkbox("Show raw game table", value=True)
    st.markdown("---")
    st.markdown("### Deploy free")
    st.markdown("""
1. Push this folder to a public GitHub repo  
2. Go to [share.streamlit.io](https://share.streamlit.io)  
3. Connect the repo → Deploy  
(or Hugging Face Spaces with Streamlit SDK)
    """)

# Load
with st.spinner(f"Loading {league} data (cached)..."):
    df = get_data(league, min_season)

if df.empty:
    st.error("No data returned. Try NFL (most reliable free source) or refresh later.")
    st.stop()

# Basic metrics
completed = df.dropna(subset=["actual_total", "ou_line"])
over_rate = compute_historical_over_rate(df)
mean_total, std_total = projected_total_distribution(df)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Games loaded", f"{len(df):,}")
col2.metric("Completed games", f"{len(completed):,}")
col3.metric("Historical Over Hit %", f"{over_rate*100:.1f}%")
col4.metric("Avg Actual Total", f"{mean_total:.1f}")

# ML section
st.subheader("Machine Learning – Next Year Over Probability")
ml_res = simple_ml_over_prob(df)

c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    st.markdown(f"**Model:** {ml_res['message']}")
    if ml_res["accuracy"] is not None:
        st.write(f"Hold-out accuracy: **{ml_res['accuracy']*100:.1f}%**  |  Brier score: **{ml_res['brier']:.3f}** (lower better)")
with c2:
    st.metric("Base historical Over %", f"{ml_res['base_over_rate']*100:.1f}%")
with c3:
    prob = ml_res["next_year_over_prob"]
    st.metric("Predicted Next-Year Over Probability", f"{prob*100:.1f}%",
              delta=f"{(prob - 0.5)*100:+.1f}% vs 50/50")

# Gauge
fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=prob * 100,
    domain={"x": [0, 1], "y": [0, 1]},
    title={"text": "P(Over) Next Season"},
    gauge={
        "axis": {"range": [0, 100]},
        "bar": {"color": "#1f77b4"},
        "steps": [
            {"range": [0, 40], "color": "#d62728"},
            {"range": [40, 60], "color": "#ff7f0e"},
            {"range": [60, 100], "color": "#2ca02c"}
        ],
        "threshold": {"line": {"color": "black", "width": 2}, "thickness": 0.75, "value": 50}
    }
))
fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
st.plotly_chart(fig_gauge, width='stretch')

# Season-by-season over rates
st.subheader("Over Hit Rate by Season")
if "season" in completed.columns and not completed.empty:
    by_season = completed.groupby("season")["over_hit"].mean().reset_index()
    by_season["over_pct"] = by_season["over_hit"] * 100
    fig = px.bar(by_season, x="season", y="over_pct",
                 labels={"over_pct": "Over Hit %", "season": "Season"},
                 color="over_pct", color_continuous_scale="RdYlGn",
                 range_color=[40, 60])
    fig.add_hline(y=50, line_dash="dash", line_color="gray")
    fig.update_layout(yaxis_range=[30, 70], height=350)
    st.plotly_chart(fig, width='stretch')

# Distribution of totals vs lines
st.subheader("Actual Totals vs Closing Lines")
if not completed.empty:
    fig2 = px.scatter(completed.sample(min(800, len(completed))), 
                      x="ou_line", y="actual_total",
                      color="over_hit",
                      color_continuous_scale="RdYlGn",
                      labels={"ou_line": "Closing Over/Under Line", "actual_total": "Actual Total Score",
                              "over_hit": "Over Hit"},
                      opacity=0.6)
    fig2.add_shape(type="line", x0=completed["ou_line"].min(), y0=completed["ou_line"].min(),
                   x1=completed["ou_line"].max(), y1=completed["ou_line"].max(),
                   line=dict(color="black", dash="dash"))
    st.plotly_chart(fig2, width='stretch')

# Gaussian projection example
st.subheader("Simple Gaussian Projection (for a typical future line)")
typ_line = float(completed["ou_line"].median()) if not completed.empty else (45.0 if league == "NFL" else 225.0 if league == "NBA" else 8.5)
from scipy.stats import norm
# approximate P(total > line)
p_over_gauss = 1 - norm.cdf(typ_line, loc=mean_total, scale=std_total if std_total > 0 else 10)
st.write(f"Using mean={mean_total:.1f}, std={std_total:.1f} → for a line of **{typ_line:.1f}**, Gaussian P(Over) ≈ **{p_over_gauss*100:.1f}%**")

# Raw table
if show_raw and not df.empty:
    st.subheader("Game Data (last 5 seasons)")
    display_cols = [c for c in ["season", "date", "week", "away", "home", "away_pts", "home_pts",
                                "actual_total", "ou_line", "over_hit", "league"] if c in df.columns]
    st.dataframe(
        df[display_cols].sort_values(["season", "date"], ascending=[False, False]).head(500),
        width='stretch',
        height=400
    )

st.markdown("---")
st.caption(f"App generated {datetime.now().strftime('%Y-%m-%d %H:%M')} | Data refreshed on load (cached). NFL source: habitatring.com / nflverse. For production betting use official paid feeds + more features (weather, injuries, pace, etc.).")
st.caption("This is an educational / research dashboard. Not financial or gambling advice.")
