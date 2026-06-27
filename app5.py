"""
Running Leaderboard — GPX Edition (v2 — Ranked + Animated + AI Coach)
"""

import json
import math
import os
import struct
import wave
from datetime import datetime
from io import BytesIO
import base64

import folium
import gpxpy
import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.cluster import DBSCAN
from streamlit_folium import st_folium

# ──────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Running Leaderboard", page_icon="🏃", layout="wide")

DATA_DIR = "data"
RUNS_FILE = os.path.join(DATA_DIR, "runs.json")
AREA_CACHE_FILE = os.path.join(DATA_DIR, "area_names.json")
PROFILES_FILE = os.path.join(DATA_DIR, "profiles.json")

EARTH_RADIUS_KM = 6371.0088
CLUSTER_RADIUS_KM = 0.2
NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
NOMINATIM_HEADERS = {"User-Agent": "running-leaderboard-app/1.0"}

GROQ_KEY = st.secrets["GROQ_API_KEY"] if "GROQ_API_KEY" in st.secrets else ""
N8N_WEBHOOK = st.secrets["N8N_WEBHOOK_URL"] if "N8N_WEBHOOK_URL" in st.secrets else ""

os.makedirs(DATA_DIR, exist_ok=True)

RANK_TIERS = [
    ("👑 Conqueror", None, None, "#FFD700", "#3d2b00"),
    ("💎 Diamond",   0.75, 1.0,  "#b9f2ff", "#003d4d"),
    ("🟣 Platinum",  0.50, 0.75, "#e0d7ff", "#2d004d"),
    ("🟡 Gold",      0.25, 0.50, "#fff3b0", "#4d3a00"),
    ("⚪ Silver",    0.10, 0.25, "#e8e8e8", "#333333"),
    ("🟤 Bronze",    0.0,  0.10, "#f5d9c0", "#4d2200"),
]

MOTIVATIONAL_LINES = [
    "Your rival's flame burns brighter only because you let it dim.",
    "Every step they take ahead of you is just the path you haven't run yet.",
    "The crown was yours once. It remembers your hands.",
    "A true runner doesn't lose a track — they temporarily lend it.",
    "Rise. The leaderboard hasn't seen your final form.",
    "Defeat is just the universe asking how badly you want it back.",
    "Their lead is the distance between who you are and who you will become.",
]

# ──────────────────────────────────────────────────────────────────────────
# Storage helpers
# ──────────────────────────────────────────────────────────────────────────
def load_json(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def load_runs():    return load_json(RUNS_FILE, [])
def save_runs(r):   save_json(RUNS_FILE, r)
def load_area_cache(): return load_json(AREA_CACHE_FILE, {})
def save_area_cache(c): save_json(AREA_CACHE_FILE, c)
def load_profiles(): return load_json(PROFILES_FILE, {})
def save_profiles(p): save_json(PROFILES_FILE, p)

# ──────────────────────────────────────────────────────────────────────────
# GPX parsing
# ──────────────────────────────────────────────────────────────────────────
def parse_gpx(file_bytes, athlete_name, file_name):
    gpx = gpxpy.parse(BytesIO(file_bytes))
    points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for p in segment.points:
                points.append((p.latitude, p.longitude))
    if not points:
        return None

    distance_km = (gpx.length_2d() or 0) / 1000.0
    duration_sec = gpx.get_duration()
    duration_min = (duration_sec / 60.0) if duration_sec else None
    pace = (duration_min / distance_km) if (duration_min and distance_km > 0) else None

    lats = [p[0] for p in points]
    lons = [p[1] for p in points]
    centroid_lat = sum(lats) / len(lats)
    centroid_lon = sum(lons) / len(lons)

    start_time = None
    try:
        start_time = gpx.tracks[0].segments[0].points[0].time
    except Exception:
        pass

    return {
        "athlete": athlete_name,
        "file_name": file_name,
        "uploaded_at": datetime.utcnow().isoformat(),
        "run_date": start_time.isoformat() if start_time else None,
        "distance_km": round(distance_km, 2),
        "duration_min": round(duration_min, 1) if duration_min else None,
        "pace_min_per_km": round(pace, 2) if pace else None,
        "centroid_lat": centroid_lat,
        "centroid_lon": centroid_lon,
        "points": points[:: max(1, len(points) // 300)],
    }

# ──────────────────────────────────────────────────────────────────────────
# Area clustering + naming
# ──────────────────────────────────────────────────────────────────────────
def assign_area_ids(runs):
    if not runs:
        return []
    coords = np.radians([[r["centroid_lat"], r["centroid_lon"]] for r in runs])
    eps = CLUSTER_RADIUS_KM / EARTH_RADIUS_KM
    labels = DBSCAN(eps=eps, min_samples=1, metric="haversine").fit(coords).labels_
    return [f"area_{lbl}" for lbl in labels]

def get_area_name(lat, lon, cache):
    key = f"{round(lat, 3)},{round(lon, 3)}"
    if key in cache:
        return cache[key]
    name = f"Unnamed Track ({lat:.3f}, {lon:.3f})"
    try:
        resp = requests.get(
            NOMINATIM_URL,
            params={"lat": lat, "lon": lon, "format": "json", "zoom": 17},
            headers=NOMINATIM_HEADERS, timeout=5,
        )
        if resp.ok:
            data = resp.json()
            addr = data.get("address", {})
            name = (addr.get("leisure") or addr.get("park") or addr.get("road")
                    or addr.get("suburb") or addr.get("neighbourhood")
                    or data.get("display_name", name).split(",")[0])
    except Exception:
        pass
    cache[key] = name
    return name

def build_dataframe(runs):
    df = pd.DataFrame(runs)
    if df.empty:
        return df
    area_ids = assign_area_ids(runs)
    df["area_id"] = area_ids

    cache = load_area_cache()
    area_names = {}
    for area_id in df["area_id"].unique():
        sub = df[df["area_id"] == area_id]
        clat, clon = sub["centroid_lat"].mean(), sub["centroid_lon"].mean()
        area_names[area_id] = get_area_name(clat, clon, cache)
    save_area_cache(cache)
    df["area_name"] = df["area_id"].map(area_names)
    return df

# ──────────────────────────────────────────────────────────────────────────
# Rank system
# ──────────────────────────────────────────────────────────────────────────
def get_rank(athlete_km, leader_km, is_leader):
    if is_leader:
        return RANK_TIERS[0]  # Conqueror
    if leader_km <= 0:
        return RANK_TIERS[5]  # Bronze
    pct = athlete_km / leader_km
    for tier in RANK_TIERS[1:]:
        _, lo, hi, bg, fg = tier
        if lo <= pct < hi:
            return tier
    return RANK_TIERS[5]

def snapshot_leaders(df):
    """Return {area_name: leader_athlete} for overtake comparison."""
    if df.empty:
        return {}
    result = {}
    for area_name, group in df.groupby("area_name"):
        lb = group.groupby("athlete")["distance_km"].sum().sort_values(ascending=False)
        result[area_name] = lb.index[0] if len(lb) else None
    return result

# ──────────────────────────────────────────────────────────────────────────
# Sound — two-note chime generated in pure Python
# ──────────────────────────────────────────────────────────────────────────
def _sine_wave(freq, duration_s, sample_rate=22050, amplitude=0.3):
    n = int(sample_rate * duration_s)
    t = np.linspace(0, duration_s, n, endpoint=False)
    wave_data = amplitude * np.sin(2 * math.pi * freq * t)
    fade = int(sample_rate * 0.05)
    wave_data[:fade] *= np.linspace(0, 1, fade)
    wave_data[-fade:] *= np.linspace(1, 0, fade)
    return (wave_data * 32767).astype(np.int16)

def make_chime_b64():
    sr = 22050
    note1 = _sine_wave(523, 0.18, sr)  # C5
    note2 = _sine_wave(659, 0.28, sr)  # E5
    silence = np.zeros(int(sr * 0.04), dtype=np.int16)
    combined = np.concatenate([note1, silence, note2])
    buf = BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(combined.tobytes())
    return base64.b64encode(buf.getvalue()).decode()

# ──────────────────────────────────────────────────────────────────────────
# Animations
# ──────────────────────────────────────────────────────────────────────────
def play_chime():
    b64 = make_chime_b64()
    st.markdown(
        f'<audio autoplay><source src="data:audio/wav;base64,{b64}" type="audio/wav"></audio>',
        unsafe_allow_html=True,
    )

def show_overtake_banner(athlete, track, new_rank_label):
    st.markdown(
        f"""
        <div style="
            animation: slideIn 0.5s ease-out;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            border: 2px solid #e94560;
            border-radius: 12px;
            padding: 1.2rem 1.8rem;
            margin: 0.8rem 0;
            text-align: center;
        ">
            <div style="font-size:2rem;">⚡</div>
            <div style="color:#e94560; font-weight:800; font-size:1.3rem; letter-spacing:2px;">RANK UP!</div>
            <div style="color:#f5f5f5; font-size:1rem; margin-top:0.4rem;">
                <b>{athlete}</b> advanced on <b>{track}</b>
            </div>
            <div style="color:#ffd700; font-size:0.9rem; margin-top:0.3rem;">{new_rank_label}</div>
        </div>
        <style>
        @keyframes slideIn {{
            from {{ opacity:0; transform: translateY(-20px); }}
            to   {{ opacity:1; transform: translateY(0);     }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def show_conqueror_banner(athlete, track):
    st.markdown(
        f"""
        <div style="
            position:relative; overflow:hidden;
            background: linear-gradient(135deg, #1a0a00, #3d1a00, #1a0a00);
            border: 3px solid #FFD700;
            border-radius: 16px;
            padding: 2rem;
            margin: 1rem 0;
            text-align: center;
        ">
            <!-- Fire particles -->
            <div class="fire-wrap">
                {"".join(f'<div class="flame" style="left:{p}%; animation-delay:{d:.1f}s; animation-duration:{dur:.1f}s;"></div>'
                         for p, d, dur in [(10,0,1.2),(20,0.3,0.9),(35,0.1,1.4),(50,0.5,1.1),(65,0.2,1.3),(80,0.4,0.8),(90,0.1,1.2)])}
            </div>
            <div style="font-size:3rem; position:relative; z-index:2;">👑</div>
            <div style="
                color: #FFD700; font-weight:900; font-size:1.8rem;
                letter-spacing:4px; position:relative; z-index:2;
                text-shadow: 0 0 20px #FFD700, 0 0 40px #ff8c00;
            ">CONQUEROR</div>
            <div style="color:#fff; font-size:1.1rem; margin-top:0.5rem; position:relative; z-index:2;">
                <b>{athlete}</b> now rules <b>{track}</b>
            </div>
            <div style="color:#ffa500; font-size:0.85rem; font-style:italic; margin-top:0.5rem; position:relative; z-index:2;">
                "{np.random.choice(MOTIVATIONAL_LINES)}"
            </div>
        </div>
        <style>
        .fire-wrap {{ position:absolute; bottom:0; left:0; width:100%; height:60px; }}
        .flame {{
            position:absolute; bottom:0; width:18px; height:40px;
            background: linear-gradient(to top, #ff4500, #ff8c00, #FFD700, transparent);
            border-radius: 50% 50% 20% 20%;
            animation: flicker linear infinite;
            transform-origin: bottom center;
            filter: blur(1px);
        }}
        @keyframes flicker {{
            0%   {{ transform: scaleX(1)   scaleY(1)   rotate(-3deg); opacity:0.9; }}
            25%  {{ transform: scaleX(0.8) scaleY(1.2) rotate(3deg);  opacity:1;   }}
            50%  {{ transform: scaleX(1.1) scaleY(0.9) rotate(-2deg); opacity:0.8; }}
            75%  {{ transform: scaleX(0.9) scaleY(1.1) rotate(2deg);  opacity:1;   }}
            100% {{ transform: scaleX(1)   scaleY(1)   rotate(-3deg); opacity:0.9; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────────────────────
# n8n webhook
# ──────────────────────────────────────────────────────────────────────────
def fire_overtake_email(to_email, to_name, track_name, new_leader):
    if not N8N_WEBHOOK:
        return
    line = np.random.choice(MOTIVATIONAL_LINES)
    try:
        requests.post(N8N_WEBHOOK, json={
            "to_email": to_email,
            "to_name": to_name,
            "track_name": track_name,
            "new_leader": new_leader,
            "motivation_line": line,
        }, timeout=5)
    except Exception:
        pass  # Never crash the app over a notification

# ──────────────────────────────────────────────────────────────────────────
# Gemini AI Coach
# ──────────────────────────────────────────────────────────────────────────
def build_athlete_summary(athlete_df):
    """Compute stats from the athlete's runs — Gemini will turn these into coaching."""
    runs = athlete_df.sort_values("run_date") if "run_date" in athlete_df else athlete_df
    paces = runs["pace_min_per_km"].dropna().tolist()
    distances = runs["distance_km"].tolist()

    pace_trend = "improving" if len(paces) >= 2 and paces[-1] < paces[0] else (
        "getting slower" if len(paces) >= 2 and paces[-1] > paces[0] else "steady")
    avg_pace = round(sum(paces) / len(paces), 2) if paces else None
    best_pace = round(min(paces), 2) if paces else None
    total_runs = len(distances)
    avg_dist = round(sum(distances) / len(distances), 2) if distances else None
    total_km = round(sum(distances), 1)

    return {
        "total_runs": total_runs,
        "total_km": total_km,
        "avg_distance_km": avg_dist,
        "avg_pace_min_per_km": avg_pace,
        "best_pace_min_per_km": best_pace,
        "pace_trend": pace_trend,
    }

def call_groq(prompt):
    if not GROQ_KEY:
        st.error("DEBUG: GROQ_KEY is empty — key not found in secrets")
        return None
    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
            json={
                "model": "llama3-8b-8192",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 300,
            },
            timeout=15,
        )
        if resp.ok:
            return resp.json()["choices"][0]["message"]["content"]
        else:
            st.error(f"DEBUG: Groq status {resp.status_code} — {resp.text[:300]}")
    except Exception as e:
        st.error(f"DEBUG: Exception — {e}")
    return None

def render_ai_coach(athlete, athlete_df):
    cache_key = f"coach_{athlete}"
    with st.popover("🤖 AI Coach"):
        if cache_key not in st.session_state:
            if not GROQ_KEY:
                st.info("Add GROQ_API_KEY to Streamlit secrets to enable AI coaching.")
                return
            if st.button("✨ Generate coaching advice", key=f"gen_{athlete}"):
                summary = build_athlete_summary(athlete_df)
                prompt = (
                    f"You are a friendly running coach. Here are the stats for runner '{athlete}':\n"
                    f"{json.dumps(summary, indent=2)}\n\n"
                    "In 3-4 short bullet points, tell them what they're doing well and give one specific "
                    "tip to improve. Be encouraging but honest. Use plain language, no jargon."
                )
                with st.spinner("Thinking..."):
                    result = call_groq(prompt)
                if result:
                    st.session_state[cache_key] = result
                    st.rerun()
                else:
                    st.error("Couldn't reach Groq — check your GROQ_API_KEY in Streamlit secrets.")
        else:
            st.markdown(st.session_state[cache_key])
            if st.button("🔄 Regenerate", key=f"regen_{athlete}"):
                del st.session_state[cache_key]
                st.rerun()
# ──────────────────────────────────────────────────────────────────────────
# Sidebar — upload form
# ──────────────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Add Your Run")
athlete_name = st.sidebar.text_input("Your name", placeholder="e.g. Bilal Ahmed")
athlete_email = st.sidebar.text_input("Your email (for notifications)", placeholder="you@example.com")
uploaded_files = st.sidebar.file_uploader("Upload GPX file(s)", type=["gpx"], accept_multiple_files=True)

if st.sidebar.button("Add to Leaderboard", type="primary"):
    if not athlete_name.strip():
        st.sidebar.error("Please enter your name first.")
    elif not uploaded_files:
        st.sidebar.error("Please upload at least one .gpx file.")
    else:
        runs = load_runs()

        # --- Snapshot leaders BEFORE adding new runs ---
        df_before = build_dataframe(runs) if runs else pd.DataFrame()
        leaders_before = snapshot_leaders(df_before)

        # --- Save email profile ---
        if athlete_email.strip():
            profiles = load_profiles()
            profiles[athlete_name.strip()] = athlete_email.strip()
            save_profiles(profiles)

        added = 0
        for f in uploaded_files:
            record = parse_gpx(f.read(), athlete_name.strip(), f.name)
            if record:
                runs.append(record)
                added += 1
        save_runs(runs)

        # --- Snapshot leaders AFTER ---
        df_after = build_dataframe(runs)
        leaders_after = snapshot_leaders(df_after)

        # --- Detect changes ---
        overtakes = []   # tracks where someone NEW took #1
        rank_ups = []    # tracks where the uploader improved position
        conquerors = []  # tracks where uploader became new #1

        for area_name, new_leader in leaders_after.items():
            old_leader = leaders_before.get(area_name)
            if new_leader == athlete_name.strip() and old_leader != athlete_name.strip():
                conquerors.append(area_name)
                # Email the person who LOST their crown
                if old_leader:
                    profiles = load_profiles()
                    old_email = profiles.get(old_leader, "")
                    if old_email:
                        fire_overtake_email(old_email, old_leader, area_name, athlete_name.strip())
            elif new_leader != athlete_name.strip():
                # Check if uploader moved up (not #1 but improved)
                area_df_after = df_after[df_after["area_name"] == area_name]
                lb_after = area_df_after.groupby("athlete")["distance_km"].sum().sort_values(ascending=False)
                if athlete_name.strip() in lb_after.index:
                    pos_after = list(lb_after.index).index(athlete_name.strip()) + 1
                    if area_name in leaders_before:
                        area_df_before = df_before[df_before["area_name"] == area_name] if not df_before.empty else pd.DataFrame()
                        if not area_df_before.empty:
                            lb_before = area_df_before.groupby("athlete")["distance_km"].sum().sort_values(ascending=False)
                            if athlete_name.strip() in lb_before.index:
                                pos_before = list(lb_before.index).index(athlete_name.strip()) + 1
                                if pos_after < pos_before:
                                    rank_ups.append((area_name, pos_after))

        # Store celebration info in session state for rendering after rerun
        if added > 0:
            st.session_state["celebrate"] = True
        if conquerors:
            st.session_state["conquerors"] = conquerors
            st.session_state["conqueror_athlete"] = athlete_name.strip()
        if rank_ups:
            st.session_state["rank_ups"] = [(t, p, athlete_name.strip()) for t, p in rank_ups]

        st.sidebar.success(f"Added {added} run(s)! 🎉")
        st.rerun()

st.sidebar.divider()
st.sidebar.caption(
    "💡 **How to get a GPX file:**\n\n"
    "- **Strava**: activity → ⋯ → Export GPX\n"
    "- **Garmin Connect**: activity → gear → Export to GPX\n"
    "- **Any GPS app**: look for 'Export' or 'Share as GPX'"
)
if st.sidebar.button("🗑️ Reset all data"):
    save_runs([])
    save_area_cache({})
    save_profiles({})
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

# ──────────────────────────────────────────────────────────────────────────
# Celebrations (fire after rerun so they're visible)
# ──────────────────────────────────────────────────────────────────────────
if st.session_state.pop("celebrate", False):
    st.balloons()
    play_chime()

conquerors_now = st.session_state.pop("conquerors", [])
conqueror_athlete = st.session_state.pop("conqueror_athlete", "")
for track in conquerors_now:
    show_conqueror_banner(conqueror_athlete, track)

rank_ups_now = st.session_state.pop("rank_ups", [])
for track, pos, ath in rank_ups_now:
    tier_label = f"#{pos} on this track"
    show_overtake_banner(ath, track, tier_label)

# ──────────────────────────────────────────────────────────────────────────
# Main content
# ──────────────────────────────────────────────────────────────────────────
st.title("🏃 Running Leaderboard")
st.caption("Powered by free GPX uploads + OpenStreetMap — no Strava subscription required.")

runs = load_runs()
if not runs:
    st.info("No runs yet! Upload a GPX file from the sidebar to get started. 👈")
    st.stop()

df = build_dataframe(runs)

col1, col2, col3 = st.columns(3)
col1.metric("🏃 Total Runs", len(df))
col2.metric("📏 Total KM", f"{df['distance_km'].sum():,.1f} km")
col3.metric("📍 Tracks Found", df["area_id"].nunique())

st.divider()

# ──────────────────────────────────────────────────────────────────────────
# Per-area leaderboard with rank badges + progress bars + AI Coach
# ──────────────────────────────────────────────────────────────────────────
st.header("🏆 Leaderboard by Track")

area_totals = (
    df.groupby(["area_id", "area_name"])["distance_km"]
    .sum().reset_index()
    .sort_values("distance_km", ascending=False)
)

for _, area_row in area_totals.iterrows():
    area_id = area_row["area_id"]
    area_name = area_row["area_name"]
    area_df = df[df["area_id"] == area_id]

    leaderboard = (
        area_df.groupby("athlete")
        .agg(total_km=("distance_km", "sum"), runs=("distance_km", "count"),
             best_pace=("pace_min_per_km", "min"))
        .reset_index()
        .sort_values("total_km", ascending=False)
    )
    owner = leaderboard.iloc[0]["athlete"]
    leader_km = leaderboard.iloc[0]["total_km"]

    with st.expander(f"📍 **{area_name}** — 👑 {owner}", expanded=True):
        medals = ["🥇", "🥈", "🥉"]

        for rank_pos, (_, row) in enumerate(leaderboard.iterrows(), start=1):
            athlete = row["athlete"]
            athlete_km = row["total_km"]
            is_leader = rank_pos == 1

            tier = get_rank(athlete_km, leader_km, is_leader)
            tier_label, *_, badge_bg, badge_fg = tier

            medal = medals[rank_pos - 1] if rank_pos <= 3 else f"#{rank_pos}"

            # --- Row layout ---
            c1, c2, c3, c4 = st.columns([0.5, 3, 1.5, 1.5])
            c1.markdown(f"<div style='font-size:1.4rem;padding-top:4px'>{medal}</div>", unsafe_allow_html=True)

            # Rank badge + name
            c2.markdown(
                f"""<div style='display:flex; align-items:center; gap:8px; padding-top:4px;'>
                    <span style='background:{badge_bg}; color:{badge_fg}; border-radius:6px;
                                 padding:2px 8px; font-size:0.75rem; font-weight:700;'>{tier_label}</span>
                    <span style='font-weight:600'>{athlete}</span>
                </div>""",
                unsafe_allow_html=True,
            )
            c3.markdown(f"<div style='padding-top:6px'>{athlete_km:.1f} km &nbsp;·&nbsp; {int(row['runs'])} runs</div>", unsafe_allow_html=True)

            # AI Coach popover
            with c4:
                athlete_runs_df = area_df[area_df["athlete"] == athlete]
                render_ai_coach(athlete, athlete_runs_df)

            # Progress bar (gap to leader) — skip for leader
            if not is_leader and leader_km > 0:
                pct = min(athlete_km / leader_km, 1.0)
                gap = round(leader_km - athlete_km, 1)
                st.progress(pct, text=f"  {gap} km behind the leader")

            st.markdown("<hr style='margin:4px 0; border-color:#333'>", unsafe_allow_html=True)

st.divider()

# ──────────────────────────────────────────────────────────────────────────
# Map
# ──────────────────────────────────────────────────────────────────────────
st.header("🗺️ Map")

avg_lat, avg_lon = df["centroid_lat"].mean(), df["centroid_lon"].mean()
m = folium.Map(location=[avg_lat, avg_lon], zoom_start=13, tiles="OpenStreetMap")

colors = ["red","blue","green","purple","orange","darkred","cadetblue","darkgreen","darkblue","pink"]
area_color_map = {aid: colors[i % len(colors)] for i, aid in enumerate(df["area_id"].unique())}

for _, run in df.iterrows():
    pts = run["points"]
    if len(pts) >= 2:
        folium.PolyLine(
            pts, color=area_color_map[run["area_id"]],
            weight=3, opacity=0.6,
            tooltip=f"{run['athlete']} — {run['distance_km']} km",
        ).add_to(m)

area_summaries = []
for area_id, area_name in df.groupby("area_id")["area_name"].first().items():
    sub = df[df["area_id"] == area_id]
    owner_row = sub.groupby("athlete")["distance_km"].sum().sort_values(ascending=False)
    area_summaries.append({
        "area_id": area_id, "area_name": area_name,
        "lat": sub["centroid_lat"].mean(), "lon": sub["centroid_lon"].mean(),
        "owner": owner_row.index[0], "owner_km": owner_row.iloc[0],
    })
area_summaries.sort(key=lambda a: a["owner_km"])

MIN_R, MAX_R = 10, 32
max_km = max(a["owner_km"] for a in area_summaries) or 1

for a in area_summaries:
    scale = math.sqrt(a["owner_km"] / max_km)
    radius = MIN_R + scale * (MAX_R - MIN_R)
    folium.CircleMarker(
        location=[a["lat"], a["lon"]], radius=radius,
        color=area_color_map[a["area_id"]], fill=True,
        fill_color=area_color_map[a["area_id"]], fill_opacity=0.75, weight=2,
        popup=f"<b>{a['area_name']}</b><br>👑 {a['owner']} — {a['owner_km']:.1f} km",
        tooltip=f"{a['area_name']} — 👑 {a['owner']} ({a['owner_km']:.1f} km)",
    ).add_to(m)

st_folium(m, width=None, height=500)

st.divider()
with st.expander("📋 All uploaded runs"):
    st.dataframe(
        df[["athlete","file_name","area_name","distance_km","duration_min","pace_min_per_km","run_date"]]
        .rename(columns={"athlete":"Athlete","file_name":"File","area_name":"Track",
                         "distance_km":"KM","duration_min":"Duration (min)",
                         "pace_min_per_km":"Pace (min/km)","run_date":"Date"}),
        hide_index=True, use_container_width=True,
    )
