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
st.set_page_config(page_title="Runova", page_icon="⚡", layout="wide")

# NOTE: Replace the background-image URL below with your actual dragon background
# (either a raw GitHub URL or keep as base64 by pasting your original base64 string)
st.markdown("""
<style>
.stApp {
    background-color: rgba(14, 17, 23, 1);
}
</style>
""", unsafe_allow_html=True)

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

def load_runs():         return load_json(RUNS_FILE, [])
def save_runs(r):        save_json(RUNS_FILE, r)
def load_area_cache():   return load_json(AREA_CACHE_FILE, {})
def save_area_cache(c):  save_json(AREA_CACHE_FILE, c)
def load_profiles():     return load_json(PROFILES_FILE, {})
def save_profiles(p):    save_json(PROFILES_FILE, p)

# ──────────────────────────────────────────────────────────────────────────
# Avatar helper — photo if available, else coloured initials circle
# ──────────────────────────────────────────────────────────────────────────
def avatar_html(athlete, profiles, size=44, border_color="#c8a84b"):
    profile = profiles.get(athlete, {})
    if isinstance(profile, str):
        profile = {"email": profile, "photo_b64": "", "trash_talk": ""}
    photo_b64 = profile.get("photo_b64", "")
    initials = "".join([w[0].upper() for w in athlete.split()[:2]])

    if photo_b64:
        return (
            f'<img src="data:image/jpeg;base64,{photo_b64}" '
            f'style="width:{size}px; height:{size}px; border-radius:50%; '
            f'object-fit:cover; border:2px solid {border_color}; flex-shrink:0;" />'
        )
    return (
        f'<div style="width:{size}px; height:{size}px; border-radius:50%; '
        f'background:linear-gradient(135deg,{border_color},#333); '
        f'display:flex; align-items:center; justify-content:center; '
        f'font-size:{int(size*0.38)}px; font-weight:700; color:#fff; '
        f'flex-shrink:0; border:2px solid {border_color};">{initials}</div>'
    )

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
            params={"lat": lat, "lon": lon, "format": "json", "zoom": 16},
            headers=NOMINATIM_HEADERS, timeout=5,
        )
        if resp.ok:
            data = resp.json()
            addr = data.get("address", {})
            name = (
                addr.get("leisure") or addr.get("amenity") or addr.get("tourism")
                or addr.get("park") or addr.get("natural") or addr.get("road")
                or addr.get("pedestrian") or addr.get("path")
                or addr.get("neighbourhood") or addr.get("suburb")
                or addr.get("village") or addr.get("town")
                or addr.get("city_district") or addr.get("county")
                or addr.get("city")
                or data.get("display_name", name).split(",")[0]
            )
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
        return RANK_TIERS[0]
    if leader_km <= 0:
        return RANK_TIERS[5]
    pct = athlete_km / leader_km
    for tier in RANK_TIERS[1:]:
        _, lo, hi, bg, fg = tier
        if lo <= pct < hi:
            return tier
    return RANK_TIERS[5]

def snapshot_leaders(df):
    if df.empty:
        return {}
    result = {}
    for area_name, group in df.groupby("area_name"):
        lb = group.groupby("athlete")["distance_km"].sum().sort_values(ascending=False)
        result[area_name] = lb.index[0] if len(lb) else None
    return result

# ──────────────────────────────────────────────────────────────────────────
# Sound — two-note chime
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
    note1 = _sine_wave(523, 0.18, sr)
    note2 = _sine_wave(659, 0.28, sr)
    silence = np.zeros(int(sr * 0.04), dtype=np.int16)
    combined = np.concatenate([note1, silence, note2])
    buf = BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
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
    st.markdown(f"""
    <div style="animation:slideIn 0.5s ease-out; background:linear-gradient(135deg,#1a1a2e,#16213e,#0f3460);
                border:2px solid #e94560; border-radius:12px; padding:1.2rem 1.8rem;
                margin:0.8rem 0; text-align:center;">
        <div style="font-size:2rem;">⚡</div>
        <div style="color:#e94560; font-weight:800; font-size:1.3rem; letter-spacing:2px;">RANK UP!</div>
        <div style="color:#f5f5f5; font-size:1rem; margin-top:0.4rem;">
            <b>{athlete}</b> advanced on <b>{track}</b>
        </div>
        <div style="color:#ffd700; font-size:0.9rem; margin-top:0.3rem;">{new_rank_label}</div>
    </div>
    <style>
    @keyframes slideIn {{ from{{opacity:0;transform:translateY(-20px)}} to{{opacity:1;transform:translateY(0)}} }}
    </style>""", unsafe_allow_html=True)

def show_conqueror_banner(athlete, track):
    flames = "".join(
        f'<div class="flame" style="left:{p}%;animation-delay:{d:.1f}s;animation-duration:{dur:.1f}s;"></div>'
        for p, d, dur in [(10,0,1.2),(20,0.3,0.9),(35,0.1,1.4),(50,0.5,1.1),(65,0.2,1.3),(80,0.4,0.8),(90,0.1,1.2)]
    )
    line = np.random.choice(MOTIVATIONAL_LINES)
    st.markdown(f"""
    <div style="position:relative;overflow:hidden;background:linear-gradient(135deg,#1a0a00,#3d1a00,#1a0a00);
                border:3px solid #FFD700; border-radius:16px; padding:2rem; margin:1rem 0; text-align:center;">
        <div class="fire-wrap">{flames}</div>
        <div style="font-size:3rem;position:relative;z-index:2;">👑</div>
        <div style="color:#FFD700;font-weight:900;font-size:1.8rem;letter-spacing:4px;
                    position:relative;z-index:2;text-shadow:0 0 20px #FFD700,0 0 40px #ff8c00;">CONQUEROR</div>
        <div style="color:#fff;font-size:1.1rem;margin-top:0.5rem;position:relative;z-index:2;">
            <b>{athlete}</b> now rules <b>{track}</b>
        </div>
        <div style="color:#ffa500;font-size:0.85rem;font-style:italic;margin-top:0.5rem;position:relative;z-index:2;">
            "{line}"
        </div>
    </div>
    <style>
    .fire-wrap{{position:absolute;bottom:0;left:0;width:100%;height:60px;}}
    .flame{{position:absolute;bottom:0;width:18px;height:40px;
            background:linear-gradient(to top,#ff4500,#ff8c00,#FFD700,transparent);
            border-radius:50% 50% 20% 20%;animation:flicker linear infinite;
            transform-origin:bottom center;filter:blur(1px);}}
    @keyframes flicker{{
        0%  {{transform:scaleX(1)   scaleY(1)   rotate(-3deg);opacity:0.9;}}
        25% {{transform:scaleX(0.8) scaleY(1.2) rotate(3deg); opacity:1;}}
        50% {{transform:scaleX(1.1) scaleY(0.9) rotate(-2deg);opacity:0.8;}}
        75% {{transform:scaleX(0.9) scaleY(1.1) rotate(2deg); opacity:1;}}
        100%{{transform:scaleX(1)   scaleY(1)   rotate(-3deg);opacity:0.9;}}
    }}
    </style>""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────
# n8n webhook
# ──────────────────────────────────────────────────────────────────────────
def fire_overtake_email(to_email, to_name, track_name, new_leader):
    if not N8N_WEBHOOK:
        return
    line = np.random.choice(MOTIVATIONAL_LINES)
    try:
        requests.post(N8N_WEBHOOK, json={
            "to_email": to_email, "to_name": to_name,
            "track_name": track_name, "new_leader": new_leader,
            "motivation_line": line,
        }, timeout=5)
    except Exception:
        pass

# ──────────────────────────────────────────────────────────────────────────
# AI Coach (Groq)
# ──────────────────────────────────────────────────────────────────────────
def build_athlete_summary(athlete_df):
    runs = athlete_df.sort_values("run_date") if "run_date" in athlete_df else athlete_df
    paces = runs["pace_min_per_km"].dropna().tolist()
    distances = runs["distance_km"].tolist()
    pace_trend = ("improving" if len(paces) >= 2 and paces[-1] < paces[0]
                  else "getting slower" if len(paces) >= 2 and paces[-1] > paces[0]
                  else "steady")
    return {
        "total_runs": len(distances),
        "total_km": round(sum(distances), 1),
        "avg_distance_km": round(sum(distances)/len(distances), 2) if distances else None,
        "avg_pace_min_per_km": round(sum(paces)/len(paces), 2) if paces else None,
        "best_pace_min_per_km": round(min(paces), 2) if paces else None,
        "pace_trend": pace_trend,
    }

def call_groq(prompt):
    if not GROQ_KEY:
        return None
    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
            json={"model": "llama-3.3-70b-versatile",
                  "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": 300},
            timeout=15,
        )
        if resp.ok:
            return resp.json()["choices"][0]["message"]["content"]
    except Exception:
        pass
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
                    st.error("Couldn't reach Groq — check your GROQ_API_KEY.")
        else:
            st.markdown(st.session_state[cache_key])
            if st.button("🔄 Regenerate", key=f"regen_{athlete}"):
                del st.session_state[cache_key]
                st.rerun()

# ──────────────────────────────────────────────────────────────────────────
# Sidebar — upload form
# ──────────────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Add Your Run")
athlete_name  = st.sidebar.text_input("Your name", placeholder="e.g. Bilal Ahmed")
athlete_email = st.sidebar.text_input("Your email (for notifications)", placeholder="you@example.com")
profile_pic   = st.sidebar.file_uploader(
    "Profile picture (optional)", type=["jpg", "jpeg", "png"], key="profile_pic_upload"
)
trash_talk    = st.sidebar.text_input(
    "Trash-talk message 🗣️", placeholder="e.g. Can't catch me 😤",
    help="Shows on the map when you own a track."
)
uploaded_files = st.sidebar.file_uploader(
    "Upload GPX file(s)", type=["gpx"], accept_multiple_files=True
)

if st.sidebar.button("Add to Leaderboard", type="primary"):
    if not athlete_name.strip():
        st.sidebar.error("Please enter your name first.")
    elif not uploaded_files:
        st.sidebar.error("Please upload at least one .gpx file.")
    else:
        runs = load_runs()

        df_before = build_dataframe(runs) if runs else pd.DataFrame()
        leaders_before = snapshot_leaders(df_before)

        # Save/update profile as nested dict
        profiles = load_profiles()
        existing = profiles.get(athlete_name.strip(), {})
        if isinstance(existing, str):
            existing = {"email": existing, "photo_b64": "", "trash_talk": ""}

        if athlete_email.strip():
            existing["email"] = athlete_email.strip()
        if trash_talk.strip():
            existing["trash_talk"] = trash_talk.strip()
        if profile_pic is not None:
            raw = profile_pic.read()
            existing["photo_b64"] = base64.b64encode(raw).decode()

        profiles[athlete_name.strip()] = existing
        save_profiles(profiles)

        added = 0
        for f in uploaded_files:
            record = parse_gpx(f.read(), athlete_name.strip(), f.name)
            if record:
                runs.append(record)
                added += 1
        save_runs(runs)

        df_after = build_dataframe(runs)
        leaders_after = snapshot_leaders(df_after)

        conquerors = []
        rank_ups   = []

        for area_name, new_leader in leaders_after.items():
            old_leader = leaders_before.get(area_name)
            if new_leader == athlete_name.strip() and old_leader != athlete_name.strip():
                conquerors.append(area_name)
                if old_leader:
                    old_profile = profiles.get(old_leader, {})
                    old_email = (old_profile.get("email", "")
                                 if isinstance(old_profile, dict) else old_profile)
                    if old_email:
                        fire_overtake_email(old_email, old_leader, area_name, athlete_name.strip())
            elif new_leader != athlete_name.strip():
                area_df_after = df_after[df_after["area_name"] == area_name]
                lb_after = area_df_after.groupby("athlete")["distance_km"].sum().sort_values(ascending=False)
                if athlete_name.strip() in lb_after.index:
                    pos_after = list(lb_after.index).index(athlete_name.strip()) + 1
                    if area_name in leaders_before and not df_before.empty:
                        area_df_before = df_before[df_before["area_name"] == area_name]
                        if not area_df_before.empty:
                            lb_before = area_df_before.groupby("athlete")["distance_km"].sum().sort_values(ascending=False)
                            if athlete_name.strip() in lb_before.index:
                                pos_before = list(lb_before.index).index(athlete_name.strip()) + 1
                                if pos_after < pos_before:
                                    rank_ups.append((area_name, pos_after))

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

if st.sidebar.button("🧪 Test n8n webhook"):
    if not N8N_WEBHOOK:
        st.sidebar.error("N8N_WEBHOOK_URL not found in secrets!")
    else:
        try:
            r = requests.post(N8N_WEBHOOK, json={
                "to_email": "test@test.com", "to_name": "TestUser",
                "track_name": "Test Track", "new_leader": "Rival",
                "motivation_line": "Test message"
            }, timeout=10)
            st.sidebar.write(f"Status: {r.status_code}")
            st.sidebar.write(f"URL used: {N8N_WEBHOOK}")
            st.sidebar.write(f"Response: {r.text[:200]}")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")

if st.sidebar.button("🗑️ Reset all data"):
    save_runs([])
    save_profiles({})
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

# ──────────────────────────────────────────────────────────────────────────
# Celebrations
# ──────────────────────────────────────────────────────────────────────────
if st.session_state.pop("celebrate", False):
    st.balloons()
    play_chime()

conquerors_now     = st.session_state.pop("conquerors", [])
conqueror_athlete  = st.session_state.pop("conqueror_athlete", "")
for track in conquerors_now:
    show_conqueror_banner(conqueror_athlete, track)

rank_ups_now = st.session_state.pop("rank_ups", [])
for track, pos, ath in rank_ups_now:
    show_overtake_banner(ath, track, f"#{pos} on this track")

# ──────────────────────────────────────────────────────────────────────────
# Main content
# ──────────────────────────────────────────────────────────────────────────
runs     = load_runs()
df       = build_dataframe(runs) if runs else pd.DataFrame()
profiles = load_profiles()   # load once, pass everywhere

total_tracks = df["area_id"].nunique() if not df.empty else 0

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;600;800&display=swap');
.hero-wrap  {{ text-align:center; padding:3rem 1rem 2.5rem; position:relative; }}
.hero-tag   {{ display:inline-block; background:linear-gradient(90deg,#c8a84b,#ffe97d,#c8a84b);
               -webkit-background-clip:text; -webkit-text-fill-color:transparent;
               font-size:0.7rem; font-family:'Inter',sans-serif; font-weight:800;
               letter-spacing:6px; text-transform:uppercase; margin-bottom:0.6rem; }}
.hero-title {{ font-family:'Bebas Neue',sans-serif; font-size:clamp(3.5rem,10vw,6rem);
               line-height:0.95; color:#fff; letter-spacing:2px; margin:0.3rem 0;
               text-shadow:0 0 60px rgba(200,168,75,0.2); }}
.hero-title span {{ background:linear-gradient(90deg,#c8a84b,#ffe97d,#ff9d00);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
.hero-sub   {{ font-family:'Inter',sans-serif; font-size:0.85rem; color:#555;
               letter-spacing:1px; margin-top:0.8rem; margin-bottom:1.8rem; }}
.hero-pill  {{ display:inline-flex; align-items:center; gap:6px;
               background:rgba(200,168,75,0.07); border:1px solid rgba(200,168,75,0.2);
               border-radius:999px; padding:0.4rem 1.2rem; font-family:'Inter',sans-serif;
               font-size:0.8rem; color:#c8a84b; font-weight:600; letter-spacing:1px; }}
.hero-divider {{ width:80px; height:2px;
                 background:linear-gradient(90deg,transparent,#c8a84b,transparent);
                 margin:1.5rem auto 0; border:none; }}
</style>
<div class="hero-wrap">
    <div class="hero-tag">⚡ Conquer · Dominate · Rule</div>
    <div class="hero-title">Run<span>ova</span></div>
    <div class="hero-sub">drop your gpx. claim your track. flex on everyone.</div>
    <div class="hero-pill">📍 {total_tracks} {'track' if total_tracks == 1 else 'tracks'} being contested</div>
    <div class="hero-divider"></div>
</div>
""", unsafe_allow_html=True)

if not runs:
    st.info("No runs yet! Upload a GPX file from the sidebar to get started. 👈")
    st.stop()

st.divider()

# ──────────────────────────────────────────────────────────────────────────
# Per-area leaderboard
# ──────────────────────────────────────────────────────────────────────────
st.header("🏆 Leaderboard by Track")

area_totals = (
    df.groupby(["area_id", "area_name"])["distance_km"]
    .sum().reset_index()
    .sort_values("distance_km", ascending=False)
)

for _, area_row in area_totals.iterrows():
    area_id   = area_row["area_id"]
    area_name = area_row["area_name"]
    area_df   = df[df["area_id"] == area_id]

    leaderboard = (
        area_df.groupby("athlete")
        .agg(total_km=("distance_km","sum"), runs=("distance_km","count"),
             best_pace=("pace_min_per_km","min"))
        .reset_index()
        .sort_values("total_km", ascending=False)
    ).reset_index(drop=True)

    owner     = leaderboard.iloc[0]["athlete"]
    leader_km = leaderboard.iloc[0]["total_km"]

    with st.expander(f"📍 {area_name}  —  👑 {owner}", expanded=True):

        # ── TOP 3 PODIUM ──
        top3 = leaderboard.head(3)
        n    = len(top3)
        order = [1, 0, 2] if n >= 3 else ([1, 0] if n == 2 else [0])
        cols  = st.columns(3)

        podium_styles = {
            0: {"border":"#FFD700","bg":"rgba(255,215,0,0.08)","size":"1.8rem","crown":"👑","height":"200px"},
            1: {"border":"#C0C0C0","bg":"rgba(192,192,192,0.06)","size":"1.4rem","crown":"🥈","height":"170px"},
            2: {"border":"#CD7F32","bg":"rgba(205,127,50,0.06)","size":"1.2rem","crown":"🥉","height":"160px"},
        }

        for col_idx, rank_idx in enumerate(order[:n]):
            row        = top3.iloc[rank_idx]
            athlete    = row["athlete"]
            athlete_km = row["total_km"]
            runs_count = int(row["runs"])
            is_leader  = rank_idx == 0
            tier       = get_rank(athlete_km, leader_km, is_leader)
            tier_label, *_, badge_bg, badge_fg = tier
            s  = podium_styles[rank_idx]
            av = avatar_html(athlete, profiles, size=52, border_color=s["border"])

            with cols[col_idx]:
                st.markdown(f"""
                <div style="background:{s['bg']};border:1.5px solid {s['border']};border-radius:16px;
                            padding:1.2rem 1rem;text-align:center;min-height:{s['height']};
                            display:flex;flex-direction:column;align-items:center;
                            justify-content:center;margin-bottom:0.5rem;">
                    <div style="font-size:1.6rem;margin-bottom:4px;">{s['crown']}</div>
                    <div style="margin-bottom:8px;">{av}</div>
                    <div style="font-size:{s['size']};font-weight:700;color:#fff;margin-bottom:4px;">{athlete}</div>
                    <div style="font-size:1.1rem;font-weight:600;color:{s['border']};">{athlete_km:.1f} km</div>
                    <div style="font-size:0.75rem;color:#888;margin-top:2px;">{runs_count} runs</div>
                    <div style="margin-top:6px;">
                        <span style="background:{badge_bg};color:{badge_fg};border-radius:6px;
                                     padding:2px 8px;font-size:0.7rem;font-weight:700;">{tier_label}</span>
                    </div>
                </div>""", unsafe_allow_html=True)
                render_ai_coach(athlete, area_df[area_df["athlete"] == athlete])

        # ── 4th+ ROWS ──
        if len(leaderboard) > 3:
            st.markdown("<div style='margin-top:1rem;'>", unsafe_allow_html=True)
            for rank_pos, (_, row) in enumerate(leaderboard.iloc[3:].iterrows(), start=4):
                athlete    = row["athlete"]
                athlete_km = row["total_km"]
                runs_count = int(row["runs"])
                tier       = get_rank(athlete_km, leader_km, False)
                tier_label, *_, badge_bg, badge_fg = tier
                pct = min(athlete_km / leader_km, 1.0) if leader_km > 0 else 0
                gap = round(leader_km - athlete_km, 1)
                av  = avatar_html(athlete, profiles, size=36, border_color="#555")

                c1, c2, c3 = st.columns([0.5, 4, 2])
                c1.markdown(f"<div style='font-size:1.1rem;padding-top:8px;color:#888;'>#{rank_pos}</div>",
                            unsafe_allow_html=True)
                c2.markdown(f"""
                <div style='display:flex;align-items:center;gap:10px;padding:6px 0;'>
                    {av}
                    <div>
                        <div style='font-weight:600;color:#fff;'>{athlete}</div>
                        <span style='background:{badge_bg};color:{badge_fg};border-radius:4px;
                                     padding:1px 6px;font-size:0.68rem;font-weight:700;'>{tier_label}</span>
                    </div>
                </div>""", unsafe_allow_html=True)
                c3.markdown(
                    f"<div style='text-align:right;padding-top:8px;color:#c8a84b;font-weight:600;'>"
                    f"{athlete_km:.1f} km <span style='color:#555;font-size:0.8rem;'>· {runs_count} runs</span></div>",
                    unsafe_allow_html=True)
                st.progress(pct, text=f"{gap} km behind leader")
                st.markdown("<hr style='margin:4px 0;border-color:#222;'>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

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

from folium.plugins import HeatMap
from folium import FeatureGroup, LayerControl

# Build area summary list
area_summaries = []
for area_id, area_name in df.groupby("area_id")["area_name"].first().items():
    sub       = df[df["area_id"] == area_id]
    owner_row = sub.groupby("athlete")["distance_km"].sum().sort_values(ascending=False)
    lb        = owner_row.reset_index().head(3)
    area_summaries.append({
        "area_id": area_id, "area_name": area_name,
        "lat": sub["centroid_lat"].mean(), "lon": sub["centroid_lon"].mean(),
        "owner": owner_row.index[0], "owner_km": owner_row.iloc[0],
        "leaderboard": lb,
    })
area_summaries.sort(key=lambda a: a["owner_km"])

# ── Ownership layer ──
ownership_layer = FeatureGroup(name="👑 Ownership", show=True)

OFFSET_DEG = 0.003
placed = []

def get_offset(lat, lon):
    nudge = 0
    for (plat, plon) in placed:
        dist = ((lat - plat)**2 + (lon - plon)**2) ** 0.5
        if dist < OFFSET_DEG * 2:
            nudge += OFFSET_DEG
    placed.append((lat + nudge, lon))
    return nudge

for a in area_summaries:
    nudge       = get_offset(a["lat"], a["lon"])
    display_lat = a["lat"] + nudge
    owner       = a["owner"]

    # Pull owner profile
    owner_profile  = profiles.get(owner, {})
    if isinstance(owner_profile, str):
        owner_profile = {"email": owner_profile, "photo_b64": "", "trash_talk": ""}
    owner_photo_b64   = owner_profile.get("photo_b64", "")
    owner_trash_talk  = owner_profile.get("trash_talk", "")

    # Avatar HTML for popup
    initials = "".join([w[0].upper() for w in owner.split()[:2]])
    if owner_photo_b64:
        avatar_popup = (
            f'<img src="data:image/jpeg;base64,{owner_photo_b64}" '
            f'style="width:52px;height:52px;border-radius:50%;object-fit:cover;'
            f'border:2px solid #FFD700;display:block;margin:0 auto 6px;" />'
        )
    else:
        avatar_popup = (
            f'<div style="width:52px;height:52px;border-radius:50%;'
            f'background:linear-gradient(135deg,#c8a84b,#333);'
            f'display:flex;align-items:center;justify-content:center;'
            f'font-size:1.2rem;font-weight:700;color:#fff;'
            f'margin:0 auto 6px;border:2px solid #FFD700;">{initials}</div>'
        )

    # Trash-talk bubble
    trash_html = ""
    if owner_trash_talk:
        safe_talk = owner_trash_talk.replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
        trash_html = (
            f'<div style="background:#1a1a1a;border-left:3px solid #e94560;'
            f'border-radius:6px;padding:5px 8px;margin-top:8px;'
            f'font-size:0.8rem;color:#e94560;font-style:italic;">'
            f'💬 &ldquo;{safe_talk}&rdquo;</div>'
        )

    # Leaderboard rows
    lb_rows  = ""
    medals   = ["🥇","🥈","🥉"]
    for i, (_, row) in enumerate(a["leaderboard"].iterrows()):
        lb_rows += (
            f"<tr><td style='padding:2px 4px;'>{medals[i] if i < 3 else i+1}</td>"
            f"<td style='padding:2px 4px;'><b>{row['athlete']}</b></td>"
            f"<td style='padding:2px 4px;'>{row['distance_km']:.1f} km</td></tr>"
        )

    popup_html = f"""
    <div style="font-family:sans-serif;min-width:220px;max-width:270px;">
        <div style="background:#1a1a1a;color:#c8a84b;padding:8px 12px;
                    border-radius:6px 6px 0 0;font-weight:800;font-size:1rem;letter-spacing:1px;">
            📍 {a['area_name']}
        </div>
        <div style="padding:10px 12px;background:#111;border-radius:0 0 6px 6px;">
            {avatar_popup}
            <div style="text-align:center;color:#FFD700;font-weight:800;font-size:0.95rem;">
                👑 {owner}
            </div>
            <div style="text-align:center;color:#888;font-size:0.75rem;margin-bottom:6px;">
                {a['owner_km']:.1f} km total
            </div>
            {trash_html}
            <hr style="border-color:#333;margin:8px 0;" />
            <table style="width:100%;border-collapse:collapse;color:#fff;font-size:0.82rem;">
                {lb_rows}
            </table>
        </div>
    </div>
    """

    # Floating label — include first 30 chars of trash-talk if set
    trash_preview = ""
    if owner_trash_talk:
        short = owner_trash_talk[:30] + ("…" if len(owner_trash_talk) > 30 else "")
        safe  = short.replace("<","&lt;").replace(">","&gt;")
        trash_preview = f'<br><span style="font-size:9px;color:#e94560;font-style:italic;">💬 {safe}</span>'

    icon_html = f"""
    <div style="background:linear-gradient(135deg,#1a1200,#2a2000);border:2px solid #c8a84b;
                border-radius:8px;padding:5px 10px;color:#c8a84b;font-family:sans-serif;
                font-size:13px;font-weight:800;white-space:nowrap;
                box-shadow:0 2px 8px rgba(0,0,0,0.6);letter-spacing:0.5px;">
        👑 {owner} &nbsp;·&nbsp; {a['area_name']}<br>
        <span style="font-size:10px;color:#e8c87a;font-weight:600;">{a['owner_km']:.1f} km</span>
        {trash_preview}
    </div>
    """

    folium.Marker(
        location=[display_lat, a["lon"]],
        icon=folium.DivIcon(html=icon_html, icon_size=(250, 65), icon_anchor=(125, 32)),
        popup=folium.Popup(popup_html, max_width=290),
        tooltip=f"Click for leaderboard — {a['area_name']}",
    ).add_to(ownership_layer)

ownership_layer.add_to(m)

# ── Heatmap layer ──
heat_layer   = FeatureGroup(name="🔥 Activity Heatmap", show=False)
heat_points  = []
for _, run in df.iterrows():
    for pt in run["points"]:
        heat_points.append([pt[0], pt[1]])
if heat_points:
    HeatMap(heat_points, radius=12, blur=10, min_opacity=0.4).add_to(heat_layer)
heat_layer.add_to(m)

LayerControl(collapsed=False).add_to(m)

st_folium(m, width=None, height=520)

st.divider()
with st.expander("📋 All uploaded runs"):
    st.dataframe(
        df[["athlete","file_name","area_name","distance_km","duration_min","pace_min_per_km","run_date"]]
        .rename(columns={
            "athlete":"Athlete","file_name":"File","area_name":"Track",
            "distance_km":"KM","duration_min":"Duration (min)",
            "pace_min_per_km":"Pace (min/km)","run_date":"Date"
        }),
        hide_index=True, use_container_width=True,
    )
