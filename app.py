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

st.markdown("""
<style>
.stApp {
    background-image: url("data:image/jpeg;base64,...");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    background-repeat: no-repeat;
    background-blend-mode: overlay;
    background-color: rgba(14, 17, 23, 0.88);
    animation: dragonDrift 25s ease-in-out infinite;
}
@keyframes dragonDrift {
    0%   { background-position: center; }
    25%  { background-position: 55% 45%; }
    50%  { background-position: 48% 55%; }
    75%  { background-position: 53% 48%; }
    100% { background-position: center; }
}
.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    background: rgba(10, 10, 15, 0.72);
    z-index: 0;
    pointer-events: none;
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
# Avatar helper
# ──────────────────────────────────────────────────────────────────────────
def avatar_html(athlete, profiles, size=44, border_color="#c8a84b"):
    profile = profiles.get(athlete, {})
    if isinstance(profile, str):
        profile = {"email": profile, "photo_b64": "", "message": ""}
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

@st.cache_data(show_spinner=False)
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

@st.cache_data(show_spinner=False)
def build_dataframe(runs_tuple):
    runs = [json.loads(r) for r in runs_tuple]
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

def render_ai_coach(athlete, athlete_df, area_id=""):
    chat_key = f"chat_{athlete}_{area_id}"
    input_key = f"chat_input_{athlete}_{area_id}"

    with st.popover("🤖 AI Coach"):
        if not GROQ_KEY:
            st.info("Add GROQ_API_KEY to Streamlit secrets to enable AI coaching.")
            return

        # Initialise chat history and auto-generate opening advice
        if chat_key not in st.session_state:
            summary = build_athlete_summary(athlete_df)
            system_prompt = (
                f"You are an expert, friendly running coach. "
                f"You are coaching {athlete}. Here are their current stats:\n"
                f"{json.dumps(summary, indent=2)}\n\n"
                "Keep replies concise (3-5 sentences max). Be encouraging, practical, "
                "and personalised to their data. Use plain language, no jargon."
            )
            st.session_state[chat_key] = {
                "system": system_prompt,
                "messages": [],
                "opening_done": False,
            }

        # Auto-generate opening advice if not done yet
        chat_state = st.session_state[chat_key]
        if not chat_state.get("opening_done"):
            summary = build_athlete_summary(athlete_df)
            opening_prompt = (
                f"You are an expert, friendly running coach. "
                f"Here are the stats for {athlete}:\n"
                f"{json.dumps(summary, indent=2)}\n\n"
                "Give a short personalised opening assessment in exactly 2 parts:\n"
                "1. **What you\'re doing well** — 2 bullet points max\n"
                "2. **What to work on** — 2 bullet points max\n"
                "Be specific to their data. Keep it under 120 words total. "
                "End with one encouraging sentence inviting them to ask questions."
            )
            with st.spinner("Coach is preparing your assessment…"):
                try:
                    resp = requests.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
                        json={
                            "model": "llama-3.3-70b-versatile",
                            "messages": [{"role": "user", "content": opening_prompt}],
                            "max_tokens": 300,
                        },
                        timeout=20,
                    )
                    if resp.ok:
                        opening = resp.json()["choices"][0]["message"]["content"]
                    else:
                        opening = "Hey! I\'m your coach. Ask me anything about your training."
                except Exception:
                    opening = "Hey! I\'m your coach. Ask me anything about your training."

            chat_state["messages"].append({"role": "assistant", "content": opening})
            chat_state["opening_done"] = True
            st.rerun()

        # ── Render chat history ──
        st.markdown(
            "<div style='font-size:0.78rem;color:#c8a84b;font-weight:700;"
            "letter-spacing:1px;margin-bottom:6px;'>🤖 COACH CHAT</div>",
            unsafe_allow_html=True
        )

        if not chat_state["messages"]:
            st.markdown(
                "<div style='color:#888;font-size:0.82rem;font-style:italic;'>"
                "Ask your coach anything about your training…</div>",
                unsafe_allow_html=True
            )
        else:
            for msg in chat_state["messages"]:
                if msg["role"] == "user":
                    st.markdown(
                        f"<div style='background:rgba(200,168,75,0.12);border-radius:10px;"
                        f"padding:7px 10px;margin:4px 0;font-size:0.83rem;color:#ffe97d;'>"
                        f"🧑 {msg['content']}</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<div style='background:rgba(255,255,255,0.05);border-radius:10px;"
                        f"padding:7px 10px;margin:4px 0;font-size:0.83rem;color:#e0e0e0;'>"
                        f"🤖 {msg['content']}</div>",
                        unsafe_allow_html=True
                    )

        # ── Input box ──
        user_input = st.text_input(
            "Your message",
            key=input_key,
            placeholder="e.g. How do I improve my pace?",
            label_visibility="collapsed"
        )

        col_send, col_clear = st.columns([3, 1])
        send = col_send.button("Send ➤", key=f"send_{athlete}_{area_id}", use_container_width=True)
        clear = col_clear.button("🗑️", key=f"clear_{athlete}_{area_id}", use_container_width=True)

        if clear:
            del st.session_state[chat_key]
            st.rerun()

        if send and user_input.strip():
            chat_state["messages"].append({"role": "user", "content": user_input.strip()})

            # Build messages array for Groq: system injected as first user turn
            groq_messages = [
                {"role": "user", "content": chat_state["system"]},
                {"role": "assistant", "content": "Got it! I have your stats and I'm ready to coach you."},
            ]
            for m in chat_state["messages"]:
                groq_messages.append({"role": m["role"], "content": m["content"]})

            with st.spinner("Coach is thinking…"):
                try:
                    resp = requests.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
                        json={
                            "model": "llama-3.3-70b-versatile",
                            "messages": groq_messages,
                            "max_tokens": 300,
                        },
                        timeout=20,
                    )
                    if resp.ok:
                        reply = resp.json()["choices"][0]["message"]["content"]
                    else:
                        reply = "Sorry, I couldn't reach the coaching server. Try again!"
                except Exception:
                    reply = "Network error — please try again."

            chat_state["messages"].append({"role": "assistant", "content": reply})
            st.rerun()


# ──────────────────────────────────────────────────────────────────────────
# Badges
# ──────────────────────────────────────────────────────────────────────────
BADGES = [
    ("🏅 First Steps",    "Completed your first run",                lambda s: s["total_runs"] >= 1),
    ("🔟 10K Club",       "Ran 10 km total",                         lambda s: s["total_km"] >= 10),
    ("💯 Century",        "Ran 100 km total",                        lambda s: s["total_km"] >= 100),
    ("⚡ Speed Demon",    "Avg pace under 5 min/km",                 lambda s: s["avg_pace_min_per_km"] is not None and s["avg_pace_min_per_km"] < 5),
    ("🐢 Iron Legs",      "Completed 10+ runs",                      lambda s: s["total_runs"] >= 10),
    ("🚀 Rocket Start",   "Best pace under 4 min/km",                lambda s: s["best_pace_min_per_km"] is not None and s["best_pace_min_per_km"] < 4),
    ("📈 Improving",      "Pace trend is improving",                 lambda s: s["pace_trend"] == "improving"),
    ("🏃 Marathon Mark",  "Ran 42+ km total",                        lambda s: s["total_km"] >= 42),
    ("🌟 Streak King",    "Ran more than 5 times",                   lambda s: s["total_runs"] > 5),
    ("👑 Conqueror",      "Leader on at least one track",            lambda s: s.get("is_leader", False)),
]

def compute_badges(summary):
    earned = []
    for name, desc, check in BADGES:
        try:
            if check(summary):
                earned.append((name, desc))
        except Exception:
            pass
    return earned

def render_badges(badges, size="1.6rem"):
    if not badges:
        return "<span style='color:#555;font-size:0.8rem;'>No badges yet — keep running!</span>"
    html = "<div style='display:flex;flex-wrap:wrap;gap:6px;margin-top:4px;'>"
    for name, desc in badges:
        html += (
            f"<div title='{desc}' style='background:rgba(200,168,75,0.1);border:1px solid rgba(200,168,75,0.3);"
            f"border-radius:8px;padding:4px 10px;font-size:0.78rem;color:#ffe97d;cursor:default;"
            f"white-space:nowrap;'>{name}</div>"
        )
    html += "</div>"
    return html


# ──────────────────────────────────────────────────────────────────────────
# Head-to-Head rivalry
# ──────────────────────────────────────────────────────────────────────────
def call_h2h_coach(athlete_a, summary_a, athlete_b, summary_b):
    if not GROQ_KEY:
        return None
    prompt = (
        f"You are a competitive running coach analysing a rivalry between two runners.\n\n"
        f"Runner A — {athlete_a}:\n{json.dumps(summary_a, indent=2)}\n\n"
        f"Runner B — {athlete_b}:\n{json.dumps(summary_b, indent=2)}\n\n"
        f"Give your response in exactly this format (no extra text):\n"
        f"**{athlete_a}:** [1-2 sentence personalised tip on what they must do to overtake {athlete_b}]\n"
        f"**{athlete_b}:** [1-2 sentence personalised tip on what they must do to stay ahead of {athlete_a}]\n"
        f"**Verdict:** [1 sentence on who has the edge right now and why]"
    )
    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
            json={"model": "llama-3.3-70b-versatile",
                  "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": 300},
            timeout=20,
        )
        if resp.ok:
            return resp.json()["choices"][0]["message"]["content"]
    except Exception:
        pass
    return None

def render_h2h_section(area_id, area_df, leaderboard, profiles):
    """Render badges + H2H rivalry section inside the area expander."""
    athletes = leaderboard["athlete"].tolist()
    leader   = athletes[0]
    leader_km = float(leaderboard.iloc[0]["total_km"])

    # ── Badges per athlete ──
    st.markdown("### 🎖️ Achievements")
    for _, row in leaderboard.iterrows():
        ath = row["athlete"]
        ath_df = area_df[area_df["athlete"] == ath]
        summary = build_athlete_summary(ath_df)
        summary["is_leader"] = (ath == leader)
        badges = compute_badges(summary)
        av = avatar_html(ath, profiles, size=28, border_color="#c8a84b")
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:6px;'>"
            f"{av}"
            f"<span style='color:#fff;font-weight:600;font-size:0.85rem;'>{ath}</span>"
            f"</div>"
            f"{render_badges(badges)}",
            unsafe_allow_html=True
        )
        st.markdown("<div style='margin-bottom:10px;'></div>", unsafe_allow_html=True)

    # ── H2H: every non-leader vs leader ──
    if len(athletes) < 2:
        return

    st.markdown("### ⚔️ Rivalry Intel")
    st.markdown(
        f"<div style='color:#888;font-size:0.8rem;margin-bottom:10px;'>"
        f"Each runner vs 👑 {leader} — AI tips to close the gap</div>",
        unsafe_allow_html=True
    )

    leader_df      = area_df[area_df["athlete"] == leader]
    leader_summary = build_athlete_summary(leader_df)
    leader_summary["is_leader"] = True

    for challenger in athletes[1:]:
        h2h_key = f"h2h_{area_id}_{challenger}"
        gap_km  = round(leader_km - float(leaderboard[leaderboard["athlete"] == challenger]["total_km"].iloc[0]), 1)

        av_l = avatar_html(leader,     profiles, size=32, border_color="#FFD700")
        av_c = avatar_html(challenger, profiles, size=32, border_color="#e94560")

        st.markdown(f"""
        <div style='background:rgba(255,255,255,0.03);border:1px solid #333;border-radius:12px;
                    padding:12px 14px;margin-bottom:10px;'>
            <div style='display:flex;align-items:center;justify-content:center;gap:10px;margin-bottom:8px;'>
                <div style='display:flex;align-items:center;gap:6px;'>
                    {av_l}
                    <span style='color:#FFD700;font-weight:700;font-size:0.88rem;'>👑 {leader}</span>
                </div>
                <span style='color:#555;font-size:1.1rem;'>⚔️</span>
                <div style='display:flex;align-items:center;gap:6px;'>
                    {av_c}
                    <span style='color:#e94560;font-weight:700;font-size:0.88rem;'>{challenger}</span>
                </div>
            </div>
            <div style='text-align:center;color:#888;font-size:0.75rem;margin-bottom:8px;'>
                Gap: <b style='color:#fff;'>{gap_km} km</b> behind
            </div>
        </div>
        """, unsafe_allow_html=True)

        if h2h_key not in st.session_state:
            if st.button(f"⚡ Get AI rivalry tips — {challenger} vs {leader}",
                         key=f"h2h_btn_{area_id}_{challenger}"):
                chall_df      = area_df[area_df["athlete"] == challenger]
                chall_summary = build_athlete_summary(chall_df)
                chall_summary["is_leader"] = False
                with st.spinner("Analysing rivalry…"):
                    result = call_h2h_coach(challenger, chall_summary, leader, leader_summary)
                if result:
                    st.session_state[h2h_key] = result
                    st.rerun()
                else:
                    st.error("Couldn't reach Groq — check GROQ_API_KEY.")
        else:
            result = st.session_state[h2h_key]
            st.markdown(
                f"<div style='background:rgba(200,168,75,0.06);border-left:3px solid #c8a84b;"
                f"border-radius:8px;padding:10px 14px;font-size:0.83rem;color:#e0e0e0;"
                f"line-height:1.6;'>{result}</div>",
                unsafe_allow_html=True
            )
            if st.button(f"🔄 Refresh tips", key=f"h2h_refresh_{area_id}_{challenger}"):
                del st.session_state[h2h_key]
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
# RENAMED: "Trash-talk message" → "Message"
message_text  = st.sidebar.text_input(
    "Message 💬", placeholder="e.g. Can't catch me 😤",
    help="Shows on the map bubble when you own a track."
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

        profiles = load_profiles()
        existing = profiles.get(athlete_name.strip(), {})
        if isinstance(existing, str):
            existing = {"email": existing, "photo_b64": "", "message": ""}

        if athlete_email.strip():
            existing["email"] = athlete_email.strip()
        # Store as "message" key (was "trash_talk")
        if message_text.strip():
            existing["message"] = message_text.strip()
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
df       = build_dataframe(tuple(json.dumps(r, sort_keys=True) for r in runs)) if runs else pd.DataFrame()
profiles = load_profiles()

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
                render_ai_coach(athlete, area_df[area_df["athlete"] == athlete], area_id=area_id)

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

        # ── Badges on podium cards (inline under each top-3 athlete) ──
        st.markdown("<div style='margin-top:0.5rem;'></div>", unsafe_allow_html=True)
        for col_idx, rank_idx in enumerate(order[:n]):
            row     = top3.iloc[rank_idx]
            ath     = row["athlete"]
            ath_df  = area_df[area_df["athlete"] == ath]
            summary = build_athlete_summary(ath_df)
            summary["is_leader"] = (rank_idx == 0)
            badges  = compute_badges(summary)
            with cols[col_idx]:
                st.markdown(render_badges(badges), unsafe_allow_html=True)

        st.divider()
        # ── Achievements + H2H rivalry section ──
        render_h2h_section(area_id, area_df, leaderboard, profiles)

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

# ── Build area summary list ──
area_summaries = []
for area_id, area_name in df.groupby("area_id")["area_name"].first().items():
    sub       = df[df["area_id"] == area_id]
    owner_row = sub.groupby("athlete")["distance_km"].sum().sort_values(ascending=False)
    lb        = owner_row.reset_index().head(3)
    area_summaries.append({
        "area_id":    area_id,
        "area_name":  area_name,
        "lat":        sub["centroid_lat"].mean(),
        "lon":        sub["centroid_lon"].mean(),
        "owner":      owner_row.index[0],
        "owner_km":   float(owner_row.iloc[0]),
        "leaderboard": lb,
    })

# ── Compute bubble radii (metres) scaled by owner_km ──
# Bigger km → bigger circle. Min 120 m, max 600 m.
all_km = [a["owner_km"] for a in area_summaries]
max_km = max(all_km) if all_km else 1.0
MIN_R, MAX_R = 120, 600

def km_to_radius(km):
    ratio = (km / max_km) ** 0.5          # sqrt for area-proportional feel
    return MIN_R + ratio * (MAX_R - MIN_R)

for a in area_summaries:
    a["radius_m"] = km_to_radius(a["owner_km"])

# ── Non-overlap: sort largest first; push smaller circles to not overlap ──
# Convert lat/lon offset so smaller circles sit at the edge of the larger one.
DEG_PER_M_LAT = 1.0 / 111_320

area_summaries.sort(key=lambda a: a["owner_km"], reverse=True)

placed = []   # list of (lat, lon, radius_m)

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))

for a in area_summaries:
    lat, lon = a["lat"], a["lon"]
    r = a["radius_m"]
    deg_per_m_lon = 1.0 / (111_320 * math.cos(math.radians(lat)))

    # Check against every already-placed circle
    for (plat, plon, pr) in placed:
        dist = haversine_m(lat, lon, plat, plon)
        min_sep = pr + r         # circles must not overlap
        if dist < min_sep and dist > 0:
            # Push current circle away from the placed one
            # along the vector connecting their centres
            push = min_sep - dist + 5   # 5 m buffer
            angle = math.atan2(lat - plat, lon - plon)
            lat += push * math.cos(angle) * DEG_PER_M_LAT
            lon += push * math.sin(angle) * deg_per_m_lon

    a["display_lat"] = lat
    a["display_lon"] = lon
    placed.append((lat, lon, r))

# ── Per-area bubble colour palette (distinct colours) ──
BUBBLE_PALETTE = [
    "#FFD700",  # gold
    "#00CFFF",  # cyan
    "#FF4ECD",  # pink/magenta
    "#39FF14",  # neon green
    "#FF6B35",  # orange
    "#A855F7",  # purple
    "#00FFB3",  # mint
    "#FF3A3A",  # red
    "#4FC3F7",  # sky blue
    "#FFAA00",  # amber
]
area_bubble_color = {
    a["area_id"]: BUBBLE_PALETTE[i % len(BUBBLE_PALETTE)]
    for i, a in enumerate(area_summaries)
}

# ── Ownership layer ──
ownership_layer = FeatureGroup(name="👑 Ownership", show=True)

for a in area_summaries:
    owner      = a["owner"]
    owner_km   = a["owner_km"]
    area_name  = a["area_name"]
    dlat       = a["display_lat"]
    dlon       = a["display_lon"]
    radius_m   = a["radius_m"]

    owner_profile  = profiles.get(owner, {})
    if isinstance(owner_profile, str):
        owner_profile = {"email": owner_profile, "photo_b64": "", "message": ""}
    owner_message  = owner_profile.get("message", "") or owner_profile.get("trash_talk", "")
    owner_photo_b64 = owner_profile.get("photo_b64", "")

    # Per-area unique colour
    ring_color = area_bubble_color.get(a["area_id"], "#FFD700")

    # ── Folium Circle (the bubble) ──
    folium.Circle(
        location=[dlat, dlon],
        radius=radius_m,
        color=ring_color,
        weight=2.5,
        fill=True,
        fill_color=ring_color,
        fill_opacity=0.10,
    ).add_to(ownership_layer)

    # ── DivIcon label centred inside the circle ──
    # Three lines: crown+name, km, optional message
    msg_line = ""
    if owner_message:
        safe_msg = owner_message[:40].replace("<","&lt;").replace(">","&gt;")
        msg_line = f'<div style="font-size:10px;color:#e94560;font-style:italic;margin-top:3px;max-width:160px;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;">💬 {safe_msg}</div>'

    # Build avatar for the map label
    initials_map = "".join([w[0].upper() for w in owner.split()[:2]])
    if owner_photo_b64:
        avatar_map = (
            f'<img src="data:image/jpeg;base64,{owner_photo_b64}" '            f'style="width:38px;height:38px;border-radius:50%;object-fit:cover;'            f'border:2px solid {ring_color};margin-bottom:4px;display:block;" />'        )
    else:
        avatar_map = (
            f'<div style="width:38px;height:38px;border-radius:50%;'            f'background:linear-gradient(135deg,{ring_color},#333);'            f'display:flex;align-items:center;justify-content:center;'            f'font-size:13px;font-weight:700;color:#fff;'            f'border:2px solid {ring_color};margin-bottom:4px;">{initials_map}</div>'        )

    label_html = f"""
    <div style="
        display:flex; flex-direction:column; align-items:center; justify-content:center;
        text-align:center; pointer-events:none;
        text-shadow: 0 1px 4px #000, 0 0 8px #000;
    ">
        {avatar_map}
        <div style="font-size:12px; font-weight:800; color:#FFD700; letter-spacing:0.5px; white-space:nowrap;">
            👑 {owner}
        </div>
        <div style="font-size:11px; font-weight:600; color:#fff; margin-top:1px;">
            {owner_km:.1f} km
        </div>
        {msg_line}
    </div>
    """

    # icon_size / icon_anchor keep the label centred at (dlat, dlon)
    folium.Marker(
        location=[dlat, dlon],
        icon=folium.DivIcon(
            html=label_html,
            icon_size=(180, 80),
            icon_anchor=(90, 40),   # centre of the div
        ),
    ).add_to(ownership_layer)

    # ── Popup (click the circle or label) ──
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

    msg_popup = ""
    if owner_message:
        safe_msg = owner_message.replace("<","&lt;").replace(">","&gt;").replace('"',"&quot;")
        msg_popup = (
            f'<div style="background:#1a1a1a;border-left:3px solid #e94560;'
            f'border-radius:6px;padding:5px 8px;margin-top:8px;'
            f'font-size:0.8rem;color:#e94560;font-style:italic;">'
            f'💬 &ldquo;{safe_msg}&rdquo;</div>'
        )

    lb_rows = ""
    medals  = ["🥇","🥈","🥉"]
    for i, (_, row) in enumerate(a["leaderboard"].iterrows()):
        lb_rows += (
            f"<tr><td style='padding:2px 4px;'>{medals[i] if i<3 else i+1}</td>"
            f"<td style='padding:2px 4px;'><b>{row['athlete']}</b></td>"
            f"<td style='padding:2px 4px;'>{row['distance_km']:.1f} km</td></tr>"
        )

    popup_html = f"""
    <div style="font-family:sans-serif;min-width:220px;max-width:270px;">
        <div style="background:#1a1a1a;color:#c8a84b;padding:8px 12px;
                    border-radius:6px 6px 0 0;font-weight:800;font-size:1rem;letter-spacing:1px;">
            📍 {area_name}
        </div>
        <div style="padding:10px 12px;background:#111;border-radius:0 0 6px 6px;">
            {avatar_popup}
            <div style="text-align:center;color:#FFD700;font-weight:800;font-size:0.95rem;">
                👑 {owner}
            </div>
            <div style="text-align:center;color:#888;font-size:0.75rem;margin-bottom:6px;">
                {owner_km:.1f} km total
            </div>
            {msg_popup}
            <hr style="border-color:#333;margin:8px 0;" />
            <table style="width:100%;border-collapse:collapse;color:#fff;font-size:0.82rem;">
                {lb_rows}
            </table>
        </div>
    </div>
    """

    # Attach popup to the circle
    folium.Circle(
        location=[dlat, dlon],
        radius=radius_m,
        color="transparent",
        fill=True,
        fill_color="transparent",
        fill_opacity=0,
        popup=folium.Popup(popup_html, max_width=290),
        tooltip=f"👑 {owner} · {owner_km:.1f} km — click for details",
    ).add_to(ownership_layer)

ownership_layer.add_to(m)

# ── Heatmap layer ──
heat_layer  = FeatureGroup(name="🔥 Activity Heatmap", show=False)
heat_points = []
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
