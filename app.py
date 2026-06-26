"""
Running Leaderboard — GPX Edition  (v2: Territory Conquest)
------------------------------------------------------------
- Each runner gets a unique color
- Conquered areas shown as colored bubbles on the map
- Whoever has most KM in an area owns it (bubble = their color)
- Overlapping bubbles: the runner with most KM wins that zone
- Official place name (OpenStreetMap) stays permanent
- "Conquered by [Name]" shown beside the track name
- Actual route line drawn on map in runner's color
"""

import json
import math
import os
from datetime import datetime
from io import BytesIO

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
st.set_page_config(page_title="Territory Run", page_icon="⚔️", layout="wide")

DATA_DIR = "data"
RUNS_FILE = os.path.join(DATA_DIR, "runs.json")
AREA_CACHE_FILE = os.path.join(DATA_DIR, "area_names.json")

EARTH_RADIUS_KM = 6371.0088
CLUSTER_RADIUS_KM = 0.2
NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
NOMINATIM_HEADERS = {"User-Agent": "territory-run-leaderboard/2.0"}

# Conquest bubble radius in meters
BUBBLE_RADIUS_M = 180

os.makedirs(DATA_DIR, exist_ok=True)

# ── Athlete color palette (up to 10 athletes) ──
ATHLETE_COLORS = [
    "#2ECC71",  # green
    "#E74C3C",  # red
    "#3498DB",  # blue
    "#F39C12",  # orange
    "#9B59B6",  # purple
    "#1ABC9C",  # teal
    "#E91E63",  # pink
    "#FF5722",  # deep orange
    "#607D8B",  # blue grey
    "#CDDC39",  # lime
]

FOLIUM_COLORS = [
    "green", "red", "blue", "orange", "purple",
    "cadetblue", "pink", "darkred", "darkblue", "lightgreen",
]


# ──────────────────────────────────────────────────────────────────────────
# Storage helpers
# ──────────────────────────────────────────────────────────────────────────
def load_json(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_runs():
    return load_json(RUNS_FILE, [])


def save_runs(runs):
    save_json(RUNS_FILE, runs)


def load_area_cache():
    return load_json(AREA_CACHE_FILE, {})


def save_area_cache(cache):
    save_json(AREA_CACHE_FILE, cache)


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
        "centroid_lat": sum(lats) / len(lats),
        "centroid_lon": sum(lons) / len(lons),
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
    name = f"Track ({lat:.3f}, {lon:.3f})"
    try:
        resp = requests.get(
            NOMINATIM_URL,
            params={"lat": lat, "lon": lon, "format": "json", "zoom": 17},
            headers=NOMINATIM_HEADERS,
            timeout=5,
        )
        if resp.ok:
            data = resp.json()
            addr = data.get("address", {})
            name = (
                addr.get("leisure")
                or addr.get("amenity")
                or addr.get("park")
                or addr.get("road")
                or addr.get("suburb")
                or addr.get("neighbourhood")
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
        clat = sub["centroid_lat"].mean()
        clon = sub["centroid_lon"].mean()
        area_names[area_id] = get_area_name(clat, clon, cache)
    save_area_cache(cache)

    df["area_name"] = df["area_id"].map(area_names)
    return df


def get_athlete_color_map(athletes):
    """Assign a stable hex color + folium color to each athlete."""
    color_map = {}
    for i, name in enumerate(sorted(set(athletes))):
        color_map[name] = {
            "hex": ATHLETE_COLORS[i % len(ATHLETE_COLORS)],
            "folium": FOLIUM_COLORS[i % len(FOLIUM_COLORS)],
        }
    return color_map


# ──────────────────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Big bold conquest header */
.conquest-header {
    font-size: 2rem;
    font-weight: 800;
    margin-bottom: 0.2rem;
}
/* Athlete color badge */
.badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.85rem;
    color: white;
    margin-right: 6px;
}
/* Track card */
.track-card {
    border-left: 5px solid;
    border-radius: 6px;
    padding: 12px 18px;
    margin-bottom: 14px;
    background: rgba(255,255,255,0.04);
}
.track-title {
    font-size: 1.15rem;
    font-weight: 700;
    margin-bottom: 2px;
}
.conquered-by {
    font-size: 1rem;
    opacity: 0.85;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────
st.sidebar.header("⚔️ Add Your Run")
athlete_name = st.sidebar.text_input("Your name", placeholder="e.g. Omais")
uploaded_files = st.sidebar.file_uploader(
    "Upload GPX file(s)", type=["gpx"], accept_multiple_files=True
)

if st.sidebar.button("🚀 Conquer Territory", type="primary"):
    if not athlete_name.strip():
        st.sidebar.error("Please enter your name first.")
    elif not uploaded_files:
        st.sidebar.error("Please upload at least one .gpx file.")
    else:
        runs = load_runs()
        added = 0
        for f in uploaded_files:
            record = parse_gpx(f.read(), athlete_name.strip(), f.name)
            if record:
                runs.append(record)
                added += 1
        save_runs(runs)
        st.sidebar.success(f"✅ {added} run(s) added! Territory updated.")
        st.rerun()

st.sidebar.divider()
st.sidebar.caption(
    "💡 **How to get a GPX file:**\n\n"
    "- **Strava**: activity → ⋯ menu → Export GPX\n"
    "- **Garmin Connect**: activity → gear icon → Export to GPX\n"
    "- **Any GPS running app**: look for Export or Share → GPX"
)

if st.sidebar.button("🗑️ Reset all data"):
    save_runs([])
    save_area_cache({})
    st.rerun()


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────
st.markdown('<div class="conquest-header">⚔️ Territory Run — Conquer Your City</div>', unsafe_allow_html=True)
st.caption("Upload your GPX runs. The more you run an area, the more territory you own.")

runs = load_runs()

if not runs:
    st.info("No runs yet! Upload a GPX file from the sidebar to claim your first territory. 👈")
    st.stop()

df = build_dataframe(runs)

athlete_color_map = get_athlete_color_map(df["athlete"].tolist())

# ── Summary stats ──
col1, col2, col3, col4 = st.columns(4)
col1.metric("⚔️ Total Runs", len(df))
col2.metric("📏 Total KM", f"{df['distance_km'].sum():,.1f} km")
col3.metric("📍 Territories", df["area_id"].nunique())
col4.metric("🏃 Runners", df["athlete"].nunique())

st.divider()

# ── Athlete legend ──
st.subheader("🎨 Runners")
legend_cols = st.columns(min(len(athlete_color_map), 5))
for i, (name, colors) in enumerate(athlete_color_map.items()):
    total_km = df[df["athlete"] == name]["distance_km"].sum()
    areas_owned = 0
    for area_id in df["area_id"].unique():
        sub = df[df["area_id"] == area_id]
        owner = sub.groupby("athlete")["distance_km"].sum().idxmax()
        if owner == name:
            areas_owned += 1
    with legend_cols[i % 5]:
        st.markdown(
            f'<div style="border-left: 5px solid {colors["hex"]}; padding: 8px 12px; border-radius:4px; margin-bottom:8px;">'
            f'<b style="color:{colors["hex"]}; font-size:1.1rem;">{name}</b><br>'
            f'<span style="font-size:0.85rem;">{total_km:.1f} km &nbsp;|&nbsp; {areas_owned} territory owned</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.divider()

# ── Leaderboard by track ──
st.subheader("🏆 Territory Leaderboard")

area_totals = (
    df.groupby(["area_id", "area_name"])["distance_km"]
    .sum()
    .reset_index()
    .sort_values("distance_km", ascending=False)
)

for _, area_row in area_totals.iterrows():
    area_id = area_row["area_id"]
    area_name = area_row["area_name"]
    area_df = df[df["area_id"] == area_id]

    leaderboard = (
        area_df.groupby("athlete")
        .agg(
            total_km=("distance_km", "sum"),
            runs=("distance_km", "count"),
            best_pace=("pace_min_per_km", "min"),
        )
        .reset_index()
        .sort_values("total_km", ascending=False)
    )
    owner = leaderboard.iloc[0]["athlete"]
    owner_color = athlete_color_map[owner]["hex"]
    owner_km = leaderboard.iloc[0]["total_km"]

    st.markdown(
        f'<div class="track-card" style="border-color:{owner_color};">'
        f'<div class="track-title">📍 {area_name}</div>'
        f'<div class="conquered-by">'
        f'<span class="badge" style="background:{owner_color};">⚔️ Conquered by {owner}</span>'
        f'{owner_km:.2f} km logged here'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    medals = ["🥇", "🥈", "🥉"]
    display_df = leaderboard.copy()
    display_df.insert(0, "", [medals[i] if i < 3 else "" for i in range(len(display_df))])
    display_df = display_df.rename(columns={
        "athlete": "Athlete",
        "total_km": "Total KM",
        "runs": "Runs",
        "best_pace": "Best Pace (min/km)",
    })
    st.dataframe(display_df, hide_index=True, use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)

st.divider()

# ── Map: territory bubbles + route lines ──
st.subheader("🗺️ Territory Map")
st.caption("Colored bubbles = conquered zones. Lines = actual routes run. Bigger bubble = more KM logged.")

avg_lat = df["centroid_lat"].mean()
avg_lon = df["centroid_lon"].mean()
m = folium.Map(location=[avg_lat, avg_lon], zoom_start=14, tiles="OpenStreetMap")

# Draw route lines first (underneath bubbles)
for _, run in df.iterrows():
    pts = run["points"]
    athlete = run["athlete"]
    hex_color = athlete_color_map[athlete]["hex"]
    if len(pts) >= 2:
        folium.PolyLine(
            pts,
            color=hex_color,
            weight=3,
            opacity=0.75,
            tooltip=f"{athlete} — {run['distance_km']} km",
        ).add_to(m)

# Draw conquest bubbles per area (owner's color)
for area_id in df["area_id"].unique():
    sub = df[df["area_id"] == area_id]
    area_name = sub["area_name"].iloc[0]
    clat = sub["centroid_lat"].mean()
    clon = sub["centroid_lon"].mean()

    owner_series = sub.groupby("athlete")["distance_km"].sum().sort_values(ascending=False)
    owner = owner_series.index[0]
    owner_km = owner_series.iloc[0]
    owner_hex = athlete_color_map[owner]["hex"]
    owner_folium = athlete_color_map[owner]["folium"]

    # Bubble size scales with KM (min 100m, max 400m radius)
    radius_m = min(400, max(100, int(owner_km * 60)))

    folium.Circle(
        location=[clat, clon],
        radius=radius_m,
        color=owner_hex,
        fill=True,
        fill_color=owner_hex,
        fill_opacity=0.25,
        weight=2,
        tooltip=f"⚔️ {area_name} — Conquered by {owner} ({owner_km:.1f} km)",
    ).add_to(m)

    # Flag marker at center
    folium.Marker(
        location=[clat, clon],
        popup=folium.Popup(
            f"<b>{area_name}</b><br>⚔️ Conquered by <b>{owner}</b><br>{owner_km:.2f} km logged",
            max_width=200,
        ),
        tooltip=f"⚔️ {owner} owns this",
        icon=folium.Icon(color=owner_folium, icon="flag"),
    ).add_to(m)

st_folium(m, width=None, height=550)

st.divider()
with st.expander("📋 All uploaded runs"):
    st.dataframe(
        df[["athlete", "file_name", "area_name", "distance_km", "duration_min", "pace_min_per_km", "run_date"]]
        .rename(columns={
            "athlete": "Athlete",
            "file_name": "File",
            "area_name": "Track",
            "distance_km": "KM",
            "duration_min": "Duration (min)",
            "pace_min_per_km": "Pace (min/km)",
            "run_date": "Date",
        }),
        hide_index=True,
        use_container_width=True,
    )
