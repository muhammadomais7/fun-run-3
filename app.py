"""
Running Leaderboard — GPX Edition
----------------------------------
No Strava API, no subscriptions, no API keys.
Runners export a .gpx file from Strava / Garmin / Apple Health / their phone's
GPS app and upload it here. The app:
  1. Parses the GPX route, distance, duration, pace
  2. Auto-detects "areas" (tracks / parks / loops) by clustering nearby runs
  3. Names each area using free OpenStreetMap reverse-geocoding
  4. Builds a leaderboard per area — whoever has the most KM "owns" that track
  5. Draws everything on a free OpenStreetMap-based map
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
st.set_page_config(page_title="Running Leaderboard", page_icon="🏃", layout="wide")

DATA_DIR = "data"
RUNS_FILE = os.path.join(DATA_DIR, "runs.json")
AREA_CACHE_FILE = os.path.join(DATA_DIR, "area_names.json")

EARTH_RADIUS_KM = 6371.0088
CLUSTER_RADIUS_KM = 0.2  # runs within ~200m of each other count as the same track
NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
NOMINATIM_HEADERS = {"User-Agent": "running-leaderboard-app/1.0"}

os.makedirs(DATA_DIR, exist_ok=True)


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
    """Parse a GPX file into a run record dict, or None if it has no usable points."""
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
        # Keep a lightly simplified point list so the map doesn't get huge
        "points": points[:: max(1, len(points) // 300)],
    }


# ──────────────────────────────────────────────────────────────────────────
# Area clustering
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
            headers=NOMINATIM_HEADERS,
            timeout=5,
        )
        if resp.ok:
            data = resp.json()
            addr = data.get("address", {})
            name = (
                addr.get("leisure")
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
        clat, clon = sub["centroid_lat"].mean(), sub["centroid_lon"].mean()
        area_names[area_id] = get_area_name(clat, clon, cache)
    save_area_cache(cache)

    df["area_name"] = df["area_id"].map(area_names)
    return df


# ──────────────────────────────────────────────────────────────────────────
# UI — Sidebar: upload
# ──────────────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Add Your Run")
athlete_name = st.sidebar.text_input("Your name", placeholder="e.g. Bilal Ahmed")
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
        added = 0
        for f in uploaded_files:
            record = parse_gpx(f.read(), athlete_name.strip(), f.name)
            if record:
                runs.append(record)
                added += 1
        save_runs(runs)
        st.sidebar.success(f"Added {added} run(s)! Scroll down to see your spot on the board.")
        st.rerun()

st.sidebar.divider()
st.sidebar.caption(
    "💡 How to get a GPX file:\n\n"
    "- **Strava**: open an activity → ⋯ menu → Export GPX\n"
    "- **Garmin Connect**: activity → gear icon → Export to GPX\n"
    "- **Apple Health / phone GPS apps**: most running apps have a GPX export option"
)

if st.sidebar.button("🗑️ Reset all data"):
    save_runs([])
    save_area_cache({})
    st.sidebar.success("Cleared!")
    st.rerun()


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────
st.title("🏃 Running Leaderboard")
st.caption("Powered by free GPX uploads + OpenStreetMap — no Strava subscription required.")

runs = load_runs()

if not runs:
    st.info("No runs yet! Upload a GPX file from the sidebar to get started. 👈")
    st.stop()

df = build_dataframe(runs)

# ---- Overall stats ----
col1, col2, col3 = st.columns(3)
col1.metric("🏃 Total Runs", len(df))
col2.metric("📏 Total KM", f"{df['distance_km'].sum():,.1f} km")
col3.metric("📍 Tracks Found", df["area_id"].nunique())

st.divider()

# ---- Per-area leaderboard ----
st.header("🏆 Leaderboard by Track")

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

    with st.expander(f"📍 **{area_name}** — 👑 held by **{owner}**", expanded=True):
        medals = ["🥇", "🥈", "🥉"]
        display_df = leaderboard.copy()
        display_df.insert(
            0, "", [medals[i] if i < 3 else "" for i in range(len(display_df))]
        )
        display_df = display_df.rename(
            columns={
                "athlete": "Athlete",
                "total_km": "Total KM",
                "runs": "Runs",
                "best_pace": "Best Pace (min/km)",
            }
        )
        st.dataframe(display_df, hide_index=True, use_container_width=True)

st.divider()

# ---- Map ----
st.header("🗺️ Map")

avg_lat, avg_lon = df["centroid_lat"].mean(), df["centroid_lon"].mean()
m = folium.Map(location=[avg_lat, avg_lon], zoom_start=13, tiles="OpenStreetMap")

colors = [
    "red", "blue", "green", "purple", "orange", "darkred",
    "cadetblue", "darkgreen", "darkblue", "pink",
]
area_color_map = {
    area_id: colors[i % len(colors)]
    for i, area_id in enumerate(df["area_id"].unique())
}

for _, run in df.iterrows():
    pts = run["points"]
    if len(pts) >= 2:
        folium.PolyLine(
            pts,
            color=area_color_map[run["area_id"]],
            weight=3,
            opacity=0.6,
            tooltip=f"{run['athlete']} — {run['distance_km']} km",
        ).add_to(m)

# Build one summary row per area, then draw bubbles smallest-total-km first.
# In Leaflet, layers added later render on top, so drawing biggest-last means
# a high-km bubble is never hidden underneath a smaller, nearby one.
area_summaries = []
for area_id, area_name in df.groupby("area_id")["area_name"].first().items():
    sub = df[df["area_id"] == area_id]
    owner_row = sub.groupby("athlete")["distance_km"].sum().sort_values(ascending=False)
    area_summaries.append(
        {
            "area_id": area_id,
            "area_name": area_name,
            "lat": sub["centroid_lat"].mean(),
            "lon": sub["centroid_lon"].mean(),
            "total_km": owner_row.sum(),
            "owner": owner_row.index[0],
            "owner_km": owner_row.iloc[0],
        }
    )

area_summaries.sort(key=lambda a: a["total_km"])  # smallest first → drawn first → ends up underneath

MIN_RADIUS, MAX_RADIUS = 10, 32
max_km_overall = max(a["total_km"] for a in area_summaries) or 1

for a in area_summaries:
    # Scale by sqrt so bubble *area* (not just radius) reflects km proportionally
    scale = math.sqrt(a["total_km"] / max_km_overall) if max_km_overall else 0
    radius = MIN_RADIUS + scale * (MAX_RADIUS - MIN_RADIUS)

    folium.CircleMarker(
        location=[a["lat"], a["lon"]],
        radius=radius,
        color=area_color_map[a["area_id"]],
        fill=True,
        fill_color=area_color_map[a["area_id"]],
        fill_opacity=0.75,
        weight=2,
        popup=f"<b>{a['area_name']}</b><br>👑 {a['owner']} ({a['owner_km']:.1f} km)<br>Total: {a['total_km']:.1f} km",
        tooltip=f"{a['area_name']} — 👑 {a['owner']} ({a['total_km']:.1f} km total)",
    ).add_to(m)

st_folium(m, width=None, height=500)

st.divider()
with st.expander("📋 All uploaded runs"):
    st.dataframe(
        df[["athlete", "file_name", "area_name", "distance_km", "duration_min", "pace_min_per_km", "run_date"]]
        .rename(
            columns={
                "athlete": "Athlete",
                "file_name": "File",
                "area_name": "Track",
                "distance_km": "KM",
                "duration_min": "Duration (min)",
                "pace_min_per_km": "Pace (min/km)",
                "run_date": "Date",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )
