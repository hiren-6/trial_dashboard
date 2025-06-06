import streamlit as st
import pandas as pd
import plotly.express as px
import pycountry
import pydeck as pdk
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import time

st.set_page_config(
    page_title="Clinical Trials Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -------------------------------------------
# 1) DATA LOADING FUNCTIONS
# -------------------------------------------
def load_dummy():
    """Load COVID.csv (dummy) from the same folder as app.py."""
    try:
        df = pd.read_csv("COVID.csv")
    except FileNotFoundError:
        st.error("❗ Could not find COVID.csv. Make sure app.py and COVID.csv are side by side.")
        return None
    return df


def load_uploaded(uploaded_file):
    """Load an uploaded CSV or Excel file into a DataFrame."""
    if uploaded_file is None:
        return None
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            return pd.read_csv(uploaded_file)
        else:
            return pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"❗ Failed to read uploaded file: {e}")
        return None


# -------------------------------------------
# 2) PREPROCESS & HELPER FUNCTIONS
# -------------------------------------------
def preprocess_for_geography(df: pd.DataFrame) -> pd.DataFrame:
    """
    For the Geography page, we assume columns like 'Study Status' etc. are present by name.
    We simply return the DataFrame as‐is, since the existing geography logic uses named columns.
    """
    return df.copy()


def extract_country_counts(loc_series: pd.Series) -> pd.DataFrame:
    """
    Fallback: if city‐level hex map fails, aggregate by country.
    From “City, Country | City2, Country2” → count countries → map to ISO α‐3.
    """
    all_countries = []
    for loc in loc_series.dropna():
        pieces = [p.strip() for p in loc.split("|")]
        for part in pieces:
            if "," in part:
                country = part.split(",")[-1].strip()
            else:
                country = part.strip()
            all_countries.append(country)

    if not all_countries:
        return pd.DataFrame(columns=["country", "iso_alpha3", "count"])

    counts = pd.Series(all_countries).value_counts().reset_index()
    counts.columns = ["country", "count"]

    def to_iso(name):
        try:
            return pycountry.countries.lookup(name).alpha_3
        except Exception:
            return None

    counts["iso_alpha3"] = counts["country"].apply(to_iso)
    counts = counts.dropna(subset=["iso_alpha3"]).copy()
    return counts


def geocode_city_country(city_country: str, geolocator, cache: dict) -> tuple[float, float] | None:
    """
    Given a "City, Country" string, return (lat, lon). Cache results in memory.
    """
    if city_country in cache:
        return cache[city_country]

    try:
        time.sleep(1)  # rate‐limit for Nominatim
        loc = geolocator.geocode(city_country, timeout=10)
        if loc:
            coords = (loc.latitude, loc.longitude)
            cache[city_country] = coords
            return coords
        else:
            cache[city_country] = None
            return None
    except (GeocoderTimedOut, GeocoderUnavailable):
        cache[city_country] = None
        return None
    except Exception:
        cache[city_country] = None
        return None


def build_city_hotspots(df_geo: pd.DataFrame) -> pd.DataFrame:
    """
    From df_geo (filtered by Study Status), parse each "City, Country" piece,
    geocode it, and return a DataFrame { city_country, lat, lon } per trial‐city.
    """
    if "geo_cache" not in st.session_state:
        st.session_state["geo_cache"] = {}

    geolocator = Nominatim(user_agent="clinical_trials_dashboard")
    geo_cache = st.session_state["geo_cache"]

    rows = []
    for _, row in df_geo.iterrows():
        loc_field = row["Location"]
        if pd.isna(loc_field):
            continue
        pieces = [p.strip() for p in loc_field.split("|")]
        for piece in pieces:
            city_country = piece  # e.g. "London, United Kingdom"
            coords = geocode_city_country(city_country, geolocator, geo_cache)
            if coords:
                lat, lon = coords
                rows.append({"city_country": city_country, "lat": lat, "lon": lon})

    if not rows:
        return pd.DataFrame(columns=["city_country", "lat", "lon"])
    return pd.DataFrame(rows)


def compute_timeline_averages(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Using positional indexing, extract the five critical columns:
      - Phase (column Q → index 16)
      - Funder Type (column S → index 18)
      - Start Date (column W → index 22)
      - Study Completion Date (column X → index 23)
      - Primary Completion Date (column Y → index 24)

    Compute durations (in months) for:
      - Overall → (Study Completion – Start) / 30
      - Primary → (Primary Completion – Start) / 30

    Then group by Phase and compute mean durations separately for:
      - Academic
      - Industry

    Return two DataFrames: avg_academic, avg_industry, each having columns:
      ["Phases", "avg_primary_months", "avg_overall_months"]
    """
    df_tl = df.copy()

    # 1) Extract the positional columns into named columns
    df_tl["Phase"] = df_tl.iloc[:, 16]
    df_tl["Funder"] = df_tl.iloc[:, 18]
    df_tl["Start"] = pd.to_datetime(df_tl.iloc[:, 22], errors="coerce")
    df_tl["Completion"] = pd.to_datetime(df_tl.iloc[:, 23], errors="coerce")
    df_tl["Primary"] = pd.to_datetime(df_tl.iloc[:, 24], errors="coerce")

    # 2) Compute durations in months (days ÷ 30)
    df_tl_
