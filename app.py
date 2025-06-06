import streamlit as st
import pandas as pd
import plotly.express as px
import pycountry
import pydeck as pdk
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderUnavailable, GeocoderTimedOut
import time

st.set_page_config(
    page_title="Clinical Trials Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_dummy():
    """Load COVID.csv from the same folder as app.py (dummy data)."""
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


def preprocess_dates_and_durations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert date columns to datetime, then compute durations (in months)
    for overall and primary completion.
    """
    df = df.copy()
    for col in ["Start Date", "Primary Completion Date", "Completion Date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df["Duration Overall (Months)"] = (
        (df["Completion Date"] - df["Start Date"]).dt.days.astype("float") / 30.0
    )
    df["Duration Primary (Months)"] = (
        (df["Primary Completion Date"] - df["Start Date"]).dt.days.astype("float") / 30.0
    )
    return df


def months_to_yr_mo_string(m: float) -> str:
    """
    Convert a float number of months into a string like '2 yr 3 mo' or '5 mo'.
    """
    if pd.isna(m) or m < 0:
        return "-"
    total_months = int(round(m))
    yrs = total_months // 12
    mos = total_months % 12
    if yrs > 0 and mos > 0:
        return f"{yrs} yr {mos} mo"
    elif yrs > 0:
        return f"{yrs} yr"
    else:
        return f"{mos} mo"


def extract_country_counts(loc_series: pd.Series) -> pd.DataFrame:
    """
    For country‐level fallback: extract country names from "City, Country | City2, Country2",
    count occurrences, and map to ISO α3.
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
    Given a "City, Country" string, return (lat, lon). Use cache dict to avoid re‐geocoding.
    On failure or missing, return None.
    """
    if city_country in cache:
        return cache[city_country]

    try:
        # Pause briefly to respect Nominatim rate limits
        time.sleep(1)
        loc = geolocator.geocode(city_country, timeout=10)
        if loc:
            coords = (loc.latitude, loc.longitude)
            cache[city_country] = coords
            return coords
        else:
            cache[city_country] = None
            return None
    except (GeocoderTimedOut, GeocoderUnavailable):
        # If geocoding fails, store None to avoid retry loops
        cache[city_country] = None
        return None
    except Exception:
        cache[city_country] = None
        return None


def build_city_hotspots(df_geo: pd.DataFrame) -> pd.DataFrame:
    """
    From the filtered df_geo (filtered by Study Status), parse each "City, Country",
    geocode to lat/lon, and return a DataFrame with one row per trial‐city:
    { "city_country", "latitude", "longitude" }.
    """
    # Initialize geocoding cache in session_state if not already present
    if "geo_cache" not in st.session_state:
        st.session_state["geo_cache"] = {}

    geolocator = Nominatim(user_agent="clinical_trials_dashboard")
    geo_cache = st.session_state["geo_cache"]

    rows = []
    for idx, row in df_geo.iterrows():
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


def main():
    st.sidebar.title("Data & Navigation")

    # ——— 1) Data source selection ———
    data_source = st.sidebar.radio("1) Data Source", ("Use Dummy Data", "Upload CSV/Excel"))

    if data_source == "Use Dummy Data":
        df = load_dummy()
        if df is None:
            st.stop()
    else:
        uploaded = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
        df = load_uploaded(uploaded)
        if df is None:
            st.info("Please upload a valid CSV or Excel to proceed.")
            st.stop()

    # ——— 2) Required‐column check ———
    required_cols = [
        "NCT Number",
        "Study Title",
        "Study Status",
        "Phases",
        "Funder Type",
        "Start Date",
        "Primary Completion Date",
        "Completion Date",
        "Location",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"❗ Missing required columns: {missing}")
        st.stop()

    # ——— 3) Preprocess dates & durations ———
    df_processed = preprocess_dates_and_durations(df)

    # ——— 4) Sidebar: page navigation ———
    page = st.sidebar.radio("2) Navigate to", ("Home", "Geography", "Timelines"))

    # ——— Home page ———
    if page == "Home":
        st.title("🏠 Clinical Trials Dashboard")
        st.markdown(
            """
            **Welcome!**  
            Explore clinical trials by location and timeline.

            **Instructions**  
            1. In the sidebar, pick **Use Dummy Data** or **Upload CSV/Excel** matching the dummy format.  
            2. Then navigate to **Home**, **Geography**, or **Timelines**.  
            """
        )

        st.subheader("Download Dummy CSV Format")
        sample = load_dummy()
        if sample is not None:
            csv_bytes = sample.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download dummy_clinical_trials.csv",
                data=csv_bytes,
                file_name="dummy_clinical_trials.csv",
                mime="text/csv",
            )
        else:
            st.info("No dummy data available (COVID.csv missing).")

        st.subheader("Preview of Your Data (first 10 rows)")
        st.dataframe(df_processed.head(10), use_container_width=True)

    # ——— Geography page with city‐level hotspots ———
    elif page == "Geography":
        st.title("🌍 Geography Dashboard (City‐Level Hotspots)")
        st.markdown(
            "An interactive hex‐bin map showing trial hotspots by city. Filter by **Study Status**."
        )

        # Sidebar filter: Study Status
        statuses = sorted(df_processed["Study Status"].dropna().unique())
        selected_status = st.sidebar.multiselect("Filter: Study Status", statuses, default=statuses)

        df_geo = df_processed[df_processed["Study Status"].isin(selected_status)].copy()
        if df_geo.shape[0] == 0:
            st.warning("No trials match the selected Study Status. Adjust filter.")
            st.stop()

        with st.spinner("Geocoding cities (may take a moment on first run)…"):
            city_df = build_city_hotspots(df_geo)

        if city_df.empty:
            st.warning("No city‐level coordinates could be extracted/geocoded.")
            # Fallback: country‐level choropleth
            st.info("Showing a country‐level map instead.")
            country_counts = extract_country_counts(df_geo["Location"])
            if country_counts.empty:
                st.error("Even country extraction failed; please check your `Location` format.")
                st.stop()

            fig = px.choropleth(
                country_counts,
                locations="iso_alpha3",
                color="count",
                hover_name="country",
                color_continuous_scale="Viridis",
                projection="natural earth",
                labels={"count": "Number of Trials"},
                title="🗺 Number of Trials per Country",
            )
            fig.update_layout(
                margin=dict(l=0, r=0, t=50, b=0), coloraxis_colorbar=dict(title="Trials")
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                "- Each row’s `Location` was parsed to a country (text after the last comma).  \n"
                "- Hover on any country to see its trial count."
            )
            return

        # Build a HexagonLayer: we duplicate each city‐level coordinate per trial occurrence
        # (i.e., if 3 trials in London, London appears 3× in city_df)
        # Actually, build a list of dicts with lat/lon per trial
        # (we already did this in build_city_hotspots).

        # Center the initial view on the mean latitude/longitude (or default to [0,0])
        if not city_df[["lat", "lon"]].dropna().empty:
            mid_lat = city_df["lat"].mean()
            mid_lon = city_df["lon"].mean()
        else:
            mid_lat, mid_lon = 0, 0

        # Create Pydeck HexagonLayer
        hex_layer = pdk.Layer(
            "HexagonLayer",
            data=city_df,
            get_position=["lon", "lat"],
            radius=50000,  # 50 km radius per hex
            elevation_scale=50,
            elevation_range=[0, 3000],
            pickable=True,
            extruded=True,
            coverage=0.8,
        )

        view_state = pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=2, bearing=0, pitch=40)

        tooltip = {
            "html": "<b>Count of Trials:</b> <br/> {elevationValue}",
            "style": {"backgroundColor": "steelblue", "color": "white"},
        }

        deck = pdk.Deck(
            layers=[hex_layer],
            initial_view_state=view_state,
            tooltip=tooltip,
            map_style="mapbox://styles/mapbox/light-v10",
        )

        st.pydeck_chart(deck)

        st.markdown(
            """
            - **HexagonLayer** groups nearby trial points into hex bins.  
            - The taller/darker a hexagon, the more trials in that area.  
            - Hover on a hexagon to see the count of trials in that bin.  
            - If city‐level geocoding fails, a fallback country‐level choropleth is shown.  
            """
        )

    # ——— Timelines page (average durations by Phase) ———
    else:  # page == "Timelines"
        st.title("⏱ Timeline Dashboard (Avg Durations by Phase)")
        st.markdown("Compare average trial durations (overall vs. primary). Filter by **Funder Type**.")

        # Funder Type toggle
        funder_option = st.sidebar.radio("Funder Type", ("Both", "Industry Only", "Academic Only"))
        if funder_option == "Industry Only":
            df_time = df_processed[df_processed["Funder Type"].str.contains("Industry", na=False)].copy()
        elif funder_option == "Academic Only":
            df_time = df_processed[df_processed["Funder Type"].str.contains("Academic", na=False)].copy()
        else:
            df_time = df_processed.copy()

        # Drop invalid durations
        df_time = df_time[
            df_time["Duration Overall (Months)"].notna()
            & (df_time["Duration Overall (Months)"] >= 0)
            & df_time["Duration Primary (Months)"].notna()
            & (df_time["Duration Primary (Months)"] >= 0)
        ]
        if df_time.shape[0] == 0:
            st.warning("No trials match the selected Funder Type or durations missing.")
            st.stop()

        # Group by Phase and compute averages
        agg = (
            df_time.groupby("Phases")
            .agg(
                avg_overall_months=("Duration Overall (Months)", "mean"),
                avg_primary_months=("Duration Primary (Months)", "mean"),
            )
            .reset_index()
        )

        # Attempt to sort phases numerically if they follow "Phase 1", "Phase 2", etc.
        try:
            agg["phase_num"] = agg["Phases"].str.extract(r"(\d+)").astype(int)
            agg = agg.sort_values("phase_num")
        except Exception:
            agg = agg.sort_values("Phases")

        # Convert to display‐friendly labels
        agg["Overall Label"] = agg["avg_overall_months"].apply(months_to_yr_mo_string)
        agg["Primary Label"] = agg["avg_primary_months"].apply(months_to_yr_mo_string)

        # Two side‐by‐side columns
        col1, col2 = st.columns(2)

        # ----- Overall Duration Chart -----
        fig_overall = px.bar(
            agg,
            x="avg_overall_months",
            y="Phases",
            orientation="h",
            text="Overall Label",
            labels={"avg_overall_months": "Avg Duration (Months)", "Phases": "Phase"},
            title="📊 Avg Overall Duration by Phase",
        )
        fig_overall.update_traces(textposition="outside")
        fig_overall.update_layout(
            margin=dict(l=100, r=20, t=50, b=20),
            yaxis={"categoryorder": "array", "categoryarray": agg["Phases"]},
        )

        # ----- Primary Completion Duration Chart -----
        fig_primary = px.bar(
            agg,
            x="avg_primary_months",
            y="Phases",
            orientation="h",
            text="Primary Label",
            labels={"avg_primary_months": "Avg Duration (Months)", "Phases": "Phase"},
            title="📈 Avg Primary Completion Duration by Phase",
        )
        fig_primary.update_traces(textposition="outside")
        fig_primary.update_layout(
            margin=dict(l=100, r=20, t=50, b=20),
            yaxis={"categoryorder": "array", "categoryarray": agg["Phases"]},
        )

        # Display charts side by side, each set to a fixed height so no scrolling is needed
        with col1:
            st.plotly_chart(fig_overall, use_container_width=True, height=500)
        with col2:
            st.plotly_chart(fig_primary, use_container_width=True, height=500)

        st.markdown(
            """
            - Bars show **average** duration (start→completion or start→primary) per Phase.  
            - Labels on each bar are in “X yr Y mo” (or “Y mo”) format.  
            - Use the **Funder Type** toggle (sidebar) to switch between Industry, Academic, or Both.  
            """
        )


if __name__ == "__main__":
    main()
