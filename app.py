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
    df_tl["Phases"] = df_tl.iloc[:, 16]
    df_tl["Funder"] = df_tl.iloc[:, 18]
    df_tl["Start"] = pd.to_datetime(df_tl.iloc[:, 22], errors="coerce")
    df_tl["Completion"] = pd.to_datetime(df_tl.iloc[:, 23], errors="coerce")
    df_tl["Primary"] = pd.to_datetime(df_tl.iloc[:, 24], errors="coerce")

    # 2) Compute durations in months (days ÷ 30)
    df_tl["DurOverall"] = (df_tl["Completion"] - df_tl["Start"]).dt.days.astype("float") / 30.0
    df_tl["DurPrimary"] = (df_tl["Primary"] - df_tl["Start"]).dt.days.astype("float") / 30.0

    # 3) Drop rows where durations are missing or negative
    df_tl = df_tl[
        df_tl["DurOverall"].notna()
        & (df_tl["DurOverall"] >= 0)
        & df_tl["DurPrimary"].notna()
        & (df_tl["DurPrimary"] >= 0)
    ]

    # Helper: group and compute means for a given funder filter
    def avg_by_funder(funder_keyword: str) -> pd.DataFrame:
        sub = df_tl[df_tl["Funder"].str.contains(funder_keyword, na=False)].copy()
        if sub.empty:
            return pd.DataFrame(columns=["Phases", "avg_primary_months", "avg_overall_months"])

        agg = (
            sub.groupby("Phases")
            .agg(
                avg_primary_months=("DurPrimary", "mean"),
                avg_overall_months=("DurOverall", "mean"),
            )
            .reset_index()
        )

        # Ensure we cover all phases present in the entire dataset
        all_phases = sorted(
            df_tl["Phases"].dropna().unique(),
            key=lambda x: int("".join(filter(str.isdigit, str(x))) or 0),
            reverse=False,  # will sort ascending, e.g. Phase 1, Phase 2, Phase 3
        )

        # Reindex so every phase appears (fill missing with 0)
        agg = agg.set_index("Phases").reindex(all_phases, fill_value=0).reset_index()
        return agg

    avg_academic = avg_by_funder("Academic")
    avg_industry = avg_by_funder("Industry")

    return avg_academic, avg_industry


# -------------------------------------------
# 3) MAIN APP
# -------------------------------------------
def main():
    st.sidebar.title("Data & Navigation")

    # 1) Data source selection
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

    # 2) Required-column check (for Geography)
    required_cols_geo = [
        "Study Status",
        "Location",
        "Phases",       # Column Q (17) by name
        "Funder Type",  # Column S (19) by name
    ]
    missing_geo = [c for c in required_cols_geo if c not in df.columns]
    if missing_geo:
        st.error(f"❗ Missing required columns (for Geography): {missing_geo}")
        st.stop()

    # 3) Precompute timeline averages (Academic vs. Industry)
    avg_academic, avg_industry = compute_timeline_averages(df)

    # 4) Page navigation
    page = st.sidebar.radio("2) Navigate to", ("Home", "Geography", "Timelines"))

    # -----------------
    # Home Page
    # -----------------
    if page == "Home":
        st.title("🏠 Clinical Trials Dashboard")
        st.markdown(
            """
            **Welcome!**  
            This dashboard lets you explore clinical trials by location and by study timeline.

            **Steps to get started:**  
            1. In the sidebar, choose **Use Dummy Data** (pre‐loaded) or **Upload CSV/Excel** matching the dummy format.  
            2. Click **Home**, **Geography**, or **Timelines** to navigate.  
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
        st.dataframe(df.head(10), use_container_width=True)

    # -------------------
    # Geography Page
    # -------------------
    elif page == "Geography":
        st.title("🌍 Trial Hotspot (City Wise)")
        st.markdown("An interactive hex‐bin map showing trial hotspots by city. Filter by **Study Status**.")

        # Sidebar filter: Study Status
        statuses = sorted(df["Study Status"].dropna().unique())
        selected_status = st.sidebar.multiselect("Filter: Study Status", statuses, default=statuses)

        df_geo = df[df["Study Status"].isin(selected_status)].copy()
        if df_geo.shape[0] == 0:
            st.warning("No trials match the selected Study Status. Adjust filter.")
            st.stop()

        with st.spinner("Geocoding cities…first run may take time (1 s per city)…"):
            city_df = build_city_hotspots(df_geo)

        if city_df.empty:
            st.warning("No city‐level coordinates could be extracted/geocoded.")
            st.info("Showing fallback country‐level choropleth instead.")
            country_counts = extract_country_counts(df_geo["Location"])
            if country_counts.empty:
                st.error("Country extraction failed; check your `Location` format.")
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
            fig.update_layout(margin=dict(l=0, r=0, t=50, b=0), coloraxis_colorbar=dict(title="Trials"))
            st.plotly_chart(fig, use_container_width=True)
            return

        # Compute map center
        mid_lat = city_df["lat"].mean() if not city_df["lat"].isna().all() else 0
        mid_lon = city_df["lon"].mean() if not city_df["lon"].isna().all() else 0

        # HexagonLayer (for hotspot bins)
        hex_layer = pdk.Layer(
            "HexagonLayer",
            data=city_df,
            get_position=["lon", "lat"],
            radius=50000,            # 50 km hex radius
            elevation_scale=50,
            elevation_range=[0, 3000],
            pickable=True,
            extruded=True,
            coverage=0.8,
        )

        # ScatterplotLayer (small red dots) so we can show city names on hover
        scatter_layer = pdk.Layer(
            "ScatterplotLayer",
            data=city_df,
            get_position=["lon", "lat"],
            get_radius=20000,          # 20 km circle
            get_fill_color=[220, 20, 60, 180],  # crimson with some transparency
            pickable=True,
        )

        view_state = pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=1.5, bearing=0, pitch=40)

        # Tooltip: {city_country} for dots, {elevationValue} for hexes
        tooltip = {
            "html": """
              <div>
                <b>City:</b> {city_country} <br/>
                <b>Trials in Bin:</b> {elevationValue}
              </div>
            """,
            "style": {"backgroundColor": "white", "color": "black", "font-size": "12px", "padding": "5px"},
        }

        deck = pdk.Deck(
            layers=[hex_layer, scatter_layer],
            initial_view_state=view_state,
            tooltip=tooltip,
            map_style="mapbox://styles/mapbox/light-v10",
        )

        st.pydeck_chart(deck)

        st.markdown(
            """
            - **Red dots:** Hover to see the exact city name (`City, Country`).  
            - **Hexagons:** Hover to see how many trials fall into that hex‐bin.  
            - If city‐level geocoding fails, a fallback country‐level choropleth is shown.  
            """
        )

    # -------------------
    # Timelines Page
    # -------------------
    else:  # page == "Timelines"
        # ——— Colored Top Banner ———
        st.markdown(
            """
            <div style="
                background-color: #0E4D64;
                padding: 10px;
                border-radius: 10px 10px 0 0;
                ">
                <h2 style="
                    color: white;
                    margin: 0;
                    text-align: center;
                    font-family: 'Helvetica Neue', Arial, sans-serif;
                ">
                    Trial Timeline Dashboard
                </h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ——— Subtitle ———
        st.markdown(
            """
            <p style="
                margin-top: 5px;
                margin-bottom: 20px;
                font-family: 'Helvetica Neue', Arial, sans-serif;
                font-size: 16px;
            ">
                Compare average trial durations (overall vs. primary).
            </p>
            """,
            unsafe_allow_html=True,
        )

        # ——— Funder Type Toggle (Radio) ———
        funder_selected = st.radio("Funder Type", ["Industry", "Academic"], index=0, horizontal=True)

        # Choose the correct precomputed DataFrame
        if funder_selected == "Industry":
            agg = avg_industry.copy()
        else:
            agg = avg_academic.copy()

        if agg.empty:
            st.warning("No data available for the selected Funder Type.")
            st.stop()

        # Build labels: e.g. "36 months"
        agg["Primary Label"] = agg["avg_primary_months"].round(0).astype(int).astype(str) + " months"
        agg["Overall Label"] = agg["avg_overall_months"].round(0).astype(int).astype(str) + " months"

        # Determine a common x‐axis max so both charts are comparable
        max_primary = agg["avg_primary_months"].max()
        max_overall = agg["avg_overall_months"].max()
        common_x_max = max(max_primary, max_overall) * 1.1  # 10% headroom

        # Sort so that Phase 3 is on top, then Phase 2, then Phase 1
        try:
            agg["phase_num"] = agg["Phases"].astype(str).str.extract(r"(\d+)").astype(int)
            agg = agg.sort_values("phase_num", ascending=False)
        except Exception:
            agg = agg.sort_values("Phases", ascending=False)

        # Two side‐by‐side columns
        col1, col2 = st.columns(2)

        # —— Left: Mean Study Primary Duration ——
        with col1:
            st.markdown(
                "<h3 style='font-family: Arial, sans-serif; color: #0E4D64;'>Mean Study Primary Duration</h3>",
                unsafe_allow_html=True,
            )
            fig_primary = px.bar(
                agg,
                x="avg_primary_months",
                y="Phases",
                orientation="h",
                text="Primary Label",
                labels={"avg_primary_months": "", "Phases": ""},
            )
            fig_primary.update_traces(textposition="outside", marker_color="#0E4D64")
            fig_primary.update_xaxes(
                range=[0, common_x_max],
                showticklabels=False,
                showgrid=False,
                zeroline=False,
            )
            fig_primary.update_yaxes(
                categoryorder="array",
                categoryarray=agg["Phases"].tolist(),
                showgrid=False,
            )
            fig_primary.update_layout(
                margin=dict(l=80, r=20, t=0, b=20),
                plot_bgcolor="white",
                paper_bgcolor="white",
                height=300,
            )
            st.plotly_chart(fig_primary, use_container_width=True)

        # —— Right: Mean Study Duration (Overall) ——
        with col2:
            st.markdown(
                "<h3 style='font-family: Arial, sans-serif; color: #0E4D64;'>Mean Study Duration</h3>",
                unsafe_allow_html=True,
            )
            fig_overall = px.bar(
                agg,
                x="avg_overall_months",
                y="Phases",
                orientation="h",
                text="Overall Label",
                labels={"avg_overall_months": "", "Phases": ""},
            )
            fig_overall.update_traces(textposition="outside", marker_color="#0E4D64")
            fig_overall.update_xaxes(
                range=[0, common_x_max],
                showticklabels=False,
                showgrid=False,
                zeroline=False,
            )
            fig_overall.update_yaxes(
                categoryorder="array",
                categoryarray=agg["Phases"].tolist(),
                showgrid=False,
            )
            fig_overall.update_layout(
                margin=dict(l=20, r=20, t=0, b=20),
                plot_bgcolor="white",
                paper_bgcolor="white",
                height=300,
            )
            st.plotly_chart(fig_overall, use_container_width=True)

        # —— Below charts: explanatory bullets ——
        st.markdown(
            """
            <ul style="font-family: Arial, sans-serif; font-size: 14px;">
                <li>Bars show <strong>average</strong> duration (start → primary or start → completion) per Phase.</li>
                <li>Use the <strong>Funder Type</strong> toggle above to switch between Industry or Academic.</li>
            </ul>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
