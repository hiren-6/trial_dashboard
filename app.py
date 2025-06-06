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


# ----------------------------
# 1) DATA LOADING FUNCTIONS
# ----------------------------
def load_dummy():
    """Load COVID.csv (dummy) from the same folder as app.py."""
    try:
        df = pd.read_csv("COVID.csv")
    except FileNotFoundError:
        st.error("❗ Could not find COVID.csv. Make sure app.py and COVID.csv are side by side.")
        return None
    return df


def load_uploaded(uploaded_file):
    """Load user‐uploaded CSV or Excel into a DataFrame."""
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


# ----------------------------
# 2) PREPROCESS FUNCTIONS
# ----------------------------
def preprocess_dates_and_durations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert date columns to datetime, then compute durations (in months)
    for overall and primary completion.
    """
    df = df.copy()
    for col in ["Start Date", "Primary Completion Date", "Completion Date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Duration in months = (days difference) / 30
    df["Duration Overall (Months)"] = (
        (df["Completion Date"] - df["Start Date"]).dt.days.astype(float) / 30.0
    )
    df["Duration Primary (Months)"] = (
        (df["Primary Completion Date"] - df["Start Date"]).dt.days.astype(float) / 30.0
    )
    return df


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


# ----------------------------
# 3) MAIN APP
# ----------------------------
def main():
    st.sidebar.title("Data & Navigation")

    # 1) Data Source Selection
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

    # 2) Required‐column check
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

    # 3) Preprocess dates & durations
    df_processed = preprocess_dates_and_durations(df)

    # 4) Page navigation
    page = st.sidebar.radio("2) Navigate to", ("Home", "Geography", "Timelines"))

    # -----------------
    # Home Page (unchanged)
    # -----------------
    if page == "Home":
        st.title("🏠 Clinical Trials Dashboard")
        st.markdown(
            """
            **Welcome!**  
            Explore clinical trials by location and timeline.

            **Instructions**  
            1. In the sidebar, pick **Use Dummy Data** (pre‐loaded) or **Upload CSV/Excel**.  
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

    # -------------------
    # Geography Page (unchanged except for minor formatting)
    # -------------------
    elif page == "Geography":
        st.title("🌍 Geography Dashboard (City‐Level Hotspots)")
        st.markdown("An interactive hex‐bin map showing trial hotspots by city. Filter by **Study Status**.")

        # Filter by Study Status
        statuses = sorted(df_processed["Study Status"].dropna().unique())
        selected_status = st.sidebar.multiselect("Filter: Study Status", statuses, default=statuses)

        df_geo = df_processed[df_processed["Study Status"].isin(selected_status)].copy()
        if df_geo.shape[0] == 0:
            st.warning("No trials match the selected Study Status. Adjust filter.")
            st.stop()

        with st.spinner("Geocoding cities…first run may take time (each city = 1s)…"):
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

        # Build the HexagonLayer
        mid_lat = city_df["lat"].mean() if not city_df["lat"].isna().all() else 0
        mid_lon = city_df["lon"].mean() if not city_df["lon"].isna().all() else 0

        hex_layer = pdk.Layer(
            "HexagonLayer",
            data=city_df,
            get_position=["lon", "lat"],
            radius=50000,  # 50 km
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
            - Taller/darker hexagons = more trials in that area.  
            - Hover on a hex to see count.  
            - If city‐level geocoding fails, we fall back to a country‐level choropleth.  
            """
        )

    # -------------------
    # Timelines Page (UPDATED to match your mock‐up exactly)
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
                    Study Timeline Dashboard
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

        # ——— Funder Type Toggle (as a horizontal radio) ———
        st.markdown(
            "<div style='display: flex; align-items: center; margin-bottom: 10px;'>"
            "<span style='font-weight: 600; margin-right: 10px; font-size: 16px;'>Industry</span>"
            "<div style='flex: 1;'>"
            f"{st.radio('', ['Industry', 'Academic'], index=0, horizontal=True)}"
            "</div>"
            "<span style='font-weight: 600; margin-left: 10px; font-size: 16px;'>Academic</span>"
            "</div>",
            unsafe_allow_html=True,
        )

        # Read which option is selected
        # (We gave the radio an empty label so that it renders inline; 
        #  the actual returned value is the selected string.)
        funder_selected = st.session_state.get(st.radio.__name__, "Industry")
        # Above: Because we inserted the radio inline, we pulled the result directly from session_state.
        # Alternatively, you could do:
        #   funder_selected = st.radio("Funder Type", ["Industry", "Academic"], horizontal=True)
        # but it will render slightly differently. The above approach more closely mimics a toggle.

        # Filter the data by Funder Type
        if funder_selected == "Industry":
            df_time = df_processed[df_processed["Funder Type"].str.contains("Industry", na=False)].copy()
        else:
            df_time = df_processed[df_processed["Funder Type"].str.contains("Academic", na=False)].copy()

        # Drop any invalid durations
        df_time = df_time[
            df_time["Duration Overall (Months)"].notna()
            & (df_time["Duration Overall (Months)"] >= 0)
            & df_time["Duration Primary (Months)"].notna()
            & (df_time["Duration Primary (Months)"] >= 0)
        ]

        if df_time.shape[0] == 0:
            st.warning("No trials match the selected funder type or durations are missing.")
            st.stop()

        # Compute average durations by Phase
        agg = (
            df_time.groupby("Phases")
            .agg(
                avg_overall_months=("Duration Overall (Months)", "mean"),
                avg_primary_months=("Duration Primary (Months)", "mean"),
            )
            .reset_index()
        )

        # Try to sort phases numerically if they follow “Phase 1”, “Phase 2”, etc.
        try:
            agg["phase_num"] = agg["Phases"].str.extract(r"(\d+)").astype(int)
            agg = agg.sort_values("phase_num", ascending=False)  # descending so Phase 3 is on top
        except Exception:
            agg = agg.sort_values("Phases", ascending=False)

        # Build labels: e.g. "36 months"
        agg["Primary Label"] = agg["avg_primary_months"].round(0).astype(int).astype(str) + " months"
        agg["Overall Label"] = agg["avg_overall_months"].round(0).astype(int).astype(str) + " months"

        # Determine a common x‐axis range across both charts
        max_primary = agg["avg_primary_months"].max()
        max_overall = agg["avg_overall_months"].max()
        common_x_max = max(max_primary, max_overall) * 1.1  # multiply by 1.1 for some headroom

        # Build two side‐by‐side columns so charts fit in one screen
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
            # Put labels outside
            fig_primary.update_traces(textposition="outside", marker_color="#0E4D64")
            # Hide x‐axis tick labels & grid lines
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
            fig_overall.update_yaxes(visible=False)  # hide y‐axis on the right chart
            fig_overall.update_layout(
                margin=dict(l=20, r=20, t=0, b=20),
                plot_bgcolor="white",
                paper_bgcolor="white",
                height=300,
            )
            st.plotly_chart(fig_overall, use_container_width=True)

        # —— Below Charts: Explanatory Bullets ——
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
