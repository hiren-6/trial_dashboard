import streamlit as st
import pandas as pd
import plotly.express as px
import pycountry

st.set_page_config(
    page_title="Clinical Trials Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_dummy():
    """Try to load COVID.csv from the same folder as app.py."""
    try:
        df = pd.read_csv("COVID.csv")
    except FileNotFoundError:
        st.error("❗ Could not find COVID.csv in this folder. Make sure app.py and COVID.csv are side by side.")
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


def preprocess_dates_and_durations(df):
    """Ensure date columns are datetimes, then compute durations (in months)."""
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


def extract_country_counts(loc_series):
    """
    Turn a Series of Location strings (e.g. "City, Country | City2, Country2")
    into a DataFrame of {"country", "iso_alpha3", "count"}.
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


def months_to_yr_mo_string(m):
    """Convert a float number of months to a string like '2 yr 3 mo' or '5 mo'."""
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


def main():
    st.sidebar.title("Data & Navigation")

    # ——— 1) Data source selection ———
    data_source = st.sidebar.radio("1) Data Source", ("Use Dummy Data", "Upload CSV/Excel"))

    if data_source == "Use Dummy Data":
        df = load_dummy()
        if df is None:
            st.stop()
    else:
        uploaded = st.sidebar.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])
        df = load_uploaded(uploaded)
        if df is None:
            st.info("Please upload a valid CSV or Excel to proceed.")
            st.stop()

    # ——— 2) Check required columns ———
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
        st.error(f"❗ The following required columns are missing from your data: {missing}")
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
            This dashboard lets you explore clinical trials geographically and over time.

            **Steps to get started:**  
            1. In the sidebar, choose **Use Dummy Data** (preloaded COVID trials) or **Upload CSV/Excel** matching the dummy format.  
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
            st.info("No dummy data available to download (COVID.csv not found).")

        st.subheader("Preview of Your Data (first 10 rows)")
        st.dataframe(df_processed.head(10), use_container_width=True)

    # ——— Geography page ———
    elif page == "Geography":
        st.title("🌍 Geography Dashboard")
        st.markdown("Visualize trial locations on a world map. Filter by **Study Status** in the sidebar.")

        # Filter by Study Status
        statuses = sorted(df_processed["Study Status"].dropna().unique())
        selected_status = st.sidebar.multiselect("Filter: Study Status", statuses, default=statuses)

        df_geo = df_processed[df_processed["Study Status"].isin(selected_status)].copy()
        if df_geo.shape[0] == 0:
            st.warning("No trials match the selected Study Status. Adjust the filter to see data.")
            st.stop()

        country_counts = extract_country_counts(df_geo["Location"])
        if country_counts.shape[0] == 0:
            st.warning("Could not parse any recognizable country names from `Location`.")
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

        st.markdown(
            """
            - Each trial’s `Location` field is split by “|”.  
            - We take the text after the last comma as the country, map it to ISO α-3, and count occurrences.  
            - Hover over any country to see its trial count.  
            """
        )

    # ——— Timelines page ———
    else:  # page == "Timelines"
        st.title("⏱ Timeline Dashboard")
        st.markdown("Compare average trial durations (overall vs. primary completion). Filter by **Funder Type**.")

        # Filter by Funder Type: Industry vs Academic vs Both
        funder_option = st.sidebar.radio("Funder Type", ("Both", "Industry Only", "Academic Only"))
        if funder_option == "Industry Only":
            df_time = df_processed[df_processed["Funder Type"].str.contains("Industry", na=False)].copy()
        elif funder_option == "Academic Only":
            df_time = df_processed[df_processed["Funder Type"].str.contains("Academic", na=False)].copy()
        else:
            df_time = df_processed.copy()

        # Drop rows with missing/negative durations
        df_time = df_time[
            df_time["Duration Overall (Months)"].notna()
            & (df_time["Duration Overall (Months)"] >= 0)
            & df_time["Duration Primary (Months)"].notna()
            & (df_time["Duration Primary (Months)"] >= 0)
        ]

        if df_time.shape[0] == 0:
            st.warning("No trials match the selected Funder Type or durations are missing/invalid.")
            st.stop()

        # Group by Phase and compute average durations
        agg = (
            df_time.groupby("Phases")
            .agg(
                avg_overall_months=("Duration Overall (Months)", "mean"),
                avg_primary_months=("Duration Primary (Months)", "mean"),
            )
            .reset_index()
        )

        # If Phases have numeric component like "Phase 1", extract number for sorting
        try:
            agg["phase_num"] = agg["Phases"].str.extract(r"(\d+)").astype(int)
            agg = agg.sort_values("phase_num")
        except Exception:
            agg = agg.sort_values("Phases")

        # Convert to display-friendly strings
        agg["Overall Text"] = agg["avg_overall_months"].apply(months_to_yr_mo_string)
        agg["Primary Text"] = agg["avg_primary_months"].apply(months_to_yr_mo_string)

        # Prepare two side-by-side columns for the charts
        col1, col2 = st.columns(2)

        # Overall Duration Bar Chart
        fig_overall = px.bar(
            agg,
            x="avg_overall_months",
            y="Phases",
            orientation="h",
            text="Overall Text",
            labels={"avg_overall_months": "Avg Duration (Months)", "Phases": "Phase"},
            title="📊 Avg Overall Duration by Phase",
        )
        fig_overall.update_traces(textposition="outside")
        fig_overall.update_layout(
            margin=dict(l=100, r=20, t=50, b=20),
            yaxis={"categoryorder": "array", "categoryarray": agg["Phases"]},
        )

        # Primary Completion Duration Bar Chart
        fig_primary = px.bar(
            agg,
            x="avg_primary_months",
            y="Phases",
            orientation="h",
            text="Primary Text",
            labels={"avg_primary_months": "Avg Duration (Months)", "Phases": "Phase"},
            title="📈 Avg Primary Completion Duration by Phase",
        )
        fig_primary.update_traces(textposition="outside")
        fig_primary.update_layout(
            margin=dict(l=100, r=20, t=50, b=20),
            yaxis={"categoryorder": "array", "categoryarray": agg["Phases"]},
        )

        # Display both charts side by side, fitting in the visible screen
        with col1:
            st.plotly_chart(fig_overall, use_container_width=True, height=500)
        with col2:
            st.plotly_chart(fig_primary, use_container_width=True, height=500)

        st.markdown(
            """
            - Bars show **average** duration (in months) for each Phase.  
            - Text labels on each bar display the duration as “X yr Y mo” (or “Y mo”).  
            - Use the **Funder Type** toggle (sidebar) to switch between Industry, Academic, or Both.  
            """
        )


if __name__ == "__main__":
    main()
