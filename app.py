# app.py

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
    """Ensure the three date columns are datetimes, then compute durations (in months)."""
    df = df.copy()
    for col in ["Start Date", "Primary Completion Date", "Completion Date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Compute durations (days ÷ 30 for an approximate month‐count).
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

    if len(all_countries) == 0:
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
        st.markdown("Compare trial durations (overall vs. primary completion). Filter by **Funder Type**.")

        # Filter by Funder Type
        funders = sorted(df_processed["Funder Type"].dropna().unique())
        selected_funders = st.sidebar.multiselect("Filter: Funder Type", funders, default=funders)

        df_time = df_processed[df_processed["Funder Type"].isin(selected_funders)].copy()
        df_time = df_time[
            df_time["Duration Overall (Months)"].notna()
            & (df_time["Duration Overall (Months)"] >= 0)
            & df_time["Duration Primary (Months)"].notna()
            & (df_time["Duration Primary (Months)"] >= 0)
        ]

        if df_time.shape[0] == 0:
            st.warning("No trials match the selected Funder Type or durations are missing/invalid.")
            st.stop()

        # Create a short label for each trial
        df_time["Trial Label"] = (
            df_time["NCT Number"].astype(str) + ": " + df_time["Study Title"].str.slice(0, 50) + "…"
        )
        df_time = df_time.sort_values("Start Date")

        # Overall Duration Chart
        fig_overall = px.bar(
            df_time,
            x="Duration Overall (Months)",
            y="Trial Label",
            orientation="h",
            color="Phases",
            hover_data={
                "Start Date": True,
                "Completion Date": True,
                "Duration Overall (Months)": ":.1f",
                "Phases": True,
            },
            height=600,
            title="📊 Overall Trial Duration (Start → Completion) in Months",
        )
        fig_overall.update_layout(
            yaxis={"categoryorder": "array", "categoryarray": df_time["Trial Label"]},
            margin=dict(l=300, r=20, t=50, b=20),
            legend_title_text="Phase",
            xaxis_title="Duration (Months)",
            yaxis_title="Trial",
        )

        # Primary Completion Duration Chart
        fig_primary = px.bar(
            df_time,
            x="Duration Primary (Months)",
            y="Trial Label",
            orientation="h",
            color="Phases",
            hover_data={
                "Start Date": True,
                "Primary Completion Date": True,
                "Duration Primary (Months)": ":.1f",
                "Phases": True,
            },
            height=600,
            title="📈 Primary Completion Duration (Start → Primary Completion) in Months",
        )
        fig_primary.update_layout(
            yaxis={"categoryorder": "array", "categoryarray": df_time["Trial Label"]},
            margin=dict(l=300, r=20, t=50, b=20),
            legend_title_text="Phase",
            xaxis_title="Duration (Months)",
            yaxis_title="Trial",
        )

        st.plotly_chart(fig_overall, use_container_width=True)
        st.plotly_chart(fig_primary, use_container_width=True)

        st.markdown(
            """
            - **Duration Overall** = `Completion Date – Start Date` (converted to months).  
            - **Duration Primary** = `Primary Completion Date – Start Date` (converted to months).  
            - Bars are colored by `Phases`.  
            - Hover on a bar for exact dates and duration.  
            """
        )


if __name__ == "__main__":
    main()
