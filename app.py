from datetime import datetime
from io import StringIO

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="JMT Predictive Asset Health Dashboard",
    layout="wide",
    page_icon="📈"
)

PRIMARY_COLOR = "#1f4788"
ACCENT_COLOR = "#ff9f1c"
DANGER_COLOR = "#d7263d"
SUCCESS_COLOR = "#2a9d8f"


def init_session_state():
    defaults = {
        "current_page": "🏠 Overview",
        "transform_applied": False,
        "presentation_mode": False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def apply_global_styles():
    base_css = f"""
        <style>
            .main-title {{
                color: {PRIMARY_COLOR};
                font-weight: 700;
            }}
            .sub-title {{
                color: {PRIMARY_COLOR};
                font-size: 1.1rem;
                font-weight: 600;
            }}
            .narration-box {{
                background-color: rgba(31,71,136,0.07);
                border-left: 5px solid {PRIMARY_COLOR};
                padding: 1rem;
                border-radius: 0.5rem;
                margin-bottom: 1.5rem;
            }}
            .summary-bubble {{
                background-color: rgba(42,157,143,0.12);
                border-left: 5px solid {SUCCESS_COLOR};
                padding: 1rem;
                border-radius: 0.5rem;
                margin: 0.5rem 0;
            }}
            .risk-critical {{ background-color: rgba(215,38,61,0.12); }}
            .risk-high {{ background-color: rgba(255,159,28,0.12); }}
            .risk-medium {{ background-color: rgba(255,221,89,0.18); }}
            .risk-low {{ background-color: rgba(42,157,143,0.12); }}
        </style>
    """
    st.markdown(base_css, unsafe_allow_html=True)

    if st.session_state.get("presentation_mode"):
        st.markdown(
            """
            <style>
                div[data-testid="stMarkdown"] p {
                    font-size: 1.15rem;
                }
                div[data-testid="stMetricValue"] {
                    font-size: 2.2rem;
                }
                div[data-testid="stMetricLabel"] {
                    font-size: 1.1rem;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )


@st.cache_data
def load_raw_data():
    return pd.read_csv("data/raw_bridge_data.csv")


@st.cache_data
def load_clean_data():
    return pd.read_csv("data/clean_bridge_data.csv")


@st.cache_data
def load_predictions():
    df = pd.read_csv("data/prediction_results.csv")
    return df


def navigation_sidebar():
    with st.sidebar:
        st.markdown("""
            <h2 style='color: #1f4788; text-align:center;'>JMT</h2>
            <p style='text-align:center; color:#1f4788;'>Predictive Asset Health Dashboard</p>
        """, unsafe_allow_html=True)
        st.session_state.presentation_mode = st.toggle("Presentation Mode", value=st.session_state.get("presentation_mode", False))

        nav_choice = st.radio(
            "Navigate",
            ["🏠 Overview", "📊 Data Ingestion & Transformation", "🔬 Model Training & Analysis", "🎯 Predictions & Recommendations"],
            index=["🏠 Overview", "📊 Data Ingestion & Transformation", "🔬 Model Training & Analysis", "🎯 Predictions & Recommendations"].index(st.session_state.current_page),
        )
        st.session_state.current_page = nav_choice

        if st.button("Reset Demo", type="secondary"):
            for key in ["transform_applied", "current_page"]:
                st.session_state[key] = "🏠 Overview" if key == "current_page" else False
            st.experimental_rerun()


def home_page():
    st.markdown("""<h1 class='main-title'>JMT Predictive Asset Health Dashboard</h1>""", unsafe_allow_html=True)
    st.subheader("Interactive Demonstration of Predictive Maintenance Analytics")
    st.markdown(
        "This guided experience follows a Department of Transportation as it modernizes bridge maintenance across a 1,000 asset portfolio. "
        "Explore how raw data is transformed, how the predictive model learns, and how leaders act on the results.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class='summary-bubble'><h4>1. Data</h4><p>Clean and enrich data from field systems.</p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class='summary-bubble'><h4>2. Model</h4><p>Train predictive analytics on historical outcomes.</p></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class='summary-bubble'><h4>3. Action</h4><p>Prioritize high-risk bridges for proactive maintenance.</p></div>""", unsafe_allow_html=True)

    st.markdown("""<div class='narration-box'>Use the navigation menu or start the guided tutorial below.</div>""", unsafe_allow_html=True)

    if st.button("🚀 Start Tutorial"):
        st.session_state.current_page = "📊 Data Ingestion & Transformation"
        st.experimental_rerun()


def chapter_one():
    raw_df = load_raw_data()
    clean_df = load_clean_data()

    st.markdown("""<h1 class='main-title'>Chapter 1 · Data Ingestion & Transformation</h1>""", unsafe_allow_html=True)
    st.subheader("Step 1: Raw Asset Data from Client Systems")
    st.markdown("""<div class='narration-box'>This simulates data pulled from the DOT's maintenance database into our Data Lake. Notice the inconsistent materials, inspection scores, and dates that make automated analysis difficult.</div>""", unsafe_allow_html=True)

    st.dataframe(raw_df.head(10), use_container_width=True)

    st.markdown("### Data Quality Issues Identified")
    material_variants = raw_df["material"].str.lower().str.strip().nunique()
    date_examples = raw_df["last_inspection_date"].sample(5, random_state=1).tolist()
    score_counts = raw_df["last_inspection_score"].value_counts()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"Detected {material_variants} variations of material names for only three actual material types.")
    with col2:
        st.warning("Date formats observed: " + ", ".join(date_examples))
    with col3:
        st.success(
            "Inspection score distribution:\n" +
            "\n".join(f"{score}: {count}" for score, count in score_counts.items())
        )

    st.markdown("### Transformation Pipeline")
    if st.button("⚙️ Apply Data Transformation", key="transform_btn") or st.session_state.transform_applied:
        st.session_state.transform_applied = True
        with st.spinner("Standardizing and engineering features..."):
            st.sleep(1.2)
        col_raw, col_clean = st.columns(2)
        with col_raw:
            st.caption("Before: Raw data sample")
            st.dataframe(raw_df.head(10), use_container_width=True)
        with col_clean:
            st.caption("After: Clean, model-ready features")
            highlight_cols = clean_df.head(10).style.background_gradient(subset=["condition_score", "days_since_inspection"], cmap="Blues")
            st.dataframe(highlight_cols, use_container_width=True)

        st.markdown("### Transformation Summary")
        st.markdown(
            """
            - ✅ Standardized material descriptions into **steel**, **concrete**, and **wood**.
            - ✅ Converted text inspection scores into a numeric health scale (1-5).
            - ✅ Calculated **days since last inspection** regardless of original date format.
            - ✅ Categorized traffic volumes into Low / Medium / High utilization bands.
            - ✅ Extracted maintenance counts from 1,000 text history entries.
            """
        )
    else:
        st.caption("Click the button above to simulate the transformation workflow.")

    st.divider()
    col_prev, col_next = st.columns(2)
    with col_prev:
        if st.button("⬅️ Back to Overview"):
            st.session_state.current_page = "🏠 Overview"
            st.experimental_rerun()
    with col_next:
        if st.button("Next: See How the Model Works ➡️"):
            st.session_state.current_page = "🔬 Model Training & Analysis"
            st.experimental_rerun()


def survival_curve_data():
    ages = list(range(0, 81, 4))
    survival = [max(5, 100 - (age ** 1.35) * 0.3) for age in ages]
    return pd.DataFrame({"Bridge Age (years)": ages, "Survival Probability (%)": survival})


def feature_importance_data():
    return pd.DataFrame({
        "Factor": ["Age", "Material Type", "Traffic Volume", "Days Since Inspection"],
        "Importance": [45, 25, 20, 10],
    })


def model_exploration(age, condition, traffic_label):
    traffic_map = {"Low": 0.3, "Medium": 0.6, "High": 0.9}
    base_prob = 0.4 * (age / 80.0) + 0.25 * ((6 - condition) / 5.0)
    base_prob += 0.2 * traffic_map[traffic_label]
    base_prob = max(0, min(base_prob, 0.98))
    return round(base_prob * 100, 1)


def chapter_two():
    st.markdown("""<h1 class='main-title'>Chapter 2 · Model Training & Analysis</h1>""", unsafe_allow_html=True)
    st.subheader("What the model learns from 20 years of historical outcomes")
    st.markdown("""<div class='narration-box'>Here we highlight how the predictive model identifies patterns in aging, materials, inspection data, and usage. The visuals below represent training results from historical bridge performance.</div>""", unsafe_allow_html=True)

    st.markdown("### Survival Curve: Learned Failure Trends")
    survival_df = survival_curve_data()
    st.line_chart(survival_df.set_index("Bridge Age (years)"))

    st.markdown("### Feature Importance")
    importance_df = feature_importance_data().set_index("Factor")
    st.bar_chart(importance_df)
    st.info("The model identified age as the strongest predictor, but material durability and traffic pressure are critical contributors.")

    st.markdown("### Model Performance Snapshot")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model Accuracy", "87%")
    with col2:
        st.metric("Correct Predictions", "870 / 1000")
    with col3:
        st.metric("False Positive Rate", "8%")

    st.markdown("### Example Predictions")
    ex_col1, ex_col2, ex_col3 = st.columns(3)
    examples = [
        {"title": "Bridge A", "details": "45 years · steel · high traffic", "prob": "92% failure probability", "color": DANGER_COLOR},
        {"title": "Bridge B", "details": "18 years · concrete · medium traffic", "prob": "42% failure probability", "color": ACCENT_COLOR},
        {"title": "Bridge C", "details": "5 years · concrete · low traffic", "prob": "8% failure probability", "color": SUCCESS_COLOR},
    ]
    for col, example in zip([ex_col1, ex_col2, ex_col3], examples):
        with col:
            st.markdown(
                f"""
                <div style='border-left:6px solid {example['color']}; padding:1rem; border-radius:0.5rem; background-color:rgba(0,0,0,0.03);'>
                    <h4>{example['title']}</h4>
                    <p>{example['details']}</p>
                    <h3 style='color:{example['color']}'>{example['prob']}</h3>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### Explore: How do factors change failure risk?")
    col_age, col_condition, col_traffic = st.columns(3)
    with col_age:
        age = st.slider("Bridge age (years)", 0, 60, 30)
    with col_condition:
        condition = st.slider("Condition score (1=Critical, 5=Excellent)", 1, 5, 3)
    with col_traffic:
        traffic = st.select_slider("Traffic volume", options=["Low", "Medium", "High"], value="Medium")

    risk = model_exploration(age, condition, traffic)
    st.success(f"Estimated failure probability: {risk}%")
    st.caption("Move the sliders to simulate different bridge scenarios and see how the model responds.")

    st.divider()
    col_prev, col_next = st.columns(2)
    with col_prev:
        if st.button("⬅️ Back to Data Transformation"):
            st.session_state.current_page = "📊 Data Ingestion & Transformation"
            st.experimental_rerun()
    with col_next:
        if st.button("Next: View Actionable Predictions ➡️"):
            st.session_state.current_page = "🎯 Predictions & Recommendations"
            st.experimental_rerun()


def style_predictions(df: pd.DataFrame):
    def color_row(row):
        if row["failure_probability"] > 0.8:
            return ["background-color: rgba(215,38,61,0.15);"] * len(row)
        if row["failure_probability"] > 0.6:
            return ["background-color: rgba(255,159,28,0.18);"] * len(row)
        if row["failure_probability"] > 0.4:
            return ["background-color: rgba(255,221,89,0.25);"] * len(row)
        return ["background-color: rgba(42,157,143,0.15);"] * len(row)

    styled = df.style.apply(color_row, axis=1)
    styled = styled.format({"failure_probability": "{:.0%}", "predicted_months_to_failure": "{} months"})
    return styled


def download_csv(df: pd.DataFrame) -> bytes:
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue().encode("utf-8")


def chapter_three():
    predictions = load_predictions()
    clean_df = load_clean_data()
    raw_df = load_raw_data()

    merged = predictions.merge(clean_df, on="bridge_id")
    merged = merged.merge(raw_df[["bridge_id", "bridge_name", "traffic_volume_daily", "material"]], on="bridge_id")

    st.markdown("""<h1 class='main-title'>Chapter 3 · Predictions & Recommendations</h1>""", unsafe_allow_html=True)
    st.subheader("Asset Risk Prioritization Dashboard")
    st.caption("1,000 bridges ranked by predicted failure probability")

    risk_counts = predictions["risk_category"].value_counts()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Critical Risk", int(risk_counts.get("Critical", 0)))
    col2.metric("High Risk", int(risk_counts.get("High", 0)))
    col3.metric("Medium Risk", int(risk_counts.get("Medium", 0)))
    col4.metric("Low Risk", int(risk_counts.get("Low", 0)))

    st.markdown("---")
    st.markdown("### Filter Portfolio")
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
    with filter_col1:
        material_filter = st.selectbox("Material", ["All", "Steel", "Concrete", "Wood"], index=0)
    with filter_col2:
        risk_filter = st.multiselect("Risk category", ["Critical", "High", "Medium", "Low"], default=["Critical", "High", "Medium", "Low"])
    with filter_col3:
        age_range = st.slider("Age range (years)", 0, 80, (0, 80))
    with filter_col4:
        traffic_filter = st.multiselect("Traffic volume", ["Low", "Medium", "High"], default=["Low", "Medium", "High"])

    filtered = merged.copy()
    if material_filter != "All":
        filtered = filtered[filtered["material_standardized"] == material_filter.lower()]
    if risk_filter:
        filtered = filtered[filtered["risk_category"].isin(risk_filter)]
    filtered = filtered[(filtered["age_years"].between(age_range[0], age_range[1]))]
    if traffic_filter:
        filtered = filtered[filtered["traffic_volume_category"].isin(traffic_filter)]

    st.markdown(f"**Showing {len(filtered)} of {len(merged)} bridges**")

    display_cols = [
        "bridge_id", "bridge_name", "material_standardized", "age_years", "traffic_volume_daily", "failure_probability",
        "predicted_months_to_failure", "risk_category", "recommended_action"
    ]

    styled_table = style_predictions(filtered[display_cols].sort_values("failure_probability", ascending=False))
    st.dataframe(styled_table, use_container_width=True)

    st.download_button(
        "💾 Export Filtered Results for Field Teams",
        data=download_csv(filtered[display_cols]),
        file_name=f"bridge_risk_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )

    st.markdown("### Top 10 Critical Bridges")
    critical_top = filtered[filtered["risk_category"].isin(["Critical", "High"])].sort_values("failure_probability", ascending=False).head(10)
    if critical_top.empty:
        st.info("Adjust filters to view critical bridges.")
    else:
        cards = st.columns(2)
        for idx, (_, row) in enumerate(critical_top.iterrows()):
            col = cards[idx % 2]
            with col:
                st.markdown(
                    f"""
                    <div style='border-left:6px solid {DANGER_COLOR if row['risk_category']=="Critical" else ACCENT_COLOR}; padding:1rem; margin:0.5rem 0; border-radius:0.5rem; background-color:rgba(215,38,61,0.08);'>
                        <h4>{row['bridge_name']} ({row['bridge_id']})</h4>
                        <p><strong>Failure probability:</strong> {row['failure_probability']*100:.1f}%</p>
                        <p><strong>Recommended action:</strong> {row['recommended_action']}</p>
                        <p><strong>Estimated cost avoidance:</strong> ${(250000 - 60000):,}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("### Business Impact Summary")
    st.success(
        """
        By proactively addressing the 87 critical bridges, the DOT can:

        * Realize an estimated **$12.4M** in avoided emergency repairs.
        * Protect **145,000 daily users** from unexpected closures.
        * Prevent **23 emergency shutdowns**, keeping supply chains and communities moving.
        """
    )

    st.divider()
    col_prev, col_next = st.columns(2)
    with col_prev:
        if st.button("⬅️ Back to Model Analysis"):
            st.session_state.current_page = "🔬 Model Training & Analysis"
            st.experimental_rerun()
    with col_next:
        if st.button("Start Over 🔁"):
            st.session_state.current_page = "🏠 Overview"
            st.experimental_rerun()


def main():
    init_session_state()
    apply_global_styles()
    navigation_sidebar()

    page = st.session_state.current_page
    if page == "🏠 Overview":
        home_page()
    elif page == "📊 Data Ingestion & Transformation":
        chapter_one()
    elif page == "🔬 Model Training & Analysis":
        chapter_two()
    else:
        chapter_three()


if __name__ == "__main__":
    main()
