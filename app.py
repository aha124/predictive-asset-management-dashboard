from datetime import datetime
from io import StringIO
import time

import pandas as pd
import streamlit as st

from advanced_model_data import (
    calculate_permutation_importance,
    create_model_comparison_data,
    get_detailed_importance,
)
from advanced_model_visuals import (
    create_comparison_chart,
    create_confusion_matrix,
    create_interaction_heatmap,
    generate_shap_explanation,
    show_forest_voting_visual,
)

st.set_page_config(
    page_title="JMT Predictive Asset Health Dashboard",
    layout="wide"
)

PRIMARY_COLOR = "#1f4788"
ACCENT_COLOR = "#ff9f1c"
DANGER_COLOR = "#d7263d"
SUCCESS_COLOR = "#2a9d8f"


def init_session_state():
    defaults = {
        "current_page": "Overview",
        "transform_applied": False,
        "presentation_mode": False,
        "technical_mode": False,
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
    try:
        return pd.read_csv("data/raw_bridge_data.csv")
    except FileNotFoundError:
        st.error("Raw data file not found. Please ensure data/raw_bridge_data.csv exists.")
        st.stop()
    except Exception as exc:
        st.error(f"Error loading raw data: {exc}")
        st.stop()


@st.cache_data
def load_clean_data():
    try:
        return pd.read_csv("data/clean_bridge_data.csv")
    except FileNotFoundError:
        st.error("Clean data file not found. Please ensure data/clean_bridge_data.csv exists.")
        st.stop()
    except Exception as exc:
        st.error(f"Error loading clean data: {exc}")
        st.stop()


@st.cache_data
def load_predictions():
    try:
        df = pd.read_csv("data/prediction_results.csv")
        return df
    except FileNotFoundError:
        st.error("Prediction results file not found. Please ensure data/prediction_results.csv exists.")
        st.stop()
    except Exception as exc:
        st.error(f"Error loading prediction results: {exc}")
        st.stop()


def navigation_sidebar():
    with st.sidebar:
        st.markdown("""
            <h2 style='color: #1f4788; text-align:center;'>JMT</h2>
            <p style='text-align:center; color:#1f4788;'>Predictive Asset Health Dashboard</p>
        """, unsafe_allow_html=True)
        st.session_state.presentation_mode = st.toggle(
            "Presentation Mode", value=st.session_state.get("presentation_mode", False)
        )

        pages = [
            "Overview",
            "Data Ingestion & Transformation",
            "Model Training & Analysis",
            "Advanced Model Deep Dive",
            "Predictions & Recommendations",
        ]

        nav_choice = st.radio(
            "Navigate",
            pages,
            index=pages.index(st.session_state.current_page),
        )
        st.session_state.current_page = nav_choice

        if st.session_state.current_page == "Advanced Model Deep Dive":
            st.session_state.technical_mode = st.toggle(
                "Technical Mode",
                value=st.session_state.get("technical_mode", False),
                help="Show code examples and mathematical details",
            )
        else:
            st.session_state.technical_mode = False

        if st.button("Reset Demo", type="secondary"):
            for key in ["transform_applied", "current_page", "technical_mode"]:
                if key == "current_page":
                    st.session_state[key] = "Overview"
                else:
                    st.session_state[key] = False
            st.rerun()


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

    if st.button("Start Tutorial"):
        st.session_state.current_page = "Data Ingestion & Transformation"
        st.rerun()


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
    if st.button("Apply Data Transformation", key="transform_btn") or st.session_state.transform_applied:
        st.session_state.transform_applied = True
        with st.spinner("Standardizing and engineering features..."):
            time.sleep(1.2)
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
            - Standardized material descriptions into **steel**, **concrete**, and **wood**.
            - Converted text inspection scores into a numeric health scale (1-5).
            - Calculated **days since last inspection** regardless of original date format.
            - Categorized traffic volumes into Low / Medium / High utilization bands.
            - Extracted maintenance counts from 1,000 text history entries.
            """
        )
    else:
        st.caption("Click the button above to simulate the transformation workflow.")

    st.divider()
    col_prev, col_next = st.columns(2)
    with col_prev:
        if st.button("Back to Overview"):
            st.session_state.current_page = "Overview"
            st.rerun()
    with col_next:
        if st.button("Next: See How the Model Works"):
            st.session_state.current_page = "Model Training & Analysis"
            st.rerun()


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
    st.line_chart(
        survival_df.set_index("Bridge Age (years)"),
        use_container_width=True,
    )
    st.caption("As bridges age, the probability of structural integrity decreases non-linearly")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### Feature Importance")
    with col2:
        st.markdown("")
        with st.popover("What is this?"):
            st.write("Feature importance shows which factors the model considers most predictive of bridge failure.")
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
        if st.button("Back to Data Transformation"):
            st.session_state.current_page = "Data Ingestion & Transformation"
            st.rerun()
    with col_next:
        if st.button("Next: Advanced Model Deep Dive"):
            st.session_state.current_page = "Advanced Model Deep Dive"
            st.rerun()


def chapter_two_b():
    st.markdown("""<h1 class='main-title'>Chapter 2B · Advanced Model Deep Dive</h1>""", unsafe_allow_html=True)
    st.subheader("Random Survival Forest: Capturing Complex Interactions")
    st.markdown(
        """<div class='narration-box'>
    The model in Chapter 2 used a simplified linear approach for demonstration.
    In production, we use a Random Survival Forest (RSF) – an ensemble technique
    that uncovers non-linear patterns and feature interactions that linear models miss.
    </div>""",
        unsafe_allow_html=True,
    )

    comparison_df = create_model_comparison_data()

    st.markdown("### Model Comparison: Simple vs. Advanced")
    comparison_chart = create_comparison_chart(comparison_df)
    st.plotly_chart(comparison_chart, use_container_width=True)

    st.dataframe(
        comparison_df[
            [
                "bridge_id",
                "bridge_name",
                "failure_probability",
                "rsf_failure_probability",
                "historical_outcome",
                "best_match",
            ]
        ]
        .rename(
            columns={
                "failure_probability": "Simple Model",
                "rsf_failure_probability": "Random Survival Forest",
                "historical_outcome": "Historical Outcome",
                "best_match": "Closer to Outcome",
            }
        )
        .assign(
            **{
                "Simple Model": lambda df: (df["Simple Model"] * 100).round(1),
                "Random Survival Forest": lambda df: (df["Random Survival Forest"] * 100).round(1),
            }
        ),
        use_container_width=True,
    )

    st.markdown(
        """<div class='narration-box'>
    Bridges like <strong>BR-0543</strong> show why we use RSF. The simplified model
    predicted a modest 35% risk, while RSF flagged 72% because it recognized the
    accelerating maintenance pattern combined with heavy traffic. That non-linear
    interaction matched the historical failure.
    </div>""",
        unsafe_allow_html=True,
    )

    st.divider()
    technical = st.session_state.get("technical_mode", False)

    if not technical:
        st.markdown(
            """
        ### How It Works (Conceptually)

        Imagine you are trying to predict when a bridge might fail. You could:

        1. **Ask one expert** (Simple Model): they look at age and say "probably 3 years".
        2. **Ask 100 experts** (Random Forest): each focuses on different combinations of factors.

        - Expert 1 focuses on Age + Material + Maintenance history.
        - Expert 2 focuses on Traffic + Condition + Inspection gaps.
        - Expert 3 focuses on Age + Traffic + Material.
        - ...and 97 more perspectives.

        The Random Survival Forest averages all 100 opinions. This captures patterns that no
        single expert could see alone.
        """
        )

        forest_fig = show_forest_voting_visual()
        st.plotly_chart(forest_fig, use_container_width=True)
    else:
        st.markdown(
            """
        ### Random Survival Forest: Technical Overview

        RSF extends Random Forests to survival data with censored observations and time-to-event outcomes.

        **Algorithm:**
        1. Bootstrap sample the training data *B* times (typically 100–500).
        2. Grow a survival tree for each bootstrap sample:
           - Randomly select *mtry* features at each node (≈√p).
           - Split on the feature that maximizes the log-rank statistic.
           - Grow trees to full depth without pruning.
        3. Aggregate cumulative hazard predictions across all trees.

        **Key Differences from Classification RF:**
        - Splitting criterion uses the log-rank test instead of Gini or entropy.
        - Predictions return survival or hazard functions instead of class probabilities.
        - Out-of-bag samples estimate concordance without a separate validation fold.
        """
        )

        with st.expander("Python Implementation"):
            st.code(
                """
from sksurv.ensemble import RandomSurvivalForest
import numpy as np

# Prepare survival data structure
y = np.array(
    [(event, duration) for event, duration in zip(events, durations)],
    dtype=[('event', bool), ('duration', float)]
)

# Train Random Survival Forest
rsf = RandomSurvivalForest(
    n_estimators=100,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features="sqrt",
    random_state=42,
)

rsf.fit(X_train, y_train)

# Predict risk scores (higher = higher risk)
risk_scores = rsf.predict(X_test)

# Predict survival function
survival_functions = rsf.predict_survival_function(X_test)
                """,
                language="python",
            )

    st.divider()

    st.markdown("### Discovered Feature Interactions")
    col1, col2 = st.columns(2)
    with col1:
        feature_x = st.selectbox(
            "Feature X",
            ["Age", "Condition Score", "Traffic Volume", "Maintenance Count"],
        )
    with col2:
        feature_y = st.selectbox(
            "Feature Y",
            ["Condition Score", "Traffic Volume", "Days Since Inspection", "Material"],
        )

    interaction_fig = create_interaction_heatmap(feature_x, feature_y)
    st.plotly_chart(interaction_fig, use_container_width=True)

    st.markdown(
        """<div class='narration-box'>
    This heatmap shows how failure probability changes when these two factors combine.
    Notice how the relationship isn't linear – some combinations create exponentially
    higher risk than others.
    </div>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    - Age 40 + Condition 4 (Good) → **35%** risk.
    - Age 40 + Condition 2 (Poor) → **72%** risk.
    - Age 60 + Condition 2 (Poor) → **94%** risk – interaction effects amplify risk beyond additive expectations.
    """
    )

    st.divider()

    st.markdown("### Model Development Timeline")
    timeline_steps = [
        {
            "phase": "Data Preparation",
            "duration": "2 weeks",
            "details": "Clean historical data, engineer features, handle missing values.",
        },
        {
            "phase": "Initial Training",
            "duration": "3 days",
            "details": "Train baseline model on 80% of data and validate on the remainder.",
        },
        {
            "phase": "Hyperparameter Tuning",
            "duration": "1 week",
            "details": "Optimize number of trees, minimum samples, and feature sampling strategy.",
        },
        {
            "phase": "Cross-Validation",
            "duration": "2 days",
            "details": "5-fold cross-validation to ensure generalization across the portfolio.",
        },
        {
            "phase": "Production Deployment",
            "duration": "1 week",
            "details": "Integration testing, monitoring, and runbook documentation.",
        },
    ]

    for step in timeline_steps:
        with st.expander(f"**{step['phase']}** – {step['duration']}"):
            st.write(step["details"])
            if technical and step["phase"] == "Cross-Validation":
                st.code(
                    """
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for train_idx, val_idx in kf.split(X):
    X_train_fold = X[train_idx]
    y_train_fold = y[train_idx]
    X_val_fold = X[val_idx]
    y_val_fold = y[val_idx]

    rsf.fit(X_train_fold, y_train_fold)
    score = rsf.score(X_val_fold, y_val_fold)
    cv_scores.append(score)

print(f"CV C-index: {np.mean(cv_scores):.3f} (+/- {np.std(cv_scores):.3f})")
                    """,
                    language="python",
                )

    st.divider()

    st.markdown("### Feature Importance Analysis")
    tab1, tab2, tab3 = st.tabs([
        "Standard Importance",
        "Permutation Importance",
        "SHAP Values",
    ])

    with tab1:
        st.markdown("**Standard Feature Importance** (based on split frequency)")
        importance_df = get_detailed_importance()
        st.bar_chart(importance_df.set_index("Feature"))

    with tab2:
        st.markdown("**Permutation Importance** (performance drop when shuffled)")
        st.markdown(
            """<div class='narration-box'>
        We measure importance by randomly shuffling each feature and observing
        the impact on model accuracy. Larger drops mean the feature was crucial
        for predictions.
        </div>""",
            unsafe_allow_html=True,
        )

        perm_importance_df = calculate_permutation_importance()
        st.bar_chart(perm_importance_df.set_index("Feature"))

        if technical:
            with st.expander("How Permutation Importance Works"):
                st.markdown(
                    """
                1. Calculate baseline model accuracy on the validation set.
                2. For each feature:
                   - Shuffle the feature values.
                   - Recalculate accuracy.
                   - Importance = baseline – shuffled accuracy.
                3. Features with larger drops are more influential.
                """
                )

    with tab3:
        st.markdown("**SHAP Values** (contribution to individual predictions)")
        bridge_to_explain = st.selectbox(
            "Select a bridge to analyze:",
            [
                "BR-0012 (Critical Risk)",
                "BR-0236 (High Risk)",
                "BR-0669 (Medium Risk)",
            ],
        )
        shap_fig = generate_shap_explanation(bridge_to_explain)
        st.plotly_chart(shap_fig, use_container_width=True)
        st.markdown(
            """<div class='narration-box'>
        This waterfall view shows how each feature pushed the prediction up (red)
        or down (blue) for the selected bridge.
        </div>""",
            unsafe_allow_html=True,
        )

    st.divider()

    st.markdown("### Model Performance Metrics")
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:
        st.metric(
            "C-Index (Concordance)",
            "0.832",
            help="Measures ranking accuracy for survival models (>0.8 is excellent).",
        )

    with metric_col2:
        st.metric(
            "Recall (Sensitivity)",
            "91.3%",
            help="Percentage of actual failures correctly identified.",
        )

    with metric_col3:
        st.metric(
            "Precision",
            "68.7%",
            help="Of bridges flagged as high-risk, the percentage that actually failed.",
        )

    with metric_col4:
        st.metric(
            "False Negative Rate",
            "2.8%",
            delta="-1.2%",
            delta_color="inverse",
            help="Critical failures missed by the model (lower is better).",
        )

    st.markdown("### Confusion Matrix (12-month failure prediction)")
    confusion_fig = create_confusion_matrix()
    st.plotly_chart(confusion_fig, use_container_width=True)

    st.markdown("### Time-Dependent Performance")
    st.markdown("Model accuracy at different time horizons:")
    time_horizon_data = {
        "Months Ahead": [3, 6, 12, 18, 24],
        "AUC-ROC": [0.89, 0.85, 0.82, 0.79, 0.75],
    }
    st.line_chart(pd.DataFrame(time_horizon_data).set_index("Months Ahead"))
    st.markdown(
        """<div class='narration-box'>
    Accuracy naturally decreases for longer time horizons because distant events
    carry more uncertainty than near-term failures.
    </div>""",
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("### Real-World Validation: Case Studies")
    case_study = st.radio(
        "Select case study:",
        [
            "True Positive: Model predicted failure, bridge failed",
            "True Negative: Model predicted safe, bridge remained safe",
            "False Positive: Model predicted failure, bridge didn't fail",
            "False Negative: Model missed failure (rare)",
        ],
    )

    if "True Positive" in case_study:
        st.markdown(
            """
        **Bridge BR-0168 – Success Story**

        - **Model Prediction (Mar 2023):** 85% failure probability within 6 months.
        - **Risk Category:** Critical.
        - **Outcome:** Severe cracking discovered during emergency inspection (May 2023).
        - **Action Taken:** Emergency repair ($75K) and 3-day closure.
        - **Cost Avoided:** $280K in emergency response and two-week shutdown.

        **What the model saw:**
        - Age: 52 years (high risk).
        - Material: Concrete (aging infrastructure).
        - Condition Score: 1 (critical).
        - Maintenance history: 5 repairs in 5 years (accelerating deterioration).
        - Traffic: 18,000 daily vehicles (high stress).

        *Engineer insight:* "We had this bridge on our watch list, but the quantified
        risk score pushed it above 12 other concerning bridges. It was the right call."
        """
        )

        bridge_features = pd.DataFrame(
            {
                "Feature": [
                    "Age",
                    "Condition",
                    "Traffic",
                    "Maintenance",
                    "Days Since Inspection",
                ],
                "Value": [52, 1, "High (18K)", "5 repairs", 145],
                "Risk Contribution": ["+35%", "+28%", "+18%", "+15%", "+4%"],
            }
        )
        st.dataframe(bridge_features, use_container_width=True)

    elif "True Negative" in case_study:
        st.markdown(
            """
        **Bridge BR-0421 – Reliability Confirmed**

        - **Model Prediction (Jan 2023):** 12% failure probability.
        - **Risk Category:** Low.
        - **Outcome:** Passed two subsequent inspections with no structural concerns.
        - **Action Taken:** Continued standard monitoring cadence.

        **Why it mattered:**
        - Focused limited capital on higher-risk assets.
        - Prevented unnecessary preventative maintenance expenditure.
        - Built trust with district engineers in model recommendations.
        """
        )

        st.info(
            "Model explanation: Age 18 years, concrete deck, excellent inspection history, and low traffic combine for resilient performance."
        )

    elif "False Positive" in case_study:
        st.markdown(
            """
        **Bridge BR-0274 – Conservative Alert**

        - **Model Prediction (Jul 2022):** 71% failure probability.
        - **Risk Category:** High.
        - **Outcome:** Follow-up inspection found only minor joint wear.
        - **Action Taken:** Preventative resurfacing ($25K) instead of full rehab.

        **What we learned:**
        - RSF reacted to a spike in traffic + prior corrosion reports.
        - Field teams appreciated the caution – the bridge remains on quarterly monitoring.
        - False positives are acceptable when the cost of failure is extreme.
        """
        )

        st.warning(
            "We adjusted maintenance notes ingestion to distinguish cosmetic repairs from structural fixes, reducing future false positives by 8%."
        )

    else:
        st.markdown(
            """
        **Bridge BR-0543 – Model Miss (Learning Opportunity)**

        - **Model Prediction:** 25% failure probability (medium risk).
        - **Outcome:** Unexpected bearing failure (Aug 2023).
        - **Root Cause:** Manufacturing defect in bearing assembly (not visible in inspection data).

        **Model improvement:**
        - Added "bearing age" and manufacturer recall data as new features.
        - Increased monitoring for supplier-specific issues.
        - Demonstrates the importance of continuous feature engineering.
        """
        )

        st.error(
            "No model is perfect – rare misses inform the next iteration of our data pipeline."
        )

    st.divider()

    st.markdown("### Model Comparison: RSF vs. Simple Rules")
    comparison_approaches = pd.DataFrame(
        {
            "Approach": [
                "Age-based rules ( >40 years = high risk )",
                "Condition score only (score ≤2 = high risk)",
                "Linear regression (weighted sum)",
                "Random Survival Forest (production model)",
            ],
            "Accuracy": [58, 62, 73, 87],
            "False Negatives": [18, 15, 8, 3],
            "Advantages": [
                "Simple, easy to explain",
                "Captures deterioration",
                "Considers multiple factors",
                "Captures complex interactions",
            ],
            "Disadvantages": [
                "Misses well-maintained older bridges",
                "Ignores traffic and usage",
                "Assumes linear relationships",
                "Requires ML expertise",
            ],
        }
    )
    st.dataframe(comparison_approaches, use_container_width=True)
    st.markdown(
        """<div class='narration-box'>
    Simple rules work for roughly 60–70% of bridges, but the most critical 30%
    exhibit complex risk profiles that require sophisticated modeling.
    </div>""",
        unsafe_allow_html=True,
    )

    st.divider()

    col_prev, col_skip, col_next = st.columns(3)
    with col_prev:
        if st.button("Back to Model Overview"):
            st.session_state.current_page = "Model Training & Analysis"
            st.rerun()
    with col_skip:
        if st.button("Skip to Predictions"):
            st.session_state.current_page = "Predictions & Recommendations"
            st.rerun()
    with col_next:
        if st.button("Next: View Predictions"):
            st.session_state.current_page = "Predictions & Recommendations"
            st.rerun()

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
        "Export Filtered Results for Field Teams",
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
        if st.button("Back to Advanced Deep Dive"):
            st.session_state.current_page = "Advanced Model Deep Dive"
            st.rerun()
    with col_next:
        if st.button("Start Over"):
            st.session_state.current_page = "Overview"
            st.rerun()


def main():
    init_session_state()
    apply_global_styles()
    navigation_sidebar()

    page = st.session_state.current_page
    if page == "Overview":
        home_page()
    elif page == "Data Ingestion & Transformation":
        chapter_one()
    elif page == "Model Training & Analysis":
        chapter_two()
    elif page == "Advanced Model Deep Dive":
        chapter_two_b()
    else:
        chapter_three()


if __name__ == "__main__":
    main()
