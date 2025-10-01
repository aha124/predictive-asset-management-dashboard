"""Visualization helpers for the Advanced Model Deep Dive section."""
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def create_comparison_chart(comparison_df):
    """Return an interactive chart comparing simple vs. RSF predictions."""

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=comparison_df["bridge_id"],
            y=comparison_df["failure_probability"] * 100,
            mode="markers",
            name="Simple Model",
            marker=dict(size=10, color="#8ecae6"),
            text=comparison_df["bridge_name"],
            hovertemplate="Bridge %{text}<br>Simple: %{y:.1f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=comparison_df["bridge_id"],
            y=comparison_df["rsf_failure_probability"] * 100,
            mode="markers",
            name="Random Survival Forest",
            marker=dict(size=10, color="#023047"),
            text=comparison_df["bridge_name"],
            hovertemplate="Bridge %{text}<br>RSF: %{y:.1f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title="Model Prediction Comparison",
        xaxis_title="Bridge ID",
        yaxis_title="Failure Probability (%)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    return fig


def _feature_axis_values(feature):
    if feature == "Age":
        return [int(x) for x in np.linspace(10, 80, 12)]
    if feature == "Condition Score":
        return [1, 2, 3, 4, 5]
    if feature == "Traffic Volume":
        return ["Low", "Medium", "High"]
    if feature == "Maintenance Count":
        return [0, 2, 4, 6, 8, 10]
    if feature == "Days Since Inspection":
        return [30, 90, 180, 270, 360]
    if feature == "Material":
        return ["Concrete", "Steel", "Wood"]
    return [0, 1]


def _to_numeric(feature, value):
    if feature == "Age":
        return float(value)
    if feature == "Condition Score":
        return float(value)
    if feature == "Traffic Volume":
        mapping = {"Low": 0.2, "Medium": 0.5, "High": 0.85}
        return mapping.get(value, 0.5)
    if feature == "Maintenance Count":
        return float(value)
    if feature == "Days Since Inspection":
        return float(value)
    if feature == "Material":
        mapping = {"Concrete": 0.35, "Steel": 0.25, "Wood": 0.65}
        return mapping.get(value, 0.35)
    return float(value)


def create_interaction_heatmap(feature_x, feature_y):
    """Simulate interaction effects between two selected features."""

    x_values = _feature_axis_values(feature_x)
    y_values = _feature_axis_values(feature_y)

    def compute_risk(values):
        risk = 0.15
        for feat, raw_val in values.items():
            val = _to_numeric(feat, raw_val)
            if feat == "Age":
                risk += 0.35 * (val / 80)
            elif feat == "Condition Score":
                risk += 0.3 * ((6 - val) / 5)
            elif feat == "Traffic Volume":
                risk += 0.18 * val
            elif feat == "Maintenance Count":
                risk += 0.22 * min(val / 10, 1)
            elif feat == "Days Since Inspection":
                risk += 0.12 * min(val / 365, 1)
            elif feat == "Material":
                risk += 0.1 * val

        if {"Age", "Condition Score"}.issubset(values.keys()):
            risk += 0.25 * (
                _to_numeric("Age", values["Age"]) / 80
            ) * ((6 - _to_numeric("Condition Score", values["Condition Score"])) / 5)
        if {"Age", "Maintenance Count"}.issubset(values.keys()):
            risk += 0.18 * (
                _to_numeric("Age", values["Age"]) / 80
            ) * min(_to_numeric("Maintenance Count", values["Maintenance Count"]) / 10, 1)
        if {"Traffic Volume", "Material"}.issubset(values.keys()):
            risk += 0.14 * _to_numeric("Traffic Volume", values["Traffic Volume"]) * (
                0.5 if values["Material"] == "Steel" else 1
            )

        return np.clip(risk, 0.01, 0.98)

    z_values = []
    for y in y_values:
        row = []
        for x in x_values:
            pair = {feature_x: x, feature_y: y}
            row.append(compute_risk(pair) * 100)
        z_values.append(row)

    fig = go.Figure(
        data=go.Heatmap(
            x=x_values,
            y=y_values,
            z=z_values,
            colorscale="RdYlGn_r",
            colorbar=dict(title="Failure Probability (%)"),
        )
    )
    fig.update_layout(
        title=f"Feature Interaction: {feature_x} × {feature_y}",
        xaxis_title=feature_x,
        yaxis_title=feature_y,
    )
    return fig


def generate_shap_explanation(bridge_id):
    """Create a SHAP-style waterfall chart for a selected bridge."""

    if "Critical" in bridge_id:
        base_value = 0.25
        contributions = {
            "Age (52 years)": 0.28,
            "Condition Score (1)": 0.22,
            "Traffic (High)": 0.15,
            "Maintenance Count (5)": 0.08,
            "Days Since Inspection": 0.02,
        }
    elif "High" in bridge_id:
        base_value = 0.25
        contributions = {
            "Age (41 years)": 0.18,
            "Condition Score (2)": 0.15,
            "Material (Steel)": -0.02,
            "Traffic (High)": 0.12,
            "Maintenance Count (3)": 0.03,
        }
    else:
        base_value = 0.25
        contributions = {
            "Age (21 years)": -0.1,
            "Condition Score (3)": 0.05,
            "Material (Wood)": 0.08,
            "Traffic (Medium)": 0.05,
            "Days Since Inspection": 0.02,
        }

    feature_names = list(contributions.keys())
    feature_values = list(contributions.values())
    cumulative_prediction = base_value + sum(feature_values)

    fig = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=["relative"] * len(feature_values) + ["total"],
            x=feature_names + ["Final Prediction"],
            text=[f"{value:+.2f}" for value in feature_values] + [f"{cumulative_prediction:.2f}"],
            y=feature_values + [cumulative_prediction],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        )
    )
    fig.update_layout(
        title=f"SHAP Value Explanation: {bridge_id}",
        yaxis_title="Contribution to Failure Probability",
        showlegend=False,
    )
    return fig


def create_confusion_matrix():
    """Create a simulated confusion matrix figure."""

    cm = np.array([[720, 45], [18, 217]])
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=["Did Not Fail", "Failed"],
        y=["Did Not Fail", "Failed"],
        text_auto=True,
        color_continuous_scale="Blues",
    )
    fig.update_layout(
        title="Confusion Matrix: 12-Month Failure Prediction",
        xaxis_title="Predicted Outcome",
        yaxis_title="Actual Outcome",
    )
    return fig


def show_forest_voting_visual():
    """Visualize individual tree votes vs. the ensemble consensus."""

    tree_ids = [f"Tree {i}" for i in range(1, 8)]
    tree_votes = [0.32, 0.48, 0.41, 0.55, 0.62, 0.44, 0.58]
    consensus = float(np.mean(tree_votes))

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=tree_ids,
            y=[vote * 100 for vote in tree_votes],
            name="Individual Trees",
            marker_color="#219ebc",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=tree_ids,
            y=[consensus * 100] * len(tree_ids),
            name="Ensemble Consensus",
            mode="lines",
            line=dict(color="#fb8500", width=3, dash="dash"),
        )
    )
    fig.update_layout(
        title="Forest Voting Illustration",
        yaxis_title="Predicted Failure Probability (%)",
        xaxis_title="Random Survival Forest Trees",
    )
    return fig
