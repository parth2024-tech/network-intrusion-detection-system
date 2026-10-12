"""
src/dashboard/app.py
=====================
Streamlit dashboard for the NIDS system.

Displays:
- Dataset summary and class distribution
- Model evaluation metrics (binary + multiclass)
- Live / batch flow prediction interface
- SHAP feature importance
- Confusion matrix visualisations

Run::

    streamlit run src/dashboard/app.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="NIDS — Network Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
DATA_DIR = Path("data/processed")
CONFIG_PATH = Path("config/model_config.yaml")


@st.cache_data(show_spinner="Loading model metrics …")
def load_metrics(mode: str = "binary") -> dict[str, Any]:
    """Load saved metrics JSON from reports/metrics/."""
    path = REPORTS_DIR / "metrics" / f"{mode}_summary.json"
    if not path.exists():
        return {}
    with path.open() as fh:
        return json.load(fh)


@st.cache_resource(show_spinner="Loading model …")
def load_model(model_name: str, mode: str = "binary"):
    """Load a joblib model from models/."""
    import joblib
    path = MODELS_DIR / f"{model_name}_{mode}.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_resource(show_spinner="Loading scaler …")
def load_scaler():
    import joblib
    path = MODELS_DIR / "scaler.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_data(show_spinner="Loading test data …")
def load_test_data():
    path = DATA_DIR / "test.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def _severity_badge(score: float) -> str:
    if score >= 0.9:
        return "🔴 CRITICAL"
    elif score >= 0.7:
        return "🟠 HIGH"
    elif score >= 0.5:
        return "🟡 MEDIUM"
    return "🟢 LOW"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🛡️ NIDS Dashboard")
    st.markdown("**Network Intrusion Detection System**")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["📊 Overview", "🎯 Model Performance", "🔍 Predict Flow", "🧠 Explainability"],
    )
    st.markdown("---")
    st.caption("Dataset: CICIDS2017")
    st.caption("Models: RF · XGBoost · LightGBM")


# ---------------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------------

if page == "📊 Overview":
    st.title("📊 Dataset Overview")

    test_df = load_test_data()
    if test_df is None:
        st.warning(
            "No processed data found. "
            "Run `python -m src.data.preprocess` first."
        )
    else:
        col1, col2, col3, col4 = st.columns(4)
        total = len(test_df)
        attacks = int(test_df["label_binary"].sum())
        col1.metric("Total Flows (test)", f"{total:,}")
        col2.metric("Attack Flows", f"{attacks:,}")
        col3.metric("Benign Flows", f"{total - attacks:,}")
        col4.metric("Attack Rate", f"{attacks / total * 100:.1f}%")

        st.markdown("### Class Distribution — Test Set (Multiclass)")
        class_names = [
            "BENIGN", "Brute Force", "DoS/DDoS", "Heartbleed",
            "Web Attack", "Infiltration", "Botnet", "Port Scan",
        ]
        label_counts = test_df["label_multi"].value_counts().sort_index()
        label_df = pd.DataFrame({
            "Attack Type": [class_names[i] if i < len(class_names) else str(i)
                            for i in label_counts.index],
            "Count": label_counts.values,
        })
        st.bar_chart(label_df.set_index("Attack Type"))

        st.markdown("### Sample Flows")
        feature_cols = [c for c in test_df.columns if c not in {"label_binary", "label_multi"}]
        st.dataframe(test_df[feature_cols].head(10), use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Model Performance
# ---------------------------------------------------------------------------

elif page == "🎯 Model Performance":
    st.title("🎯 Model Performance")

    tab_binary, tab_multi = st.tabs(["Binary (Benign vs Attack)", "Multiclass (Attack Type)"])

    with tab_binary:
        metrics = load_metrics("binary")
        if not metrics:
            st.info("No binary metrics found. Train models with `python -m src.models.train_binary`.")
        else:
            cols = st.columns(len(metrics))
            for col, (model_name, m) in zip(cols, metrics.items()):
                col.subheader(model_name.replace("_", " ").title())
                col.metric("Recall (Attack)", f"{m.get('recall_attack', 0):.3f}")
                col.metric("Precision", f"{m.get('precision_attack', 0):.3f}")
                col.metric("F1 (Attack)", f"{m.get('f1_attack', 0):.3f}")
                col.metric("ROC-AUC", f"{m.get('roc_auc', 0):.3f}")

        # Confusion matrices
        st.markdown("### Confusion Matrices")
        plots = list((REPORTS_DIR / "plots").glob("*_binary_cm.png"))
        if plots:
            img_cols = st.columns(len(plots))
            for col, img_path in zip(img_cols, plots):
                col.image(str(img_path), caption=img_path.stem, use_container_width=True)
        else:
            st.info("No confusion matrix plots yet.")

        # ROC curves
        roc_plots = list((REPORTS_DIR / "plots").glob("*binary*roc*.png"))
        if roc_plots:
            st.markdown("### ROC Curves")
            img_cols = st.columns(len(roc_plots))
            for col, img_path in zip(img_cols, roc_plots):
                col.image(str(img_path), caption=img_path.stem, use_container_width=True)

    with tab_multi:
        metrics = load_metrics("multiclass")
        if not metrics:
            st.info("No multiclass metrics found. Train with `python -m src.models.train_multiclass`.")
        else:
            for model_name, m in metrics.items():
                st.subheader(model_name.replace("_", " ").title())
                report = m.get("report", {})
                if report:
                    report_df = pd.DataFrame(report).T
                    st.dataframe(report_df.style.format("{:.3f}", na_rep="-"), use_container_width=True)
                st.markdown("---")


# ---------------------------------------------------------------------------
# Page: Predict Flow
# ---------------------------------------------------------------------------

elif page == "🔍 Predict Flow":
    st.title("🔍 Predict Network Flow")
    st.markdown(
        "Upload a CSV of network flows **or** adjust sliders to simulate a single flow."
    )

    model_choice = st.selectbox(
        "Select Model",
        ["random_forest", "xgboost", "lightgbm"],
    )
    mode_choice = st.radio("Classification Mode", ["binary", "multiclass"], horizontal=True)

    model = load_model(model_choice, mode_choice)
    scaler = load_scaler()

    if model is None:
        st.warning(f"Model `{model_choice}_{mode_choice}.joblib` not found. Train it first.")
    else:
        # --- Batch prediction via file upload ---
        uploaded = st.file_uploader("Upload CSV of flows", type=["csv"])
        if uploaded:
            try:
                df_up = pd.read_csv(uploaded, encoding="latin-1", low_memory=False)
                st.write(f"Loaded {len(df_up):,} flows")

                # Load feature cols
                feat_path = MODELS_DIR / "feature_cols.json"
                if feat_path.exists():
                    feature_cols = json.loads(feat_path.read_text())
                    available = [c for c in feature_cols if c in df_up.columns]
                    missing = [c for c in feature_cols if c not in df_up.columns]
                    if missing:
                        st.warning(f"Missing {len(missing)} feature columns — padding with 0.")
                    X_up = np.zeros((len(df_up), len(feature_cols)), dtype=np.float32)
                    for i, col in enumerate(feature_cols):
                        if col in df_up.columns:
                            X_up[:, i] = pd.to_numeric(df_up[col], errors="coerce").fillna(0).values
                else:
                    st.error("feature_cols.json not found. Re-run preprocessing.")
                    X_up = None

                if X_up is not None and scaler:
                    X_up = scaler.transform(X_up)
                    preds = model.predict(X_up)
                    df_up["prediction"] = preds
                    if mode_choice == "binary":
                        class_map = {0: "✅ BENIGN", 1: "🚨 ATTACK"}
                        df_up["label"] = df_up["prediction"].map(class_map)
                    st.success(f"Predictions complete. Attack flows: {int((preds == 1).sum())}")
                    st.dataframe(df_up[["prediction"] + (["label"] if mode_choice == "binary" else [])].head(50))

                    csv_out = df_up.to_csv(index=False).encode()
                    st.download_button("📥 Download Predictions CSV", csv_out, "predictions.csv", "text/csv")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Error processing file: {exc}")

        # --- Single flow simulation via sliders ---
        st.markdown("---")
        st.subheader("🔬 Single Flow Simulator")
        with st.expander("Adjust flow features"):
            flow_duration = st.slider("Flow Duration (µs)", 0, 120_000_000, 5000)
            total_fwd_pkts = st.slider("Total Fwd Packets", 0, 10000, 10)
            total_bwd_pkts = st.slider("Total Bwd Packets", 0, 10000, 10)
            fwd_pkt_len_mean = st.slider("Avg Fwd Packet Length", 0, 1500, 500)
            syn_flag = st.slider("SYN Flag Count", 0, 100, 1)
            flow_bytes_s = st.slider("Flow Bytes/s", 0, 10_000_000, 10000)

        if st.button("🚀 Predict This Flow"):
            # Build a minimal feature vector (all others = 0)
            feat_path = MODELS_DIR / "feature_cols.json"
            if feat_path.exists() and scaler:
                feature_cols = json.loads(feat_path.read_text())
                x = np.zeros((1, len(feature_cols)), dtype=np.float32)
                feature_map = {
                    "Flow Duration": flow_duration,
                    "Total Fwd Packets": total_fwd_pkts,
                    "Total Backward Packets": total_bwd_pkts,
                    "Fwd Packet Length Mean": fwd_pkt_len_mean,
                    "SYN Flag Count": syn_flag,
                    "Flow Bytes/s": flow_bytes_s,
                }
                for fname, val in feature_map.items():
                    if fname in feature_cols:
                        x[0, feature_cols.index(fname)] = val

                x_scaled = scaler.transform(x)
                pred = model.predict(x_scaled)[0]
                proba = model.predict_proba(x_scaled)[0] if hasattr(model, "predict_proba") else None

                if mode_choice == "binary":
                    label = "🚨 ATTACK" if pred == 1 else "✅ BENIGN"
                    confidence = float(proba[pred]) if proba is not None else float("nan")
                    st.metric("Prediction", label)
                    st.metric("Confidence", f"{confidence:.1%}")
                    if pred == 1 and proba is not None:
                        st.metric("Severity", _severity_badge(proba[1]))
                else:
                    class_names = [
                        "BENIGN", "Brute Force", "DoS/DDoS", "Heartbleed",
                        "Web Attack", "Infiltration", "Botnet", "Port Scan",
                    ]
                    label = class_names[pred] if pred < len(class_names) else str(pred)
                    st.metric("Predicted Attack Type", label)
            else:
                st.error("Scaler or feature_cols.json not found.")


# ---------------------------------------------------------------------------
# Page: Explainability
# ---------------------------------------------------------------------------

elif page == "🧠 Explainability":
    st.title("🧠 Model Explainability (SHAP)")

    # Show pre-computed SHAP plots if available
    shap_plots = list((REPORTS_DIR / "plots").glob("shap_*.png"))
    if shap_plots:
        st.markdown("### Global Feature Importance")
        for p in shap_plots:
            st.image(str(p), caption=p.stem, use_container_width=True)
    else:
        st.info(
            "No SHAP plots found. Run:\n"
            "```\npython -m src.explainability.run_shap\n```"
        )

    st.markdown("---")
    st.markdown("### How SHAP Works")
    st.markdown(
        """
        **SHAP (SHapley Additive exPlanations)** assigns each feature a contribution
        value (Shapley value) to explain a model prediction.

        | Value | Meaning |
        |-------|---------|
        | Large positive | Feature pushes prediction toward **ATTACK** |
        | Large negative | Feature pushes prediction toward **BENIGN** |
        | Near zero | Feature has little effect on this prediction |

        > **Example:** A high `SYN Flag Count` with zero `ACK Flag Count` strongly
        > indicates a SYN flood (DDoS), giving SYN a large positive SHAP value.
        """
    )
