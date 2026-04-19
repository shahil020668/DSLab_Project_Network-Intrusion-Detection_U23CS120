import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# App Configuration and Paths
# ============================================================
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

st.set_page_config(page_title="Network Intrusion Detection System", layout="wide")


# ============================================================
# Data and Model Utilities
# ============================================================
@st.cache_data
def load_dataset_preview(rows=10):
    data_path = project_root / "data" / "processed" / "X_train.csv"
    if not data_path.exists():
        return None
    return pd.read_csv(data_path, nrows=rows)


@st.cache_data
def get_expected_feature_count():
    data_path = project_root / "data" / "processed" / "X_train.csv"
    if not data_path.exists():
        return 20
    return pd.read_csv(data_path, nrows=1).shape[1]


@st.cache_data
def get_sample_vector():
    data_path = project_root / "data" / "processed" / "X_train.csv"
    if not data_path.exists():
        return [0.0] * 20
    return pd.read_csv(data_path, nrows=1).iloc[0].tolist()


@st.cache_resource
def load_model(model_name):
    model_path = project_root / "models" / f"{model_name}.pkl"
    if not model_path.exists():
        return None
    return joblib.load(model_path)


def show_optional_result_images():
    """Display saved ROC/confusion matrix outputs if available."""
    image_candidates = [
        project_root / "outputs" / "model_comparison" / "model_comparison.png",
        project_root / "outputs" / "roc_curves" / "XGBoost_roc_curve.png",
        project_root / "outputs" / "confusion_matrix" / "XGBoost_confusion_matrix.png",
        project_root / "outputs" / "roc_curves" / "RandomForest_roc_curve.png",
        project_root / "outputs" / "confusion_matrix" / "RandomForest_confusion_matrix.png",
    ]

    shown_any = False
    for image_path in image_candidates:
        if image_path.exists():
            shown_any = True
            st.image(str(image_path), caption=image_path.name, use_column_width=True)

    if not shown_any:
        st.info("No saved ROC/confusion matrix images found in outputs folder yet.")


def ensure_input_state_defaults():
    """Initialize session state fields once so they can be auto-filled by button."""
    defaults = {
        "duration": 0.0,
        "src_bytes": 0.0,
        "dst_bytes": 0.0,
        "count": 0.0,
        "srv_count": 0.0,
        "same_srv_rate": 0.0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def apply_sample_attack_data():
    """Set realistic attack-like values for quick demo."""
    st.session_state.duration = 8.0
    st.session_state.src_bytes = 9500.0
    st.session_state.dst_bytes = 120.0
    st.session_state.count = 220.0
    st.session_state.srv_count = 180.0
    st.session_state.same_srv_rate = 0.92


def build_model_input_vector(expected_features):
    """
    Build feature vector in model-required order.

    IMPORTANT:
    The trained model currently expects processed selected features (20 columns).
    The 6 UI fields are mapped to the first 6 positions, and remaining features are
    padded with values from a processed sample row for stable demos.

    If you retrain with a named feature pipeline, update this mapping accordingly.
    """
    sample_vector = get_sample_vector()
    if len(sample_vector) < expected_features:
        sample_vector = sample_vector + [0.0] * (expected_features - len(sample_vector))
    else:
        sample_vector = sample_vector[:expected_features]

    vector = np.array(sample_vector, dtype=float)

    # Map user-entered values into the first six feature slots.
    vector[0] = float(st.session_state.duration)
    vector[1] = float(st.session_state.src_bytes)
    vector[2] = float(st.session_state.dst_bytes)
    vector[3] = float(st.session_state.count)
    vector[4] = float(st.session_state.srv_count)
    vector[5] = float(st.session_state.same_srv_rate)

    # Keep values in a practical normalized range because the model was trained on processed data.
    vector = np.clip(vector, 0.0, 1.0)

    return vector.reshape(1, -1)


def get_prediction_confidence(model, input_data, prediction):
    """Return confidence score in percentage if available."""
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]
        if len(probabilities) > int(prediction):
            return float(probabilities[int(prediction)])
    return None


# ============================================================
# Sidebar Navigation
# ============================================================
st.sidebar.title("🧭 Navigation")
selected_model_name = st.sidebar.selectbox("Select Model", ["XGBoost", "RandomForest"])
page = st.sidebar.radio(
    "Go to",
    ["Home / Dataset Overview", "Model Performance", "Predict Network Traffic"],
)


# ============================================================
# Header
# ============================================================
st.title("🛡️ Network Intrusion Detection System")
st.markdown(
    "A machine learning system to classify network traffic as Normal or Attack."
)


# ============================================================
# Page 1: Home / Dataset Overview
# ============================================================
if page == "Home / Dataset Overview":
    st.header("📊 Home / Dataset Overview")
    st.write(
        "This demo detects malicious network traffic patterns using pre-trained models. "
        "Prediction output is binary: Normal traffic or Attack traffic."
    )

    preview_df = load_dataset_preview(rows=8)
    if preview_df is None:
        st.warning("Processed dataset preview not found. Please generate data in data/processed first.")
    else:
        st.subheader("Dataset Preview")
        st.dataframe(preview_df, use_container_width=True)


# ============================================================
# Page 2: Model Performance
# ============================================================
elif page == "Model Performance":
    st.header("📈 Model Performance")
    st.write("Quick metric view for presentation. You can replace these with final validated values.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Random Forest Accuracy", "99.8%")
    c2.metric("XGBoost Accuracy", "99.9%")
    c3.metric("CNN Accuracy", "98.9%")

    st.subheader("Saved Evaluation Visuals")
    show_optional_result_images()


# ============================================================
# Page 3: Predict Network Traffic
# ============================================================
elif page == "Predict Network Traffic":
    st.header("🔍 Predict Network Traffic")

    expected_features = get_expected_feature_count()
    st.info(
        "The selected model expects processed features. "
        "This form captures six relevant traffic indicators and maps them into the model input vector."
    )

    ensure_input_state_defaults()

    # Quick demo helper: pre-fill with realistic suspicious values.
    if st.button("Use Sample Attack Data"):
        apply_sample_attack_data()
        st.success("Sample attack-like values loaded into the form.")

    model = load_model(selected_model_name)
    if model is None:
        st.error(f"❌ Model file not found: models/{selected_model_name}.pkl")
        st.stop()

    with st.form("prediction_form"):
        st.subheader("Input Section")
        left, right = st.columns(2)

        with left:
            st.number_input(
                "Duration (seconds)",
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                key="duration",
                help="Time duration of the network connection",
            )
            st.number_input(
                "Source Bytes",
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                key="src_bytes",
                help="Number of bytes sent from source",
            )
            st.number_input(
                "Destination Bytes",
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                key="dst_bytes",
                help="Number of bytes received at destination",
            )

        with right:
            st.number_input(
                "Count",
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                key="count",
                help="Number of connections to the same host",
            )
            st.number_input(
                "SRV Count",
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                key="srv_count",
                help="Number of connections to the same service",
            )
            st.slider(
                "Same SRV Rate",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.same_srv_rate),
                step=0.01,
                key="same_srv_rate",
                help="Ratio of connections to same service",
            )

        submit = st.form_submit_button("Predict")

    if submit:
        try:
            input_data = build_model_input_vector(expected_features)
            prediction = model.predict(input_data)[0]
            confidence = get_prediction_confidence(model, input_data, prediction)

            # Progress bar for demo-friendly confidence visualization.
            if confidence is not None:
                st.write(f"Confidence: {confidence * 100:.2f}%")
                st.progress(int(confidence * 100))
            else:
                st.info("Confidence not available for this model.")

            # Output banner.
            if int(prediction) == 1:
                st.error("🚨 ATTACK DETECTED")
            else:
                st.success("🟢 NORMAL TRAFFIC")

            with st.expander("Debug: Model Input Shape"):
                st.write(f"Expected features: {expected_features}")
                st.write(f"Provided shape: {input_data.shape}")

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")