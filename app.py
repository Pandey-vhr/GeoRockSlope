import json
from pathlib import Path

import joblib
import numpy as np
import streamlit as st

st.set_page_config(page_title="🧩 GeoRockSlope", page_icon="🪨", layout="centered")

# ----------------------------
# Paths and setup
# ----------------------------
BASE = Path(__file__).parent.resolve()
MODELS_DIR = BASE / "models"
MANIFEST_PATH = MODELS_DIR / "models_manifest.json"
RANGES_PATH = BASE / "training_ranges.json"

LOGO_URL = "https://raw.githubusercontent.com/Pandey-vhr/GeoRockSlope/main/assets/GECL.png"

FEATURE_ORDER = ["SlopeHeight","SlopeAngle","UCS","GSI","mi","D","PoissonsRatio","E","Density"]

DEFAULT_BOUNDS = {
    "SlopeHeight": (13.0, 74.0),
    "SlopeAngle":  (55.0, 84.0),
    "UCS":        (42.0, 87.0),
    "GSI":        (25, 85),
    "mi":         (23, 35),
    "PoissonsRatio": (0.15, 0.22),
    "E":          (8783.0, 36123.0),
    "Density":    (2.55, 2.75)
}

UNITS = {
    "SlopeHeight": " m",
    "SlopeAngle": "°",
    "UCS": " MPa",
    "GSI": "",
    "mi": "",
    "D": "",
    "PoissonsRatio": "",
    "E": " MPa",
    "Density": " g/cm³",
}

D_VALS = {
    'Moderately Disturbed Rock Mass': 0.7,
    'Very Disturbed Rock Mass': 1.0,
}

SAT_FACTOR_LOW  = 0.821
SAT_FACTOR_HIGH = 0.881

# ----------------------------
# Header
# ----------------------------

def header_with_logo(title: str = "GeoRockSlope", logo_width: int = 96):
    col1, col2 = st.columns([0.75, 0.25])
    with col1:
        st.markdown(f"<h1 style='margin:0'>{title}</h1>", unsafe_allow_html=True)
    with col2:
        st.image(LOGO_URL, width=logo_width)

# ----------------------------
# Persistent RNG
# ----------------------------

def _get_rng(seed: int | None):
    if "sat_rng" not in st.session_state or (seed is not None and st.session_state.get("sat_seed") != seed):
        st.session_state["sat_seed"] = seed
        st.session_state["sat_rng"] = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    return st.session_state["sat_rng"]

# ----------------------------
# Loaders
# ----------------------------

@st.cache_data
def load_ranges():
    if RANGES_PATH.exists():
        try:
            return json.loads(RANGES_PATH.read_text(encoding="utf-8"))
        except Exception as e:
            st.warning(f"Could not parse training_ranges.json: {e}")
    return None


@st.cache_resource
def load_manifest():
    if MANIFEST_PATH.exists():
        try:
            data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "models" in data:
                return data["models"]
            if isinstance(data, list):
                return data
        except Exception as e:
            st.warning(f"Manifest read error: {e}")
    entries = []
    if MODELS_DIR.exists():
        for sub in sorted(p for p in MODELS_DIR.iterdir() if p.is_dir() and not p.name.startswith(".")):
            m, sx, sy = sub/"model.joblib", sub/"scaler_X.joblib", sub/"scaler_y.joblib"
            if m.exists() and sx.exists() and sy.exists():
                entries.append({
                    "id": sub.name,
                    "name": sub.name.replace("_", " ").title(),
                    "model_path": str(m),
                    "scaler_X_path": str(sx),
                    "scaler_y_path": str(sy),
                    "target_name": "FoS",
                    "feature_names": FEATURE_ORDER
                })
    return entries


@st.cache_resource(show_spinner=False)
def load_artifacts(entry):
    try:
        model = joblib.load(entry["model_path"])
        scaler_X = joblib.load(entry["scaler_X_path"])
        scaler_y = joblib.load(entry["scaler_y_path"])
        return model, scaler_X, scaler_y
    except Exception as e:
        raise RuntimeError(f"Failed to load artifacts: {e}")

# ----------------------------
# Input helpers
# ----------------------------

def get_bounds(name, ranges_data):
    if ranges_data and "ranges" in ranges_data and name in ranges_data["ranges"]:
        r = ranges_data["ranges"][name]
        return float(r["min"]), float(r["max"])
    return DEFAULT_BOUNDS.get(name, (None, None))


def float_input(label, mn, mx, step, fmt, help_txt, key):
    return st.number_input(label, min_value=float(mn), max_value=float(mx), value=float(mn), step=float(step), format=fmt, help=help_txt, key=key)


def label_with_unit(base_label, field_key):
    unit = UNITS.get(field_key, "")
    return f"{base_label}{f' ({unit.strip()})' if unit else ''}"


def render_inputs(feature_names, ranges_data):
    vals = {}
    for name in feature_names:
        mn, mx = get_bounds(name, ranges_data)
        vals[name] = float_input(label_with_unit(name, name), mn, mx, 0.1, "%.2f", f"Training range: {mn} to {mx}", key=name)
    return vals, [vals[n] for n in feature_names]


def predict_one(model, scaler_X, scaler_y, row_vals):
    X = np.array(row_vals, dtype=float).reshape(1, -1)
    Xs = scaler_X.transform(X)
    y_scaled = model.predict(Xs).reshape(-1, 1)
    y = scaler_y.inverse_transform(y_scaled).ravel()
    return float(y[0])

# ----------------------------
# Page layout
# ----------------------------

header_with_logo()

ranges = load_ranges()
models = load_manifest()
if not models:
    st.error("No models found in 'models/' directory.")
    st.stop()

entry = models[0]
model, scaler_X, scaler_y = load_artifacts(entry)
feature_names = entry.get("feature_names", FEATURE_ORDER)

use_saturated_estimate = st.checkbox("Estimate FoS under Saturated condition", value=False, key="use_saturated_estimate")

values, x_row = render_inputs(feature_names, ranges)

if st.button("Predict", type="primary"):
    try:
        fos = predict_one(model, scaler_X, scaler_y, x_row)
        if use_saturated_estimate:
            rng = _get_rng(None)
            low, high = fos * SAT_FACTOR_LOW, fos * SAT_FACTOR_HIGH
            y_sat = float(rng.uniform(low, high))
            st.success(f"Saturated FoS: **{y_sat:.4f}**")
        else:
            st.success(f"Predicted FoS: **{fos:.4f}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
