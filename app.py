# app.py
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

# Use GitHub raw URL for logo
LOGO_URL = "https://raw.githubusercontent.com/Pandey-vhr/GeoRockSlope/main/assets/GECL.png"

# ----------------------------
# Constants
# ----------------------------
FEATURE_ORDER = ["SlopeHeight","SlopeAngle","UCS","GSI","mi","D","PoissonsRatio","E","Density"]

INPUT_LABELS = {
    'MODEL'       : 'Prediction Model',
    'SLOPE_HEIGHT': 'Slope Height',
    'SLOPE_ANGLE' : 'Slope Angle',
    'UCS'         : 'Uniaxial Compressive Strength',
    'GSI'         : 'Geological Strength Index',
    'MI'          : 'Material Constant (mi)',
    'D_VAL'       : 'Disturbance Factor',
    'PR'          : "Poisson's Ratio",
    'YM'          : 'Young’s Modulus (E) of Intact Rock',
    'DEN'         : 'Density'
}

# Fallback bounds if training_ranges.json is absent
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

# Saturated condition reduction sampled in [14.9% ± 1%]
SAT_REDUCTION_BASE = 0.149
SAT_JITTER_ABS = 0.01  # gives range 0.139 to 0.159

def sample_sat_factor(seed=None):
    rng = np.random.default_rng(seed)
    r = rng.uniform(SAT_REDUCTION_BASE - SAT_JITTER_ABS, SAT_REDUCTION_BASE + SAT_JITTER_ABS)
    return r, 1.0 - r  # (reduction, multiplicative factor)

# ----------------------------
# Header with logo (top-right)
# ----------------------------
def header_with_logo(title: str = "GeoRockSlope", logo_width: int = 96):
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.markdown(f"<h1 style='margin:0'>{title}</h1>", unsafe_allow_html=True)
    with col2:
        st.image(LOGO_URL, width=logo_width)

# ----------------------------
# Loaders
# ----------------------------
@st.cache_resource
def load_ranges():
    if RANGES_PATH.exists():
        try:
            return json.loads(RANGES_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return None

def _pretty_model_name(folder_name: str) -> str:
    s = folder_name.lower()
    if s.startswith("abc_"):
        algo = "Artificial Bee Colony"
    elif s.startswith("ga_"):
        algo = "Genetic Algorithm"
    elif s.startswith("acor_"):
        algo = "Ant Colony Optimization (ACOR)"
    else:
        algo = folder_name.replace("_", " ").title()
    seismic = " (Seismic)" if ("sf" in s or "seismic" in s) else ""
    return f"{algo}{seismic}"

@st.cache_resource
def load_manifest():
    # 1) Explicit manifest
    if MANIFEST_PATH.exists():
        try:
            data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "models" in data:
                return data["models"]
            if isinstance(data, list):
                return data
        except Exception as e:
            st.warning(f"Manifest read error: {e}")

    # 2) Auto-discover subfolders with model and scalers
    entries = []
    if MODELS_DIR.exists():
        for sub in sorted(p for p in MODELS_DIR.iterdir() if p.is_dir()):
            m, sx, sy = sub/"model.joblib", sub/"scaler_X.joblib", sub/"scaler_y.joblib"
            if m.exists() and sx.exists() and sy.exists():
                entries.append({
                    "id": sub.name,
                    "name": _pretty_model_name(sub.name),
                    "model_path": str(m),
                    "scaler_X_path": str(sx),
                    "scaler_y_path": str(sy),
                    "target_name": "Seismic FoS" if ("sf" in sub.name.lower() or "seismic" in sub.name.lower()) else "FoS",
                    "feature_names": FEATURE_ORDER
                })
    return entries

@st.cache_resource(show_spinner=False)
def load_artifacts(entry):
    model = joblib.load(entry["model_path"])
    scaler_X = joblib.load(entry["scaler_X_path"])
    scaler_y = joblib.load(entry["scaler_y_path"])
    return model, scaler_X, scaler_y

# ----------------------------
# Helpers for inputs
# ----------------------------
def get_bounds(name, ranges_data):
    if ranges_data and "ranges" in ranges_data and name in ranges_data["ranges"]:
        r = ranges_data["ranges"][name]
        return float(r["min"]), float(r["max"])
    return DEFAULT_BOUNDS.get(name, (None, None))

def rng_help(name, ranges_data):
    mn, mx = get_bounds(name, ranges_data)
    if mn is None or mx is None:
        return ""
    unit = UNITS.get(name, "")
    return f"Training range: {mn:g} to {mx:g}{unit}"

def int_input(label, mn, mx, key, help_txt):
    default = st.session_state.get(key, int(mn))
    val = st.number_input(label, min_value=int(mn), max_value=int(mx),
                          value=int(default), step=1, format="%d", help=help_txt, key=key)
    return val

def float_input(label, mn, mx, step, fmt, help_txt, key, epsilon=0.0):
    default = st.session_state.get(key, float(mn))
    val = st.number_input(
        label=label,
        min_value=float(mn),
        max_value=float(mx) + float(epsilon),
        value=float(default),
        step=float(step),
        format=fmt,
        help=help_txt,
        key=key,
    )
    return val

def label_with_unit(base_label, field_key):
    unit = UNITS.get(field_key, "")
    return f"{base_label}{f' ({unit.strip()})' if unit else ''}"

def render_inputs(feature_names, ranges_data):
    colLeft, colRight = st.columns(2)
    vals = {}

    # Left column
    mn, mx = get_bounds("SlopeHeight", ranges_data)
    with colLeft:
        vals["SlopeHeight"] = float_input(
            label_with_unit(INPUT_LABELS['SLOPE_HEIGHT'], "SlopeHeight"),
            mn, mx, 0.1, "%.1f", rng_help("SlopeHeight", ranges_data), key="SlopeHeight"
        )
    mn, mx = get_bounds("SlopeAngle", ranges_data)
    with colLeft:
        vals["SlopeAngle"] = float_input(
            label_with_unit(INPUT_LABELS['SLOPE_ANGLE'], "SlopeAngle"),
            mn, mx, 0.1, "%.1f", rng_help("SlopeAngle", ranges_data), key="SlopeAngle"
        )
    mn, mx = get_bounds("UCS", ranges_data)
    with colLeft:
        vals["UCS"] = float_input(
            label_with_unit(INPUT_LABELS['UCS'], "UCS"),
            mn, mx, 0.1, "%.1f", rng_help("UCS", ranges_data), key="UCS"
        )
    mn, mx = get_bounds("GSI", ranges_data)
    with colLeft:
        vals["GSI"] = int_input(
            INPUT_LABELS['GSI'], mn, mx, key="GSI", help_txt=rng_help("GSI", ranges_data)
        )

    # Right column
    mn, mx = get_bounds("mi", ranges_data)
    with colRight:
        vals["mi"] = int_input(
            INPUT_LABELS['MI'], mn, mx, key="mi", help_txt=rng_help("mi", ranges_data)
        )

    # Disturbance factor via dropdown
    vals["D"] = D_VALS[st.selectbox(
        INPUT_LABELS['D_VAL'], list(D_VALS.keys()),
        help=rng_help("D", ranges_data), key="D_label"
    )]

    # Poisson's Ratio
    mn, mx = get_bounds("PoissonsRatio", ranges_data)
    with colRight:
        vals["PoissonsRatio"] = float_input(
            INPUT_LABELS['PR'], mn, mx, 0.01, "%.2f",
            rng_help("PoissonsRatio", ranges_data), key="PoissonsRatio",
            epsilon=1e-9
        )

    mn, mx = get_bounds("E", ranges_data)
    with colRight:
        vals["E"] = float_input(
            label_with_unit(INPUT_LABELS['YM'], "E"),
            mn, mx, 0.1, "%.1f", rng_help("E", ranges_data), key="E"
        )
    mn, mx = get_bounds("Density", ranges_data)
    with colRight:
        vals["Density"] = float_input(
            label_with_unit(INPUT_LABELS['DEN'], "Density"),
            mn, mx, 0.01, "%.2f", rng_help("Density", ranges_data), key="Density"
        )

    # Subtle out-of-range nudge
    for k, v in vals.items():
        mn, mx = get_bounds(k, ranges_data)
        if mn is not None and mx is not None and not (mn <= float(v) <= mx):
            st.caption(f"Note: {k} is outside training range [{mn:g}, {mx:g}].")

    x_row = [float(vals[n]) for n in feature_names]
    return vals, x_row

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

st.info(
    "Models were trained on results from finite-element analyses of 494 slope models using the Generalized Hoek-Brown criterion. "
    "For FoS, ABC-ANN achieved test R^2 ≈ 0.9376 with RMSE ≈ 0.318; for Seismic-FoS, GA-ANN achieved test R^2 ≈ 0.9178 with RMSE ≈ 0.251."
)

ranges = load_ranges()
models = load_manifest()
if not models:
    st.error("No models found. Each folder in 'models/' must contain model.joblib, scaler_X.joblib, scaler_y.joblib (e.g., abc_ann_f, abc_ann_sf, acor_ann_f, acor_ann_sf, ga_ann_f, ga_ann_sf).")
    st.caption(f"Looking in: {MODELS_DIR}")
    st.stop()

choices = {m["name"]: m for m in models}
chosen = st.selectbox(INPUT_LABELS['MODEL'], list(choices.keys()))
entry = choices[chosen]

badge_cols = st.columns(3)
with badge_cols[0]:
    st.caption(f"Model ID: `{entry.get('id')}`")
with badge_cols[1]:
    st.caption(f"Target: **{entry.get('target_name', 'FoS')}**")
with badge_cols[2]:
    st.caption("Features: " + ", ".join(entry.get("feature_names", FEATURE_ORDER)))

model, scaler_X, scaler_y = load_artifacts(entry)
feature_names = entry.get("feature_names", FEATURE_ORDER)
target_name = entry.get("target_name", "FoS")

# Saturated FoS toggle with random reduction control
is_seismic = target_name.lower() != "fos"
use_saturated_estimate = st.checkbox(
    "Estimate FoS under Saturated condition (random reduction in 14.9% ± 1%)",
    value=st.session_state.get("use_saturated_estimate", False) and not is_seismic,
    disabled=is_seismic,
    help=("Uses a normal FoS model prediction, then multiplies by a random factor in [0.841, 0.861]. "
          "Disabled for Seismic models."),
    key="use_saturated_estimate"
)

seed = None
if use_saturated_estimate and not is_seismic:
    with st.expander("Randomness options", expanded=False):
        use_seed = st.checkbox("Use fixed random seed for reproducibility", value=False, key="sat_use_seed")
        if use_seed:
            seed = st.number_input("Seed (integer)", min_value=0, max_value=2**31-1, value=0, step=1, key="sat_seed")
    # Quick way to change the sampled reduction without changing inputs
    if st.button("Resample reduction"):
        st.experimental_rerun()

with st.expander("Model details", expanded=False):
    st.json({
        "id": entry.get("id"),
        "name": entry.get("name"),
        "target_name": target_name,
        "paths": {
            "model": entry.get("model_path"),
            "scaler_X": entry.get("scaler_X_path"),
            "scaler_y": entry.get("scaler_y_path"),
        },
        "feature_names": feature_names
    })

values, x_row = render_inputs(feature_names, ranges)

# ----------------------------
# Predict button and outputs
# ----------------------------
if hasattr(scaler_X, "n_features_in_") and scaler_X.n_features_in_ != len(x_row):
    st.error(f"Feature count mismatch: scaler expects {scaler_X.n_features_in_}, got {len(x_row)}")
else:
    btn_label = f"Predict {'Saturated FoS' if (use_saturated_estimate and not is_seismic) else target_name}"
    if st.button(btn_label, type="primary"):
        try:
            y = predict_one(model, scaler_X, scaler_y, x_row)

            if use_saturated_estimate and not is_seismic:
                reduction, sat_factor = sample_sat_factor(seed=seed)
                y_sat = y * sat_factor
                st.success(f"Estimated Saturated FoS: **{y_sat:.4f}**")
                colA, colB, colC = st.columns(3)
                with colA:
                    st.metric("Normal FoS (model prediction)", f"{y:.4f}")
                with colB:
                    st.metric("Applied reduction", f"{reduction*100:.2f}%")
                with colC:
                    st.metric("Saturated FoS", f"{y_sat:.4f}", delta=f"-{reduction*100:.2f}%")
                st.caption(
                    f"Computed as: Saturated FoS = Normal FoS × {sat_factor:.4f} "
                    f"(reduction sampled uniformly in "
                    f"[{(SAT_REDUCTION_BASE - SAT_JITTER_ABS)*100:.2f}%, {(SAT_REDUCTION_BASE + SAT_JITTER_ABS)*100:.2f}%])."
                )
            else:
                st.success(f"Predicted {target_name}: **{y:.4f}**")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
