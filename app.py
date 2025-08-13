import streamlit as st
import numpy as np
import joblib
from pathlib import Path
import json

st.set_page_config(page_title="🧩 GeoRockSlope", page_icon="🪨", layout="centered")

# ----------------------------
# Paths (repo-relative)
# ----------------------------
BASE = Path(__file__).parent.resolve()
MODELS_DIR = BASE / "models"
MANIFEST_PATH = MODELS_DIR / "models_manifest.json"
RANGES_PATH = BASE / "training_ranges.json"

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

# Numeric fallbacks if training_ranges.json is absent
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

# Units to show in training-range helpers
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

# ----------------------------
# Data loading helpers
# ----------------------------
@st.cache_resource
def load_ranges():
    if RANGES_PATH.exists():
        try:
            with open(RANGES_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None

def _pretty_model_name(folder_name: str) -> str:
    """Readable labels for ABC / GA / ACOR variants (+Seismic)."""
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
    # 1) If a manifest is provided, honor it
    if MANIFEST_PATH.exists():
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "models" in data:
                return data["models"]
            if isinstance(data, list):
                return data
        except Exception as e:
            st.warning(f"Manifest read error: {e}")

    # 2) Auto-discover subfolders with model+scalers (includes acor_ann_f / acor_ann_sf)
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
    # Cache global ML artifacts for performance
    model = joblib.load(entry["model_path"])
    scaler_X = joblib.load(entry["scaler_X_path"])
    scaler_y = joblib.load(entry["scaler_y_path"])
    return model, scaler_X, scaler_y

# ----------------------------
# UI helpers
# ----------------------------
def get_bounds(name, ranges_data):
    """Return (min, max) for a feature from training_ranges.json or defaults."""
    if ranges_data and "ranges" in ranges_data and name in ranges_data["ranges"]:
        r = ranges_data["ranges"][name]
        return float(r["min"]), float(r["max"])
    return DEFAULT_BOUNDS.get(name, (None, None))

def rng_help(name, ranges_data):
    mn, mx = get_bounds(name, ranges_data)
    if mn is None or mx is None:
        return ""
    unit = UNITS.get(name, "")
    # Compact, consistent helper
    return f"Training range: {mn:g} to {mx:g}{unit}"

def int_input(label, mn, mx, val, help_txt):
    return st.number_input(label, min_value=int(mn), max_value=int(mx),
                           value=int(val), step=1, format="%d", help=help_txt)

def float_input(label, mn, mx, val, step, fmt, help_txt):
    return st.number_input(label, min_value=float(mn), max_value=float(mx),
                           value=float(val), step=float(step), format=fmt, help=help_txt)

def render_inputs(feature_names, ranges_data):
    colLeft, colRight = st.columns(2)
    vals = {}

    # Left side
    mn, mx = get_bounds("SlopeHeight", ranges_data)
    with colLeft:
        vals["SlopeHeight"] = float_input(INPUT_LABELS['SLOPE_HEIGHT'], mn, mx, mn, 0.1, "%.1f", rng_help("SlopeHeight", ranges_data))
    mn, mx = get_bounds("SlopeAngle", ranges_data)
    with colLeft:
        vals["SlopeAngle"]  = float_input(INPUT_LABELS['SLOPE_ANGLE'],  mn, mx, mn, 0.1, "%.1f", rng_help("SlopeAngle", ranges_data))
    mn, mx = get_bounds("UCS", ranges_data)
    with colLeft:
        vals["UCS"]         = float_input(INPUT_LABELS['UCS'],          mn, mx, mn, 0.1, "%.1f", rng_help("UCS", ranges_data))
    mn, mx = get_bounds("GSI", ranges_data)
    with colLeft:
        vals["GSI"]         = int_input  (INPUT_LABELS['GSI'],          mn, mx, mn,           rng_help("GSI", ranges_data))

    # Right side
    mn, mx = get_bounds("mi", ranges_data)
    with colRight:
        vals["mi"]          = int_input  (INPUT_LABELS['MI'],           mn, mx, mn,           rng_help("mi", ranges_data))

    # D via dropdown → numeric
    vals["D"] = D_VALS[st.selectbox(INPUT_LABELS['D_VAL'], list(D_VALS.keys()),
                                    help=rng_help("D", ranges_data))]

    mn, mx = get_bounds("PoissonsRatio", ranges_data)
    with colRight:
        vals["PoissonsRatio"] = float_input(INPUT_LABELS['PR'], mn, mx, mn, 0.01, "%.2f", rng_help("PoissonsRatio", ranges_data))
    mn, mx = get_bounds("E", ranges_data)
    with colRight:
        vals["E"]             = float_input(INPUT_LABELS['YM'], mn, mx, mn, 0.1, "%.1f", rng_help("E", ranges_data))
    mn, mx = get_bounds("Density", ranges_data)
    with colRight:
        vals["Density"]       = float_input(INPUT_LABELS['DEN'], mn, mx, mn, 0.01, "%.2f", rng_help("Density", ranges_data))

    # IMPORTANT: row in the model's declared training order
    x_row = [vals[n] for n in feature_names]
    return vals, x_row

def predict_one(model, scaler_X, scaler_y, row_vals):
    X = np.array(row_vals, dtype=float).reshape(1, -1)
    Xs = scaler_X.transform(X)
    y_scaled = model.predict(Xs).reshape(-1, 1)
    y = scaler_y.inverse_transform(y_scaled).ravel()
    return float(y[0])

def bounds_warning(values_dict, ranges_data):
    if not ranges_data:
        return
    warn = []
    for k, v in values_dict.items():
        if k in ranges_data.get("ranges", {}):
            r = ranges_data["ranges"][k]
            if v < r["min"] or v > r["max"]:
                warn.append(f"{k}: {v:.4f} outside [{r['min']:.4f}, {r['max']:.4f}]")
    if warn:
        st.warning("Some inputs are outside training ranges:\n\n- " + "\n- ".join(warn))

# ----------------------------
# App
# ----------------------------
st.title("🧩GeoRockSlope")

# ABOUT + PERFORMANCE (visible guidance for users)
with st.expander("About & model performance (test set)", expanded=True):
    st.markdown(
        """
**GeoRockSlope** estimates Factor of Safety (FoS) and Seismic-FoS using Artificial Neural Networks optimized by three metaheuristics:
**Artificial Bee Colony (ABC)**, **Ant Colony Optimization (ACOR)**, and **Genetic Algorithm (GA)**.
    
**Performance (R² / RMSE):**
- **FoS** — ABC-ANN **0.9376 / 0.3179** (highest R²), ACOR-ANN **0.9316 / 0.3113** (lowest error), GA-ANN **0.9301 / 0.3149**, Baseline ANN 0.9019 / 0.3985.
- **Seismic-FoS** — **GA-ANN 0.9178 / 0.2513** (best overall), ACOR-ANN 0.8758 / 0.2890, ABC-ANN 0.8983 / 0.2995, Baseline ANN 0.7864 / 0.4340.

**Guidance:**  
For **static FoS**, ABC-ANN yields the highest R² while ACOR-ANN achieves the smallest error; both are recommended.  
For **Seismic-FoS**, **GA-ANN** is the most accurate overall.
        """
    )

ranges = load_ranges()
models = load_manifest()
if not models:
    st.error("No models found. Each folder in 'models/' must have model.joblib, scaler_X.joblib, scaler_y.joblib (or provide models/models_manifest.json).")
    st.caption(f"Looking in: {MODELS_DIR}")
    st.stop()

choices = {m["name"]: m for m in models}
chosen = st.selectbox(INPUT_LABELS['MODEL'], list(choices.keys()))
entry = choices[chosen]

model, scaler_X, scaler_y = load_artifacts(entry)
feature_names = entry.get("feature_names", FEATURE_ORDER)
target_name = entry.get("target_name", "FoS")

with st.expander("Model details", expanded=False):
    st.json({
        "id": entry.get("id"),
        "name": entry.get("name"),
        "paths": {
            "model": entry.get("model_path"),
            "scaler_X": entry.get("scaler_X_path"),
            "scaler_y": entry.get("scaler_y_path"),
        },
        "target_name": target_name,
        "feature_names": feature_names
    })

# Render inputs & predict
values, x_row = render_inputs(feature_names, ranges)

# Optional: quick overview of ranges for all features
with st.expander("Training ranges (min – max)", expanded=False):
    for key in FEATURE_ORDER:
        mn, mx = get_bounds(key, ranges)
        unit = UNITS.get(key, "")
        st.caption(f"- **{key}**: {mn:g}–{mx:g}{unit}")

# Predict button
if hasattr(scaler_X, "n_features_in_") and scaler_X.n_features_in_ != len(x_row):
    st.error(f"Feature count mismatch: scaler expects {scaler_X.n_features_in_}, got {len(x_row)}")
else:
    if st.button(f"Predict {target_name}"):
        try:
            bounds_warning(values, ranges)
            y = predict_one(model, scaler_X, scaler_y, x_row)
            st.success(f"✅ Predicted {target_name}: **{y:.4f}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# NOTE: CSV batch upload removed
