import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import json, math

st.set_page_config(page_title="🧩 GeoRockSlope", page_icon="🪨", layout="centered")

# ----------------------------
# Paths (robust, relative to this file)
# ----------------------------
BASE = Path(__file__).parent.resolve()
MODELS_DIR = BASE / "models"
MANIFEST_PATH = MODELS_DIR / "models_manifest.json"
RANGES_PATH = BASE / "training_ranges.json"

# ----------------------------
# Constants
# ----------------------------
FEATURE_ORDER = ["SlopeHeight","SlopeAngle","UCS","GSI","mi","D","PoissonsRatio","E","Density"]

# Your original labels/help (spelling preserved)
D_VALS = {
    'Moderately Disturbed Rock Mass': 0.7,
    'Very Disturbed Rock Mass': 1.0,
}
HELP_DESCRIPTIONS = {
    'SLOPE_HEIGHT': 'Value Range: 13m to 74m',
    'SLOPE_ANGLE' : 'Value Range: 55° to 84°',
    'UCS'         : 'Value Range: 42MPa to 87MPa',
    'GSI'         : 'Value Range: 25 to 85',
    'MI'          : 'Value Range: 23 to 35',
    'D_VAL'       : '',
    'PR'          : 'Value Range: 0.15 to 0.22',
    'YM'          : 'Value Range: 8783 to 36123',
    'DEN'         : 'Value Range: 2.55g/cm³ to 2.75g/cm³'
}
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

# Default numeric bounds (used if training_ranges.json is missing)
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

@st.cache_resource
def load_manifest():
    # 1) Try explicit manifest first
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

    # 2) Auto-discover: subfolders with model+scalers
    entries = []
    if MODELS_DIR.exists():
        for sub in sorted(p for p in MODELS_DIR.iterdir() if p.is_dir()):
            m, sx, sy = sub/"model.joblib", sub/"scaler_X.joblib", sub/"scaler_y.joblib"
            if m.exists() and sx.exists() and sy.exists():
                name = sub.name.replace("_", " ").upper()
                target = "Seismic FoS" if ("sf" in sub.name.lower() or "seismic" in sub.name.lower()) else "FoS"
                entries.append({
                    "id": sub.name,
                    "name": name,
                    "model_path": str(m),
                    "scaler_X_path": str(sx),
                    "scaler_y_path": str(sy),
                    "target_name": target,
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
# UI helpers
# ----------------------------
def get_bounds(name, ranges_data):
    """Get bounds from training_ranges.json if present; else fallback to defaults."""
    if ranges_data and "ranges" in ranges_data and name in ranges_data["ranges"]:
        r = ranges_data["ranges"][name]
        return float(r["min"]), float(r["max"])
    return DEFAULT_BOUNDS.get(name, (None, None))

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
        vals["SlopeHeight"] = float_input(INPUT_LABELS['SLOPE_HEIGHT'], mn, mx, mn, 0.1, "%.1f", HELP_DESCRIPTIONS['SLOPE_HEIGHT'])
    mn, mx = get_bounds("SlopeAngle", ranges_data)
    with colLeft:
        vals["SlopeAngle"]  = float_input(INPUT_LABELS['SLOPE_ANGLE'],  mn, mx, mn, 0.1, "%.1f", HELP_DESCRIPTIONS['SLOPE_ANGLE'])
    mn, mx = get_bounds("UCS", ranges_data)
    with colLeft:
        vals["UCS"]         = float_input(INPUT_LABELS['UCS'],          mn, mx, mn, 0.1, "%.1f", HELP_DESCRIPTIONS['UCS'])
    mn, mx = get_bounds("GSI", ranges_data)
    with colLeft:
        vals["GSI"]         = int_input  (INPUT_LABELS['GSI'],          mn, mx, mn,           HELP_DESCRIPTIONS['GSI'])

    # Right side
    mn, mx = get_bounds("mi", ranges_data)
    with colRight:
        vals["mi"]          = int_input  (INPUT_LABELS['MI'],           mn, mx, mn,           HELP_DESCRIPTIONS['MI'])

    # D via dropdown → numeric
    vals["D"] = D_VALS[st.selectbox(INPUT_LABELS['D_VAL'], list(D_VALS.keys()),
                                    help=HELP_DESCRIPTIONS['D_VAL'])]

    mn, mx = get_bounds("PoissonsRatio", ranges_data)
    with colRight:
        vals["PoissonsRatio"] = float_input(INPUT_LABELS['PR'], mn, mx, mn, 0.01, "%.2f", HELP_DESCRIPTIONS['PR'])
    mn, mx = get_bounds("E", ranges_data)
    with colRight:
        vals["E"]             = float_input(INPUT_LABELS['YM'], mn, mx, mn, 0.1, "%.1f", HELP_DESCRIPTIONS['YM'])
    mn, mx = get_bounds("Density", ranges_data)
    with colRight:
        vals["Density"]       = float_input(INPUT_LABELS['DEN'], mn, mx, mn, 0.01, "%.2f", HELP_DESCRIPTIONS['DEN'])

    # row in exact training order
    x_row = [vals[n] for n in FEATURE_ORDER]
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
st.write("A machine learning-powered FoS / Seismic-FoS prediction tool integrating finite element analysis and the Generalized Hoek-Brown failure criterion.")

ranges = load_ranges()
models = load_manifest()
if not models:
    st.error("No models found. Make sure each folder in 'models/' has model.joblib, scaler_X.joblib, scaler_y.joblib (or provide models/models_manifest.json).")
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

if st.button(f"Predict {target_name}"):
    try:
        bounds_warning(values, ranges)
        y = predict_one(model, scaler_X, scaler_y, x_row)
        st.success(f"✅ Predicted {target_name}: **{y:.4f}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Batch CSV
st.markdown("---")
st.subheader("Batch prediction (CSV)")
st.write("Upload a CSV with the 9 feature columns (any order is ok if headers match).")
up = st.file_uploader("CSV file", type=["csv"])
if up is not None:
    dfu = pd.read_csv(up)
    try:
        cols = feature_names
        if all(c in dfu.columns for c in cols):
            X = dfu[cols].values
        else:
            X = dfu.values
        Xs = scaler_X.transform(X)
        y_scaled = model.predict(Xs).reshape(-1, 1)
        y = scaler_y.inverse_transform(y_scaled).ravel()
        out = dfu.copy()
        out[target_name] = y
        st.success("Predictions computed.")
        st.dataframe(out.head())
        st.download_button("Download CSV with predictions",
                           data=out.to_csv(index=False).encode("utf-8"),
                           file_name=f"predictions_{entry.get('id','model')}.csv",
                           mime="text/csv")
    except Exception as e:
        st.error(f"Failed to predict: {e}")
