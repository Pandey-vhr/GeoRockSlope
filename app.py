# app.py
import base64
import json
from pathlib import Path

import joblib
import numpy as np
import streamlit as st

# ---------- Page setup ----------
st.set_page_config(page_title="🧩 GeoRockSlope", page_icon="🪨", layout="centered")

BASE = Path(__file__).parent.resolve()
MODELS_DIR = BASE / "models"
MANIFEST_PATH = MODELS_DIR / "models_manifest.json"
RANGES_PATH = BASE / "training_ranges.json"

# ---------- Background image ----------
# Choose one of these (local file preferred for speed); order is fallback priority.
BG_IMAGE = "assets/bg_1920x1080.jpg"  # put your resized background here
BG_FALLBACK = "assets/bg.jpg"         # raw logo if you prefer
BG_MODE = "contain"                   # 'contain' (no crop) or 'cover' (fills screen, can crop)

def set_background(image_path: str, mode: str = "contain"):
    """Set a full-page background. mode: 'contain' (no crop) or 'cover' (edge-to-edge)."""
    p = BASE / image_path
    css = None
    if p.exists():
        data = base64.b64encode(p.read_bytes()).decode()
        css = f"""
        <style>
          .stApp {{
            background-image: url("data:image/jpeg;base64,{data}");
            background-size: {mode};
            background-repeat: no-repeat;
            background-position: center top;
            background-attachment: fixed;
            background-color: #f6f7f9;
          }}
          .block-container {{
            background: rgba(255,255,255,.82);
            border-radius: 12px;
            padding: 1.2rem 1.4rem;
          }}
        </style>
        """
    else:
        # treat as URL (or last-resort: light canvas only)
        css = f"""
        <style>
          .stApp {{
            background: url("{image_path}") center top / {mode} no-repeat fixed, #f6f7f9;
          }}
          .block-container {{
            background: rgba(255,255,255,.82);
            border-radius: 12px;
            padding: 1.2rem 1.4rem;
          }}
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)

# Try main, then fallback
if (BASE / BG_IMAGE).exists():
    set_background(BG_IMAGE, BG_MODE)
elif (BASE / BG_FALLBACK).exists():
    set_background(BG_FALLBACK, BG_MODE)
# else: no background applied (safe to proceed)

# ---------- Constants ----------
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

# Fallback training bounds if training_ranges.json is missing
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

# ---------- Loaders ----------
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
    # 1) Use explicit manifest if present
    if MANIFEST_PATH.exists():
        try:
            data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "models" in data:
                return data["models"]
            if isinstance(data, list):
                return data
        except Exception as e:
            st.warning(f"Manifest read error: {e}")

    # 2) Auto-discover subfolders with model+scalers
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

# ---------- UI helpers ----------
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

def int_input(label, mn, mx, val, help_txt):
    return st.number_input(label, min_value=int(mn), max_value=int(mx),
                           value=int(val), step=1, format="%d", help=help_txt)

def float_input(label, mn, mx, val, step, fmt, help_txt):
    return st.number_input(label, min_value=float(mn), max_value=float(mx),
                           value=float(val), step=float(step), format=fmt, help=help_txt)

def render_inputs(feature_names, ranges_data):
    colLeft, colRight = st.columns(2)
    vals = {}

    # Left column
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

    # Right column
    mn, mx = get_bounds("mi", ranges_data)
    with colRight:
        vals["mi"]          = int_input  (INPUT_LABELS['MI'],           mn, mx, mn,           rng_help("mi", ranges_data))
    vals["D"] = D_VALS[st.selectbox(INPUT_LABELS['D_VAL'], list(D_VALS.keys()), help=rng_help("D", ranges_data))]
    mn, mx = get_bounds("PoissonsRatio", ranges_data)
    with colRight:
        eps = 1e-9
        vals["PoissonsRatio"] = st.number_input(..., max_value=float(mx) + eps, step=0.01, format="%.2f", ...)

    mn, mx = get_bounds("E", ranges_data)
    with colRight:
        vals["E"]             = float_input(INPUT_LABELS['YM'], mn, mx, mn, 0.1, "%.1f", rng_help("E", ranges_data))
    mn, mx = get_bounds("Density", ranges_data)
    with colRight:
        vals["Density"]       = float_input(INPUT_LABELS['DEN'], mn, mx, mn, 0.01, "%.2f", rng_help("Density", ranges_data))

    # Order exactly as the model expects
    x_row = [vals[n] for n in feature_names]
    return vals, x_row

def predict_one(model, scaler_X, scaler_y, row_vals):
    X = np.array(row_vals, dtype=float).reshape(1, -1)
    Xs = scaler_X.transform(X)
    y_scaled = model.predict(Xs).reshape(-1, 1)
    y = scaler_y.inverse_transform(y_scaled).ravel()
    return float(y[0])

# ---------- App ----------
st.title("🧩 GeoRockSlope")

# Ultra-brief guidance + dataset note
st.info("Models were trained on results from finite-element analyses of 494 slope models using the Generalized Hoek–Brown criterion. Best models: For FoS prediction is ABC-ANN (Test R² 0.9376, RMSE 0.3179), and for Seismic-FoS is GA-ANN (Test R² 0.9178, RMSE 0.2513)")

ranges = load_ranges()
models = load_manifest()
if not models:
    st.error("No models found. Each folder in 'models/' must contain model.joblib, scaler_X.joblib, scaler_y.joblib (e.g., abc_ann_f, abc_ann_sf, acor_ann_f, acor_ann_sf, ga_ann_f, ga_ann_sf).")
    st.caption(f"Looking in: {MODELS_DIR}")
    st.stop()

choices = {m["name"]: m for m in models}
chosen = st.selectbox(INPUT_LABELS['MODEL'], list(choices.keys()))
entry = choices[chosen]

model, scaler_X, scaler_y = load_artifacts(entry)
feature_names = entry.get("feature_names", FEATURE_ORDER)
target_name = entry.get("target_name", "FoS")

# Optional details
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

# Inputs & prediction
values, x_row = render_inputs(feature_names, ranges)

if hasattr(scaler_X, "n_features_in_") and scaler_X.n_features_in_ != len(x_row):
    st.error(f"Feature count mismatch: scaler expects {scaler_X.n_features_in_}, got {len(x_row)}")
else:
    if st.button(f"Predict {target_name}"):
        try:
            y = predict_one(model, scaler_X, scaler_y, x_row)
            st.success(f"✅ Predicted {target_name}: **{y:.4f}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
