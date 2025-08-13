import streamlit as st
import numpy as np
import joblib
from pathlib import Path
import json, base64

st.set_page_config(page_title="🧩 GeoRockSlope", page_icon="🪨", layout="centered")

# ----------------------------
# Background image (optional)
# ----------------------------
BASE = Path(__file__).parent.resolve()

def set_background(image_path: str = "assets/bg.jpg"):
    """Use a local image (recommended) or an external URL as app background."""
    p = BASE / image_path
    if p.exists():
        data = base64.b64encode(p.read_bytes()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{data}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}
            .block-container {{
                background-color: rgba(255,255,255,0.78);
                border-radius: 12px;
                padding: 1.2rem 1.4rem;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        # Treat as URL fallback
        st.markdown(
            f"""
            <style>
            .stApp {{
                background: url("{image_path}") center/cover no-repeat fixed;
            }}
            .block-container {{
                background-color: rgba(255,255,255,0.78);
                border-radius: 12px;
                padding: 1.2rem 1.4rem;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

# Call this with your image path or URL (comment out if you don't want a background)
set_background("assets/bg.jpg")

# ----------------------------
# Paths & constants
# ----------------------------
MODELS_DIR = BASE / "models"
MANIFEST_PATH = MODELS_DIR / "models_manifest.json"
RANGES_PATH = BASE / "training_ranges.json"

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

# ----------------------------
# Data loading & discovery
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
    # 1) Respect explicit manifest if present
    if MANIFEST_PATH.exists():
        try:
            data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "models" in data:
                return data["models"]
            if isinstance(data, list):
                return data
        except Exception as e:
            st.warning(f"Manifest read error: {e}")

    # 2) Auto-discover (includes acor_ann_f / acor_ann_sf)
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
# Input helpers (inline range help)
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

def int_input(label, mn, mx, val, help_txt):
    return st.number_input(label, min_value=int(mn), max_value=int(mx),
                           value=int(val), step=1, format="%d", help=help_txt)

def float_input(label, mn, mx, val, step, fmt, help_txt):
    return st.number_input(label, min_value=float(mn), max_value=float(mx),
                           value=float(val), step=float(step), format=fmt, help=help_txt)

def render_inputs(feature_names, ranges_data):
    colLeft, colRight = st.columns(2)
    vals = {}
    # Left
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
    # Right
    mn, mx = get_bounds("mi", ranges_data)
    with colRight:
        vals["mi"]          = int_input  (INPUT_LABELS['MI'],           mn, mx, mn,           rng_help("mi", ranges_data))
    vals["D"] = D_VALS[st.selectbox(INPUT_LABELS['D_VAL'], list(D_VALS.keys()), help=rng_help("D", ranges_data))]
    mn, mx = get_bounds("PoissonsRatio", ranges_data)
    with colRight:
        vals["PoissonsRatio"] = float_input(INPUT_LABELS['PR'], mn, mx, mn, 0.01, "%.2f", rng_help("PoissonsRatio", ranges_data))
    mn, mx = get_bounds("E", ranges_data)
    with colRight:
        vals["E"]             = float_input(INPUT_LABELS['YM'], mn, mx, mn, 0.1, "%.1f", rng_help("E", ranges_data))
    mn, mx = get_bounds("Density", ranges_data)
    with colRight:
        vals["Density"]       = float_input(INPUT_LABELS['DEN'], mn, mx, mn, 0.01, "%.2f", rng_help("Density", ranges_data))

    x_row = [vals[n] for n in feature_names]  # ensure model’s own feature order
    return vals, x_row

def predict_one(model, scaler_X, scaler_y, row_vals):
    X = np.array(row_vals,_
