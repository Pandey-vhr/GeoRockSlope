# app_with_logo.py
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="GeoRockSlope", page_icon="🪨", layout="centered")

BASE = Path(__file__).parent.resolve()
LOGO_PATH = BASE / "GECL.png"   # keep GECL.png beside this file

def header_with_logo(title: str = "GeoRockSlope", logo_width: int = 96):
    # Header row with title (left) and logo (right)
    col1, col2 = st.columns([0.75, 0.25])
    with col1:
        st.markdown(f"<h1 style='margin:0'>{title}</h1>", unsafe_allow_html=True)
    with col2:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=logo_width, caption=None)
        else:
            st.warning(f"Logo not found at: {LOGO_PATH}")

# ---- Page ----
header_with_logo()

st.write("This is a minimal example. Your existing controls and logic can follow below.")
st.write("To use in your own app, copy the header_with_logo() function and the header_with_logo() call near the top of your script.")
st.write("Ensure GECL.png is in the same folder as app.py. Adjust the column ratio or logo_width as desired.")
