import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import joblib

# === ANN Model Class
class ANN(nn.Module):
    def __init__(self, input_dim, h1, h2, h3):
        super(ANN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, h3), nn.ReLU(),
            nn.Linear(h3, 1)
        )
    def forward(self, x):
        return self.model(x)

# === Load model and scaler
MODELS = {
    'ABC_REGULAR': {
        'model': r"models/artificialBeeColony/regular/model.pth",
        'scaler': r"models/artificialBeeColony/regular/scaler.pkl"
    },
    'ABC_SEISMIC': {
        'model': '',
        'scaler': ''
    },
    'GENETIC_REGULAR': {
        'model': r"models/genetic/regular/model.pth",
        'scaler': r"models/genetic/regular/model.pth",
    },
    'GENETIC_SEISMIC': {
        'model': '',
        'scaler': ''
    }
}

D_VALS = {
    'Moderately Disturbed Rock Mass': 0.7,
    'Undisturbed Rock Mass': 1
}

MODEL_LABELS = {
    'Artificial Bee Colony': 'ABC_REGULAR',
    'Artificial Bee Colony (Seismic)': 'ABC_SEISMIC',
    'Genetic Algorithm': 'GENETIC_REGULAR',
    'Genetic Algorithm (Seismic)': 'GENETIC_SEISMIC'
}

# Feature name labels
MODEL_INPUT_LABEL = 'Prediction Model'
SLOPE_HEIGHT_INPUT_LABEL = 'Slope Height'
SLOPE_ANGLE_INPUT_LABEL = 'Slope Angle'
UCS_INPUT_LABEL = 'Uniaxial Compressio Strength'
GSI_INPUT_LABEL = 'Geological Strength Index'
MI_INPUT_LABEL = 'Material Constant (mi)'
D_VAL_INPUT_LABEL = 'Disturbance Factor'
PR_INPUT_LABEL = 'Poissons Ratio'
YM_INPUT_LABEL = 'Youngs Modulus (E) of Intact Rock'
DEN_INPUT_LABEL = 'Density'

input_values = []

# === Streamlit App Interface
st.title("🧠 Factor of Safety Prediction")
st.write("Please provide the following to predict the Factor of Safety (FoS):")

# Prepare input form
selected_model = MODEL_LABELS[st.selectbox(MODEL_INPUT_LABEL, MODEL_LABELS.keys())]

colLeft, colRight = st.columns(2)

with colLeft:
    input_values.append(st.number_input(SLOPE_HEIGHT_INPUT_LABEL, min_value=13.0, max_value=74.0, value="min", step=0.1, format="%.1f"))
    input_values.append(st.number_input(SLOPE_ANGLE_INPUT_LABEL, min_value=55.0, max_value=84.0, value="min", step=0.1, format="%.1f"))
    input_values.append(st.number_input(UCS_INPUT_LABEL, min_value=42.0, max_value=87.0, value="min", step=0.1, format="%.1f"))
    input_values.append(st.number_input(GSI_INPUT_LABEL, min_value=25, max_value=85, value="min", step=1, format="%i"))

with colRight:
    input_values.append(st.number_input(MI_INPUT_LABEL, min_value=25, max_value=35, value="min", step=1, format="%i"))

input_values.append(D_VALS[st.selectbox(D_VAL_INPUT_LABEL, D_VALS.keys(), help='Disturbance Value')])

with colRight:
    input_values.append(st.number_input(PR_INPUT_LABEL, min_value=0.15, max_value=0.21, value="min", step=0.01, format="%.2f"))
    input_values.append(st.number_input(YM_INPUT_LABEL, min_value=8783.0, max_value=36123.0, value="min", step=0.1, format="%.1f"))
    input_values.append(st.number_input(DEN_INPUT_LABEL, min_value=2.55, max_value=2.75, value="min", step=0.01, format="%.2f"))


checkpoint = torch.load(MODELS[selected_model]['model'])
input_dim = checkpoint['input_dim']
model = ANN(input_dim, checkpoint['h1'], checkpoint['h2'], checkpoint['h3'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

scaler = joblib.load(MODELS[selected_model]['scaler'])

if st.button("Predict Factor of Safety"):
    input_array = np.array(input_values).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

    with torch.no_grad():
        prediction = model(input_tensor).item()
    
    st.success(f"✅ Predicted Factor of Safety (FoS): **{prediction:.4f}**")
