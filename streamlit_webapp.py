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
    'ABC': {
        'model': r"models/artificialBeeColony/model.pth",
        'scaler': r"models/artificialBeeColony/scaler.pkl"
    },
    'GENETIC': {
        'model': r"models/genetic/model.pth",
        'scaler': r"models/genetic/model.pth",
    }
}

# Feature name labels
MODEL = 'Prediction Model'
SLOPE_HEIGHT = 'Slope Height'
SLOPE_ANGLE = 'Slope Angle'
UCS = 'Uniaxial Compressio Strength'
GSI = 'Geological Strength Index'
MI = 'Material Constant (mi)'
D_VAL = 'Disturbance Factor'
PR = 'Poissons Ratio'
YM = 'Youngs Modulus (E) of Intact Rock'
DEN = 'Density'

input_values = []

# === Streamlit App Interface
st.title("🧠 ABC-ANN Model for FoS Prediction")
st.write("Enter the 9 input parameters to predict the Factor of Safety (FoS):")

# Prepare input form
selected_model = st.selectbox(MODEL, options=MODELS.keys())
selected_seismic = st.selectbox('Seismic Activity', options=['Yes', 'No'])

input_values.append(st.number_input(SLOPE_HEIGHT, min_value=13.0, max_value=74.0, value="min", step=0.1, format="%.1f"))
input_values.append(st.number_input(SLOPE_ANGLE, min_value=55.0, max_value=84.0, value="min", step=0.1, format="%.1f"))
input_values.append(st.number_input(UCS, min_value=42.0, max_value=87.0, value="min", step=0.1, format="%.1f"))
input_values.append(st.number_input(GSI, min_value=25, max_value=85, value="min", step=1, format="%i"))
input_values.append(st.number_input(MI, min_value=25, max_value=35, value="min", step=1, format="%i"))
input_values.append(st.selectbox(D_VAL, [0.7, 1], help='Disturbance Value'))
input_values.append(st.number_input(PR, min_value=0.15, max_value=0.21, value="min", step=0.01, format="%.2f"))
input_values.append(st.number_input(YM, min_value=8783.0, max_value=36123.0, value="min", step=0.1, format="%.1f"))
input_values.append(st.number_input(DEN, min_value=2.55, max_value=2.75, value="min", step=0.01, format="%.2f"))


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
