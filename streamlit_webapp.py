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
model_path = r"C:\Users\vishn\OneDrive\Documents\Machine Learning\ABC_ANN\ABC_ANN_FoS_BestModel.pth"
scaler_path = r"C:\Users\vishn\OneDrive\Documents\Machine Learning\ABC_ANN\ABC_ANN_FoS_Scaler.pkl"

checkpoint = torch.load(model_path)
input_dim = checkpoint['input_dim']
model = ANN(input_dim, checkpoint['h1'], checkpoint['h2'], checkpoint['h3'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

scaler = joblib.load(scaler_path)

# === Streamlit App Interface
st.title("🧠 ABC-ANN Model for FoS Prediction")
st.write("Enter the 9 input parameters to predict the Factor of Safety (FoS):")

# Feature names from your dataset
feature_names = [
    "SlopeHeight",
    "SlopeAngle",
    "UCS",
    "GSI",
    "mi",
    "D",
    "PoissonsRatio",
    "E",
    "Density"
]

input_values = []
for name in feature_names:
    val = st.number_input(f"{name}", value=0.0, step=0.1, format="%.4f")
    input_values.append(val)




if st.button("Predict FoS"):
    input_array = np.array(input_values).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

    with torch.no_grad():
        prediction = model(input_tensor).item()
    
    st.success(f"✅ Predicted Factor of Safety (FoS): **{prediction:.4f}**")
