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
        'model': r"models/ga_ann_f/model.joblib",
        'scaler_X': r"models/genetic/scaler_X.joblib",
        'scaler_Y': r"models/genetic/scaler_Y.joblib",
    },
    'GENETIC_SEISMIC': {
        'model': '',
        'scaler': ''
    }
}

D_VALS = {
    'Moderately Disturbed Rock Mass': 0.7,
    'Very Disturbed Rock Mass': 1
}

MODEL_LABELS = {
    'Artificial Bee Colony': 'ABC_REGULAR',
    'Artificial Bee Colony (Seismic)': 'ABC_SEISMIC',
    'Genetic Algorithm': 'GENETIC_REGULAR',
    'Genetic Algorithm (Seismic)': 'GENETIC_SEISMIC'
}

HELP_DESCRIPTONS = {
    'SLOPE_HEIGHT': 'Value Range: 13m to 74m',
    'SLOPE_ANGLE': 'Value Range: 55° to 84°',
    'UCS': 'Value Range: 42MPa to 87MPa',
    'GSI': 'Value Range: 25 to 85',
    'MI': 'Value Range: 23 to 35',
    'D_VAL': '',
    'PR': 'Value Range: 0.15 to 0.22',
    'YM': 'Value Range: 8783 to 36123',
    'DEN': 'Value Range: 2.55g/cm³ to 2.75g/cm³'
}

# Feature name labels
INPUT_LABELS = {
    'MODEL': 'Prediction Model',
    'SLOPE_HEIGHT': 'Slope Height',
    'SLOPE_ANGLE': 'Slope Angle',
    'UCS': 'Uniaxial Compressio Strength',
    'GSI': 'Geological Strength Index',
    'MI': 'Material Constant (mi)',
    'D_VAL': 'Disturbance Factor',
    'PR': 'Poissons Ratio',
    'YM': 'Youngs Modulus (E) of Intact Rock',
    'DEN': 'Density'
}

input_values = []

# === Streamlit App Interface
st.title("🧩GeoRockSlope")
st.write("A machine learning-powered FoS prediction tool integrating finite element analysis and the Generalized Hoek-Brown failure criterion.")

# Prepare input form
selected_model = MODEL_LABELS[st.selectbox(INPUT_LABELS['MODEL'], MODEL_LABELS.keys())]

colLeft, colRight = st.columns(2)

with colLeft:
    input_values.append(st.number_input(INPUT_LABELS['SLOPE_HEIGHT'], min_value=13.0, max_value=74.0, value="min", step=0.1, format="%.1f", help=HELP_DESCRIPTONS['SLOPE_HEIGHT']))
    input_values.append(st.number_input(INPUT_LABELS['SLOPE_ANGLE'], min_value=55.0, max_value=84.0, value="min", step=0.1, format="%.1f", help=HELP_DESCRIPTONS['SLOPE_ANGLE']))
    input_values.append(st.number_input(INPUT_LABELS['UCS'], min_value=42.0, max_value=87.0, value="min", step=0.1, format="%.1f", help=HELP_DESCRIPTONS['UCS']))
    input_values.append(st.number_input(INPUT_LABELS['GSI'], min_value=25, max_value=85, value="min", step=1, format="%i", help=HELP_DESCRIPTONS['GSI']))

with colRight:
    input_values.append(st.number_input(INPUT_LABELS['MI'], min_value=25, max_value=35, value="min", step=1, format="%i", help=HELP_DESCRIPTONS['MI']))

input_values.append(D_VALS[st.selectbox(INPUT_LABELS['D_VAL'], D_VALS.keys(), help=HELP_DESCRIPTONS['D_VAL'])])

with colRight:
    input_values.append(st.number_input(INPUT_LABELS['PR'], min_value=0.15, max_value=0.22, value="min", step=0.01, format="%.2f", help=HELP_DESCRIPTONS['PR']))
    input_values.append(st.number_input(INPUT_LABELS['YM'], min_value=8783.0, max_value=36123.0, value="min", step=0.1, format="%.1f", help=HELP_DESCRIPTONS['YM']))
    input_values.append(st.number_input(INPUT_LABELS['DEN'], min_value=2.55, max_value=2.75, value="min", step=0.01, format="%.2f", help=HELP_DESCRIPTONS['DEN']))


# checkpoint = torch.load(MODELS[selected_model]['model'])
# input_dim = checkpoint['input_dim']
# model = ANN(input_dim, checkpoint['h1'], checkpoint['h2'], checkpoint['h3'])
# model.load_state_dict(checkpoint['model_state_dict'])
# model.eval()
# scaler = joblib.load(MODELS[selected_model]['scaler'])

# New Version
model = joblib.load(MODELS[selected_model]['model'])
scaler_x = joblib.load(MODELS[selected_model]['scaler_x'])
scaler_y = joblib.load(MODELS[selected_model]['scaler_y'])

if st.button("Predict Factor of Safety"):
    # === Preprocess, predict, inverse scale ===
    input_array = np.array(input_values).reshape(1, -1)
    X_scaled = scaler_x.transform(input_array)
    y_pred_scaled = model.predict(X_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
    
    st.success(f"✅ Predicted Factor of Safety (FoS): **{y_pred[0,0]:.4f}**")

    
    # input_scaled = scaler.transform(input_array)
    # input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

    # with torch.no_grad():
    #     prediction = model(input_tensor).item()
    
    # st.success(f"✅ Predicted Factor of Safety (FoS): **{prediction:.4f}**")
