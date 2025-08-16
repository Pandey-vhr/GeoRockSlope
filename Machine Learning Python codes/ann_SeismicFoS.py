# %%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os
import scipy.stats as stats

np.random.seed(42)
torch.manual_seed(42)

csv_path = r"C:\Users\vishn\OneDrive\Desktop\ML01\Vishnu_phd.csv"
output_dir = r"C:\Users\vishn\OneDrive\Desktop\ML01\ANN_sfos"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(csv_path)
X = df.iloc[:, :9].values
y = df['SeismicFoS'].values.reshape(-1, 1)

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

class ANNModel(nn.Module):
    def __init__(self):
        super(ANNModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 35),
            nn.ReLU(),
            nn.Linear(35, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.model(x)

model = ANNModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(300):
    model.train()
    optimizer.zero_grad()
    output = model(X_train_tensor)
    loss = criterion(output, y_train_tensor)
    loss.backward()
    optimizer.step()

def evaluate(model, X_tensor, y_tensor, scaler_y):
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_tensor).numpy()
        y_true_scaled = y_tensor.numpy()
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        y_true = scaler_y.inverse_transform(y_true_scaled)

    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    slope, intercept, r_val, p_val, std_err = stats.linregress(y_true.flatten(), y_pred.flatten())

    return {
        "R2": r2, "MSE": mse, "RMSE": rmse, "MAE": mae, "p-value": p_val,
        "y_true": y_true, "y_pred": y_pred
    }

train_metrics = evaluate(model, X_train_tensor, y_train_tensor, scaler_y)
test_metrics = evaluate(model, X_test_tensor, y_test_tensor, scaler_y)

with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
    f.write("Training Metrics:\n")
    for k, v in train_metrics.items():
        if k not in ["y_true", "y_pred"]:
            f.write(f"{k:10}: {v:.4f}\n")
    f.write("\nTest Metrics:\n")
    for k, v in test_metrics.items():
        if k not in ["y_true", "y_pred"]:
            f.write(f"{k:10}: {v:.4f}\n")




# %%
# ===== SAVE PREDICTIONS + PUBLICATION-READY PLOTS (match above model) =====
# Save predictions (SeismicFoS naming)
train_df = pd.DataFrame({
    'Actual_SeismicFoS':   train_metrics["y_true"].flatten(),
    'Predicted_SeismicFoS':train_metrics["y_pred"].flatten(),
    'Set': 'Train'
})
test_df = pd.DataFrame({
    'Actual_SeismicFoS':   test_metrics["y_true"].flatten(),
    'Predicted_SeismicFoS':test_metrics["y_pred"].flatten(),
    'Set': 'Test'
})
combined_df = pd.concat([train_df, test_df], ignore_index=True)
combined_df.to_csv(os.path.join(output_dir, "predictions_split.csv"), index=False)

# High-DPI plot style
hq_dir = os.path.join(output_dir, "d_ann")
os.makedirs(hq_dir, exist_ok=True)
plt.rcParams.update({
    'figure.dpi': 300, 'savefig.dpi': 300,
    'font.size': 14, 'axes.labelsize': 14, 'axes.titlesize': 16,
    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12,
    'lines.linewidth': 2, 'lines.markersize': 6, 'grid.alpha': 0.5,
})
def save_png(path): plt.tight_layout(); plt.savefig(path, bbox_inches='tight'); plt.close()

# ---------- Scatter plots (Train green, Test blue; red unity; equal axes) ----------
from sklearn.metrics import mean_squared_error
def scatter_ap(actual, pred, title, fname, color):
    r2  = r2_score(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    plt.figure(figsize=(7.2, 6.2))
    plt.scatter(actual, pred, alpha=0.7, s=40, color=color)
    mn, mx = float(min(actual.min(), pred.min())), float(max(actual.max(), pred.max()))
    pad = 0.02*(mx-mn) if mx>mn else 1.0
    lims = [mn - pad, mx + pad]
    plt.plot(lims, lims, linestyle='--', color='red'); plt.xlim(lims); plt.ylim(lims)
    ax = plt.gca(); ax.set_aspect('equal', adjustable='box')
    plt.xlabel("Actual SeismicFoS"); plt.ylabel("Predicted SeismicFoS")
    plt.title(title); plt.grid(True)
    plt.text(0.02, 0.98, f"$R^2$ = {r2:.4f}\nRMSE = {rmse:.4f}",
             transform=ax.transAxes, va='top', ha='left',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    save_png(os.path.join(hq_dir, fname))

scatter_ap(train_df['Actual_SeismicFoS'].values,
           train_df['Predicted_SeismicFoS'].values,
           "Train: Actual vs Predicted SeismicFoS",
           "train_scatter_r2.png", color='green')

scatter_ap(test_df['Actual_SeismicFoS'].values,
           test_df['Predicted_SeismicFoS'].values,
           "Test: Actual vs Predicted SeismicFoS",
           "test_scatter_r2.png", color='blue')

# ---------- Line plots (UNSORTED; Blue=Actual, Orange=Predicted) ----------
def line_plot(actual, pred, set_name, fname, ylabel="SeismicFoS"):
    idx = np.arange(len(actual))
    plt.figure(figsize=(12, 5))
    plt.plot(idx, actual,    label=f"Actual {set_name}",
             color="#1f77b4", linewidth=2)   # blue
    plt.plot(idx, pred,      label=f"Predicted {set_name}",
             color="#ff7f0e", linewidth=2)   # orange
    plt.xlabel("Sample Index"); plt.ylabel(ylabel)
    plt.title(f"Actual vs Predicted {set_name} ({ylabel})")
    plt.legend(); plt.grid(True)
    save_png(os.path.join(hq_dir, fname))

line_plot(train_df['Actual_SeismicFoS'].values,
          train_df['Predicted_SeismicFoS'].values,
          "Train", "train_line_plot.png")

line_plot(test_df['Actual_SeismicFoS'].values,
          test_df['Predicted_SeismicFoS'].values,
          "Test", "test_line_plot.png")

# ---------- Save model for Streamlit/web ----------
torch.save({
    'model_state_dict': model.state_dict(),
    'scaler_X': scaler_X,
    'scaler_y': scaler_y
}, os.path.join(output_dir, "ann_seismic_full_model.pth"))

print("✅ All plots saved in:", hq_dir)
print("✅ Model saved to:", os.path.join(output_dir, "ann_seismic_full_model.pth"))


