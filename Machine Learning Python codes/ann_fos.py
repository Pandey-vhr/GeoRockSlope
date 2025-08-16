# %%
# ===== SECTION 1: SETUP, DATA, MODEL, TRAINING (DETERMINISTIC) =====
import os
os.environ["PYTHONHASHSEED"] = "0"   # stable hashing

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import scipy.stats as stats

# --- Reproducibility ---
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
# Make PyTorch more deterministic (safe for CPU; on GPU may reduce perf slightly)
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --- Paths ---
csv_path = r"C:\Users\vishn\OneDrive\Desktop\ML01\Vishnu_phd.csv"
output_dir = r"C:\Users\vishn\OneDrive\Desktop\ML01\ANN_fos"
os.makedirs(output_dir, exist_ok=True)

# --- Load data ---
df = pd.read_csv(csv_path)
X = df.iloc[:, :9].values
y = df['FoS'].values.reshape(-1, 1)

# --- Scale (fit on full since no hyperparam tuning here) ---
scaler_X = StandardScaler().fit(X)
scaler_y = StandardScaler().fit(y)
X_scaled = scaler_X.transform(X)
y_scaled = scaler_y.transform(y)

# --- Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=SEED
)

# --- To tensors (CPU) ---
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor  = torch.tensor(X_test,  dtype=torch.float32)
y_test_tensor  = torch.tensor(y_test,  dtype=torch.float32)

# --- Model ---
class ANNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(9, 64), nn.ReLU(),
            nn.Linear(64, 25), nn.ReLU(),
            nn.Linear(25, 16), nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x): return self.model(x)

model = ANNModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- Train ---
EPOCHS = 300
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(X_train_tensor)
    loss = criterion(output, y_train_tensor)
    loss.backward()
    optimizer.step()


# %%
# ===== SECTION 2: EVALUATION, SAVES, PUBLICATION-READY PLOTS =====
from sklearn.metrics import mean_squared_error, mean_absolute_error
import scipy.stats as stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch

def evaluate(model, X_tensor, y_tensor, scaler_y):
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_tensor).detach().cpu().numpy()
        y_true_scaled = y_tensor.detach().cpu().numpy()
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        y_true = scaler_y.inverse_transform(y_true_scaled)
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    slope, intercept, r_val, p_val, std_err = stats.linregress(
        y_true.flatten(), y_pred.flatten()
    )
    return {
        "R2": r2, "MSE": mse, "RMSE": rmse, "MAE": mae, "p-value": p_val,
        "y_true": y_true, "y_pred": y_pred
    }

# --- Metrics ---
train_metrics = evaluate(model, X_train_tensor, y_train_tensor, scaler_y)
test_metrics  = evaluate(model,  X_test_tensor,  y_test_tensor,  scaler_y)

# --- Save metrics ---
with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
    f.write("Training Metrics:\n")
    for k, v in train_metrics.items():
        if k not in ["y_true", "y_pred"]:
            f.write(f"{k:10}: {v:.6f}\n")
    f.write("\nTest Metrics:\n")
    for k, v in test_metrics.items():
        if k not in ["y_true", "y_pred"]:
            f.write(f"{k:10}: {v:.6f}\n")

# --- Save predictions CSVs ---
train_df = pd.DataFrame({
    'Actual_FoS': train_metrics["y_true"].flatten(),
    'Predicted_FoS': train_metrics["y_pred"].flatten(),
    'Set': 'Train'
})
test_df = pd.DataFrame({
    'Actual_FoS': test_metrics["y_true"].flatten(),
    'Predicted_FoS': test_metrics["y_pred"].flatten(),
    'Set': 'Test'
})
pd.concat([train_df, test_df], ignore_index=True)\
  .to_csv(os.path.join(output_dir, "predictions_split.csv"), index=False)

# --- Save model bundle ---
torch.save({
    'model_state_dict': model.state_dict(),
    'scaler_X': scaler_X,
    'scaler_y': scaler_y
}, os.path.join(output_dir, "ann_fos_full_model.pth"))

# --- Plot settings (PNG only) ---
hq_dir = os.path.join(output_dir, "d_ann")
os.makedirs(hq_dir, exist_ok=True)

plt.rcParams.update({
    'figure.dpi': 300, 'savefig.dpi': 300,
    'font.size': 14, 'axes.labelsize': 14, 'axes.titlesize': 16,
    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12,
    'lines.linewidth': 2, 'lines.markersize': 6, 'grid.alpha': 0.5,
})

def save_png(path):
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()

# --- Scatter helper (unchanged; default blue points, red 1:1 line) ---
def scatter_ap(actual, pred, title, fname, xlabel="Actual FoS", ylabel="Predicted FoS"):
    r2  = r2_score(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    plt.figure(figsize=(7.2, 6.2))
    plt.scatter(actual, pred, alpha=0.7, s=40)  # default blue
    mn = min(np.min(actual), np.min(pred))
    mx = max(np.max(actual), np.max(pred))
    pad = 0.02*(mx - mn) if mx > mn else 1.0
    lims = [mn - pad, mx + pad]
    plt.plot(lims, lims, linestyle='--', color='red')
    plt.xlim(lims); plt.ylim(lims)
    ax = plt.gca(); ax.set_aspect('equal', adjustable='box')
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.title(title); plt.grid(True)
    plt.text(0.02, 0.98, f"$R^2$ = {r2:.4f}\nRMSE = {rmse:.4f}",
             transform=ax.transAxes, va='top', ha='left',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    save_png(os.path.join(hq_dir, fname))

# --- Scatter plots ---
scatter_ap(train_df['Actual_FoS'].values, train_df['Predicted_FoS'].values,
           "Train: Actual vs Predicted FoS", "train_scatter_r2.png")
scatter_ap(test_df['Actual_FoS'].values,  test_df['Predicted_FoS'].values,
           "Test:  Actual vs Predicted FoS", "test_scatter_r2.png")

# --- LINE PLOTS (ONLY these use blue/orange) ---
def plot_line(actual, predicted, set_name, fname, ylabel="FoS"):
    idx = np.arange(len(actual))
    plt.figure(figsize=(12, 5))
    plt.plot(idx, actual,    label=f'Actual {set_name}',
             color='#1f77b4', linewidth=2)   # Blue
    plt.plot(idx, predicted, label=f'Predicted {set_name}',
             color='#ff7f0e', linewidth=2)   # Orange
    plt.xlabel("Sample Index"); plt.ylabel(ylabel)
    plt.title(f"Actual vs Predicted {set_name} ({ylabel})")
    plt.legend(); plt.grid(True)
    save_png(os.path.join(hq_dir, fname))

# Train/Test line plots (blue/orange)
plot_line(train_df['Actual_FoS'].values,
          train_df['Predicted_FoS'].values,
          "Train", "train_line_plot.png", ylabel="FoS")
plot_line(test_df['Actual_FoS'].values,
          test_df['Predicted_FoS'].values,
          "Test", "test_line_plot.png",  ylabel="FoS")

print("✅ Metrics, CSVs, and plots saved.")
print("   Output dir:", output_dir)
print("   Figures in:", hq_dir)


