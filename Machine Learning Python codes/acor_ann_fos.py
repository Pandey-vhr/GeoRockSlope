# %%
# ===== SECTION 1: SETUP, DATA, ACOR SEARCH (REPRODUCIBLE) =====
import os
os.environ["PYTHONHASHSEED"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

SEED = 50
random.seed(SEED)
np.random.seed(SEED)
rng = np.random.default_rng(SEED)

# === CONFIGURATION ===
csv_path   = r"C:\Users\vishn\OneDrive\Desktop\ML01\Vishnu_phd.csv"
output_dir = r"C:\Users\vishn\OneDrive\Desktop\ML01\ACOR-ANN_fos"
TARGET_COL = "FoS"
os.makedirs(output_dir, exist_ok=True)

# === LOAD AND PREPROCESS ===
df = pd.read_csv(csv_path)
X = df.iloc[:, :9].values
y = df[TARGET_COL].values.reshape(-1, 1)

scaler_X = StandardScaler().fit(X)
scaler_y = StandardScaler().fit(y)
X_scaled = scaler_X.transform(X)
y_scaled = scaler_y.transform(y).ravel()

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=SEED
)

# === ACOR PARAMETERS ===
K_ARCHIVE   = 25
ANTS        = 25
MAX_ITER    = 30
q           = 0.20
xi          = 0.90
H1_BOUNDS   = (8, 128)
H2_BOUNDS   = (8, 128)
H3_BOUNDS   = (0, 128)
LR_BOUNDS   = (1e-4, 1e-2)

def clamp_round(sol):
    a,b,c,lr = sol
    a = int(np.clip(int(round(a)), *H1_BOUNDS))
    b = int(np.clip(int(round(b)), *H2_BOUNDS))
    c = int(np.clip(int(round(c)), *H3_BOUNDS))
    lr = float(np.clip(lr, *LR_BOUNDS))
    return [a,b,c,lr]

def build_model(sol):
    layers = [n for n in sol[:3] if n > 0]
    lr     = sol[3]
    return MLPRegressor(
        hidden_layer_sizes=tuple(layers),
        learning_rate_init=lr,
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=SEED
    )

def evaluate_solution(sol):
    try:
        model = build_model(sol)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_inv = scaler_y.inverse_transform(y_pred.reshape(-1,1)).ravel()
        y_true_inv = scaler_y.inverse_transform(y_test.reshape(-1,1)).ravel()
        return r2_score(y_true_inv, y_pred_inv)
    except Exception:
        return -np.inf

def random_solution():
    return [
        random.randint(*H1_BOUNDS),
        random.randint(*H2_BOUNDS),
        random.randint(*H3_BOUNDS),
        round(random.uniform(*LR_BOUNDS), 5),
    ]

def rank_archive(X, F):
    idx = np.argsort(F)[::-1]
    return [X[i] for i in idx], [F[i] for i in idx]

def acor_weights(k, q):
    i = np.arange(1, k+1)
    denom = 2 * (q * k)**2
    w = np.exp( - (i-1)**2 / denom )
    w = w / np.sum(w)
    return w

def column_dispersion(values):
    v = np.asarray(values, dtype=float)
    diffs = np.abs(v.reshape(-1,1) - v.reshape(1,-1))
    d = np.sum(diffs) / (len(v)*len(v) + 1e-12)
    return d

# --- Initialize archive ---
archive_X = [random_solution() for _ in range(K_ARCHIVE)]
archive_F = [evaluate_solution(s) for s in archive_X]
archive_X, archive_F = rank_archive(archive_X, archive_F)
weights = acor_weights(K_ARCHIVE, q)

fitness_history = [max(archive_F)]
global_best_sol = archive_X[0][:]
global_best_fit = archive_F[0]

for it in range(MAX_ITER):
    cols = list(zip(*archive_X))
    S = [column_dispersion(col) for col in cols]
    S = [s if s > 1e-9 else 1.0 for s in S]

    new_X, new_F = [], []

    for _ in range(ANTS):
        idx = rng.choice(K_ARCHIVE, p=weights)
        cand = []
        for j in range(4):
            mu = archive_X[idx][j]
            sigma = xi * S[j]
            xj = rng.normal(mu, sigma)
            cand.append(xj)
        cand = clamp_round(cand)

        if random.random() < 0.25:
            d = random.randint(0,3)
            if d < 3:
                bounds = H3_BOUNDS if d == 2 else (H1_BOUNDS if d == 0 else H2_BOUNDS)
                cand[d] = random.randint(bounds[0], bounds[1])
            else:
                cand[d] = round(random.uniform(*LR_BOUNDS), 5)
            cand = clamp_round(cand)

        f = evaluate_solution(cand)
        new_X.append(cand)
        new_F.append(f)
        if f > global_best_fit:
            global_best_fit = f
            global_best_sol = cand[:]

    merged_X = archive_X + new_X
    merged_F = archive_F + new_F
    archive_X, archive_F = rank_archive(merged_X, merged_F)
    archive_X = archive_X[:K_ARCHIVE]
    archive_F = archive_F[:K_ARCHIVE]
    weights = acor_weights(K_ARCHIVE, q)
    fitness_history.append(global_best_fit)

# === FINAL BEST MODEL ===
best_layers = tuple([n for n in global_best_sol[:3] if n > 0])
best_lr     = global_best_sol[3]
final_model = build_model(global_best_sol)
final_model.fit(X_train, y_train)

# ===== SECTION 2: METRICS, PLOTS =====
joblib.dump(final_model, os.path.join(output_dir, "acor_ann_model.joblib"))
joblib.dump(scaler_X,    os.path.join(output_dir, "scaler_X.joblib"))
joblib.dump(scaler_y,    os.path.join(output_dir, "scaler_y.joblib"))

def inverse_y(y_scaled):
    return scaler_y.inverse_transform(np.asarray(y_scaled).reshape(-1, 1)).ravel()

y_train_true = inverse_y(y_train)
y_test_true  = inverse_y(y_test)
y_train_pred = inverse_y(final_model.predict(X_train))
y_test_pred  = inverse_y(final_model.predict(X_test))

def compute_metrics(y_true, y_pred):
    return {
        "R2":   r2_score(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
        "MSE":  float(mean_squared_error(y_true, y_pred)),
    }

train_metrics = compute_metrics(y_train_true, y_train_pred)
test_metrics  = compute_metrics(y_test_true,  y_test_pred)

with open(os.path.join(output_dir, "acor_ann_metrics.txt"), "w", encoding="utf-8") as f:
    f.write(f"ACOR-ANN Model Evaluation ({TARGET_COL})\n")
    f.write("======================================\n")
    f.write(f"Best Architecture: {best_layers}\n")
    f.write(f"Learning Rate: {best_lr}\n\n")
    f.write("Train Metrics:\n")
    for k, v in train_metrics.items():
        f.write(f"  {k}: {v:.6f}\n")
    f.write("\nTest Metrics:\n")
    for k, v in test_metrics.items():
        f.write(f"  {k}: {v:.6f}\n")

pd.DataFrame({"iteration": list(range(len(fitness_history))),
              "best_r2": fitness_history}).to_csv(
    os.path.join(output_dir, "fitness_progress.csv"), index=False
)

hq_dir = os.path.join(output_dir, "d_acor")
os.makedirs(hq_dir, exist_ok=True)
plt.rcParams.update({'figure.dpi': 300, 'savefig.dpi': 300, 'font.size': 14})

def save_png(path): plt.tight_layout(); plt.savefig(path, bbox_inches='tight'); plt.close()

# Fitness curve
plt.figure(figsize=(7.5, 5.5))
plt.plot(fitness_history, linewidth=2)
plt.xlabel("Iteration"); plt.ylabel("Best $R^2$ (Test)")
plt.title(f"ACOR Fitness Progress ({TARGET_COL})")
plt.grid(True)
save_png(os.path.join(hq_dir, "acor_fitness_progress.png"))

# Scatter helper
def scatter_actual_vs_pred(y_true, y_pred, title, fname, color=None):
    r2  = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    plt.figure(figsize=(7.2, 6.2))
    plt.scatter(y_true, y_pred, alpha=0.7, s=40, color=color)
    mn = min(np.min(y_true), np.min(y_pred))
    mx = max(np.max(y_true), np.max(y_pred))
    pad = 0.02 * (mx - mn) if mx > mn else 1.0
    lims = [mn - pad, mx + pad]
    plt.plot(lims, lims, linestyle='--', color='red')
    plt.xlim(lims); plt.ylim(lims)
    ax = plt.gca(); ax.set_aspect('equal', adjustable='box')
    plt.xlabel(f"Actual {TARGET_COL}"); plt.ylabel(f"Predicted {TARGET_COL}")
    plt.title(title); plt.grid(True)
    plt.text(0.02, 0.98, f"$R^2$ = {r2:.4f}\nRMSE = {rmse:.4f}",
             transform=ax.transAxes, va='top', ha='left',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    save_png(os.path.join(hq_dir, fname))

scatter_actual_vs_pred(y_train_true, y_train_pred,
                       f"Train Set: Actual vs Predicted ({TARGET_COL})",
                       "acor_scatter_train.png", color='green')
scatter_actual_vs_pred(y_test_true,  y_test_pred,
                       f"Test Set: Actual vs Predicted ({TARGET_COL})",
                       "acor_scatter_test.png",  color='blue')

# Line plots
def line_plot(y_true, y_pred, title, fname):
    idx = np.arange(len(y_true))
    plt.figure(figsize=(12, 5))
    plt.plot(idx, y_true, label="Actual",    color="#1f77b4", linewidth=2)
    plt.plot(idx, y_pred, label="Predicted", color="#ff7f0e", linewidth=2)
    plt.title(title); plt.xlabel("Sample Index"); plt.ylabel(TARGET_COL)
    plt.legend(frameon=False); plt.grid(True)
    save_png(os.path.join(hq_dir, fname))

line_plot(y_train_true, y_train_pred, f"Actual vs Predicted (Train, {TARGET_COL})", "acor_line_train.png")
line_plot(y_test_true,  y_test_pred,  f"Actual vs Predicted (Test, {TARGET_COL})",  "acor_line_test.png")

print("\n🎯 ACOR-ANN completed for FoS.")
print("   Best Architecture:", best_layers)
print("   Learning Rate:", best_lr)
print("   Train R²: {:.4f}".format(train_metrics['R2']))
print("   Test  R²: {:.4f}".format(test_metrics['R2']))
print("   Outputs:", output_dir)
print("   High-quality figures:", hq_dir)


# %%


