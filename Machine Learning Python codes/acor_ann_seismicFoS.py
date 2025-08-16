# %%
# ===== SECTION 1: SETUP, DATA, ACOR SEARCH (REPRODUCIBLE) =====
# Reproducibility switches
import os
os.environ["PYTHONHASHSEED"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Imports
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Global seed
SEED = 50
random.seed(SEED)
np.random.seed(SEED)
rng = np.random.default_rng(SEED)

# === CONFIGURATION ===
csv_path   = r"C:\Users\vishn\OneDrive\Desktop\ML01\Vishnu_phd.csv"
output_dir = r"C:\Users\vishn\OneDrive\Desktop\ML01\ACOR-ANN_s"
TARGET_COL = "SeismicFoS"        # <- change to "FoS" if you want FoS
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
K_ARCHIVE   = 25      # archive size (pheromone memory)
ANTS        = 25      # samples per iteration (usually = K_ARCHIVE)
MAX_ITER    = 30
q           = 0.20    # rank spread (smaller = greedier selection)
xi          = 0.90    # kernel width factor (0.5–1.0 common)
H1_BOUNDS   = (8, 128)
H2_BOUNDS   = (8, 128)
H3_BOUNDS   = (0, 128)   # 0 = skip layer 3
LR_BOUNDS   = (1e-4, 1e-2)

def clamp_round(sol):
    """Clamp to bounds and round neuron counts."""
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

# --- Initialize archive (random) ---
archive_X = [random_solution() for _ in range(K_ARCHIVE)]
archive_F = [evaluate_solution(s) for s in archive_X]

# Rank archive (desc by fitness)
def rank_archive(X, F):
    idx = np.argsort(F)[::-1]
    return [X[i] for i in idx], [F[i] for i in idx]

archive_X, archive_F = rank_archive(archive_X, archive_F)

# Rank-based selection weights (ACOR)
def acor_weights(k, q):
    # w_i ∝ exp( - (i-1)^2 / (2 (qk)^2) ), i = 1..k
    i = np.arange(1, k+1)
    denom = 2 * (q * k)**2
    w = np.exp( - (i-1)**2 / denom )
    w = w / np.sum(w)
    return w

weights = acor_weights(K_ARCHIVE, q)

def column_dispersion(values):
    # robust width estimate used by ACOR; mean absolute deviation w.r.t. all archive members
    v = np.asarray(values, dtype=float)
    diffs = np.abs(v.reshape(-1,1) - v.reshape(1,-1))
    # average absolute pairwise difference
    d = np.sum(diffs) / (len(v)*len(v) + 1e-12)
    return d

fitness_history = [max(archive_F)]
global_best_sol = archive_X[0][:]
global_best_fit = archive_F[0]

for it in range(MAX_ITER):
    # Precompute per-dimension dispersions from archive (after ranking)
    cols = list(zip(*archive_X))  # 4 lists (h1,h2,h3,lr)
    S = [column_dispersion(col) for col in cols]
    S = [s if s > 1e-9 else 1.0 for s in S]  # guard tiny dispersion

    new_X = []
    new_F = []

    for _ in range(ANTS):
        # Select an index from archive by rank weights
        idx = rng.choice(K_ARCHIVE, p=weights)

        # For each dimension j, sample from N(mu = x[idx][j], sigma = xi * S_j)
        cand = []
        for j in range(4):
            mu = archive_X[idx][j]
            sigma = xi * S[j]
            xj = rng.normal(mu, sigma)
            cand.append(xj)

        # Repair & round
        cand = clamp_round(cand)

        # Small diversity injection (with low prob change one dim)
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

    # Update archive: merge old + new, keep top K
    merged_X = archive_X + new_X
    merged_F = archive_F + new_F
    archive_X, archive_F = rank_archive(merged_X, merged_F)
    archive_X = archive_X[:K_ARCHIVE]
    archive_F = archive_F[:K_ARCHIVE]

    # Recompute weights (optional; q fixed)
    weights = acor_weights(K_ARCHIVE, q)

    fitness_history.append(global_best_fit)

# === FINAL BEST MODEL ===
best_layers = tuple([n for n in global_best_sol[:3] if n > 0])
best_lr     = global_best_sol[3]
final_model = build_model(global_best_sol)
final_model.fit(X_train, y_train)


# %%
# ===== SECTION 2: METRICS, SAVES, PUBLICATION-READY PLOTS (PNG) =====

# Save model and scalers
joblib.dump(final_model, os.path.join(output_dir, "acor_ann_model.joblib"))
joblib.dump(scaler_X,    os.path.join(output_dir, "scaler_X.joblib"))
joblib.dump(scaler_y,    os.path.join(output_dir, "scaler_y.joblib"))

# Inverse transforms & metrics
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

# Save metrics
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

# Save fitness history & predictions
pd.DataFrame({"iteration": list(range(len(fitness_history))),
              "best_r2": fitness_history}).to_csv(
    os.path.join(output_dir, "fitness_progress.csv"), index=False
)
np.savetxt(os.path.join(output_dir, "train_actual.txt"),    y_train_true)
np.savetxt(os.path.join(output_dir, "train_predicted.txt"), y_train_pred)
np.savetxt(os.path.join(output_dir, "test_actual.txt"),     y_test_true)
np.savetxt(os.path.join(output_dir, "test_predicted.txt"),  y_test_pred)

# High-quality plots (PNG only)
hq_dir = os.path.join(output_dir, "d_acor")
os.makedirs(hq_dir, exist_ok=True)

plt.rcParams.update({
    'figure.dpi': 300, 'savefig.dpi': 300,
    'font.size': 14, 'axes.labelsize': 14, 'axes.titlesize': 16,
    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12,
    'lines.linewidth': 2, 'lines.markersize': 6, 'grid.alpha': 0.5,
})
def save_png(path): plt.tight_layout(); plt.savefig(path, bbox_inches='tight'); plt.close()

# Fitness curve
plt.figure(figsize=(7.5, 5.5))
plt.plot(fitness_history, linewidth=2)
plt.xlabel("Iteration"); plt.ylabel("Best $R^2$ (Test)")
plt.title(f"ACOR Fitness Progress ({TARGET_COL})")
plt.grid(True)
save_png(os.path.join(hq_dir, "acor_fitness_progress.png"))

# Scatter helper (equal axes + embedded stats)
def scatter_actual_vs_pred(y_true, y_pred, title, fname,
                           xlabel=f"Actual {TARGET_COL}", ylabel=f"Predicted {TARGET_COL}", color=None):
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
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.title(title); plt.grid(True)
    plt.text(0.02, 0.98, f"$R^2$ = {r2:.4f}\nRMSE = {rmse:.4f}",
             transform=ax.transAxes, va='top', ha='left',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    save_png(os.path.join(hq_dir, fname))

# Scatter plots (Train=green, Test=blue)
scatter_actual_vs_pred(y_train_true, y_train_pred,
                       f"Train Set: Actual vs Predicted ({TARGET_COL})",
                       "acor_scatter_train.png", color='green')
scatter_actual_vs_pred(y_test_true,  y_test_pred,
                       f"Test Set: Actual vs Predicted ({TARGET_COL})",
                       "acor_scatter_test.png",  color='blue')

# Line plots (UNSORTED; Blue=Actual, Orange=Predicted)
def line_plot(y_true, y_pred, title, fname, ylabel=TARGET_COL):
    idx = np.arange(len(y_true))
    plt.figure(figsize=(12, 5))
    plt.plot(idx, y_true, label="Actual",    color="#1f77b4", linewidth=2)  # blue
    plt.plot(idx, y_pred, label="Predicted", color="#ff7f0e", linewidth=2)  # orange
    plt.title(title); plt.xlabel("Sample Index"); plt.ylabel(ylabel)
    plt.legend(frameon=False); plt.grid(True)
    save_png(os.path.join(hq_dir, fname))

line_plot(y_train_true, y_train_pred, f"Actual vs Predicted (Train, {TARGET_COL})", "acor_line_train.png")
line_plot(y_test_true,  y_test_pred,  f"Actual vs Predicted (Test, {TARGET_COL})",  "acor_line_test.png")

print("\n🎯 ACOR-ANN completed.")
print("   Best Architecture:", best_layers)
print("   Learning Rate:", best_lr)
print("   Train R²: {:.4f}".format(train_metrics['R2']))
print("   Test  R²: {:.4f}".format(test_metrics['R2']))
print("   Outputs:", output_dir)
print("   High-quality figures:", hq_dir)


