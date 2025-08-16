# %%
# ===== SECTION 1: SETUP, DATA, ABC SEARCH (REPRODUCIBLE) =====
# Reproducibility switches
import os
os.environ["PYTHONHASHSEED"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import random
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Global seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
rng = np.random.default_rng(SEED)

# === CONFIGURATION ===
csv_path = "C:/Users/vishn/OneDrive/Desktop/ML01/Vishnu_phd.csv"
output_dir = "C:/Users/vishn/OneDrive/Desktop/ML01/ABC-ANN_s"
os.makedirs(output_dir, exist_ok=True)

# === LOAD DATA FOR SeismicFoS ===
df = pd.read_csv(csv_path)
X = df.iloc[:, :9].values
y = df['SeismicFoS'].values.reshape(-1, 1)

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y).ravel()

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=SEED
)

# === ABC PARAMETERS ===
NUM_BEES = 20
LIMIT = 5
MAX_ITER = 30

def generate_solution():
    return [
        random.randint(8, 128),
        random.randint(8, 128),
        random.randint(0, 128),
        round(random.uniform(0.0001, 0.01), 5)
    ]

def evaluate_solution(sol):
    try:
        layers = tuple(n for n in sol[:3] if n > 0)
        lr = sol[3]
        model = MLPRegressor(
            hidden_layer_sizes=layers, learning_rate_init=lr,
            activation='relu', solver='adam', max_iter=1000, random_state=SEED
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_true_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
        y_pred_inv = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()
        return r2_score(y_true_inv, y_pred_inv)
    except Exception:
        return -np.inf

# === INITIALIZE POPULATION ===
solutions = [generate_solution() for _ in range(NUM_BEES)]
fitness = [evaluate_solution(sol) for sol in solutions]
trial = [0 for _ in range(NUM_BEES)]

global_best_index = int(np.argmax(fitness))
global_best_sol = solutions[global_best_index].copy()
global_best_fitness = fitness[global_best_index]
best_fitness_progress = [global_best_fitness]

# === ABC MAIN LOOP ===
for it in range(MAX_ITER):
    # Employed bee phase
    for i in range(NUM_BEES):
        phi = generate_solution()
        k = random.randint(0, NUM_BEES - 1)
        while k == i:
            k = random.randint(0, NUM_BEES - 1)
        new_sol = [(solutions[i][j] if random.random() > 0.5 else phi[j]) for j in range(4)]
        new_fit = evaluate_solution(new_sol)
        if new_fit > fitness[i]:
            solutions[i] = new_sol
            fitness[i] = new_fit
            trial[i] = 0
        else:
            trial[i] += 1
        if new_fit > global_best_fitness:
            global_best_sol = new_sol.copy()
            global_best_fitness = new_fit

    # Onlooker phase
    adjusted_fitness = [max(0.0, f) if np.isfinite(f) else 0.0 for f in fitness]
    fit_sum = float(sum(adjusted_fitness))
    probs = [f / fit_sum if fit_sum > 0 else 1.0 / NUM_BEES for f in adjusted_fitness]

    for _ in range(NUM_BEES):
        i = int(rng.choice(NUM_BEES, p=probs))
        phi = generate_solution()
        new_sol = [(solutions[i][j] if random.random() > 0.5 else phi[j]) for j in range(4)]
        new_fit = evaluate_solution(new_sol)
        if new_fit > fitness[i]:
            solutions[i] = new_sol
            fitness[i] = new_fit
            trial[i] = 0
        else:
            trial[i] += 1
        if new_fit > global_best_fitness:
            global_best_sol = new_sol.copy()
            global_best_fitness = new_fit

    # Scout phase
    for i in range(NUM_BEES):
        if trial[i] > LIMIT:
            new_sol = generate_solution()
            new_fit = evaluate_solution(new_sol)
            solutions[i] = new_sol
            fitness[i] = new_fit
            trial[i] = 0
            if new_fit > global_best_fitness:
                global_best_sol = new_sol.copy()
                global_best_fitness = new_fit

    best_fitness_progress.append(global_best_fitness)


# %%
# ===== SECTION 2: FINAL TRAIN, METRICS, SAVE ARTIFACTS =====
best_layers = tuple(n for n in global_best_sol[:3] if n > 0)
best_lr = global_best_sol[3]
final_model = MLPRegressor(
    hidden_layer_sizes=best_layers, learning_rate_init=best_lr,
    activation='relu', solver='adam', max_iter=1000, random_state=SEED
)
final_model.fit(X_train, y_train)

joblib.dump(final_model, os.path.join(output_dir, "abc_ann_model.joblib"))
joblib.dump(scaler_X, os.path.join(output_dir, "scaler_X.joblib"))
joblib.dump(scaler_y, os.path.join(output_dir, "scaler_y.joblib"))

def inverse(y_scaled): 
    return scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).ravel()

y_train_true = inverse(y_train)
y_test_true  = inverse(y_test)
y_train_pred = inverse(final_model.predict(X_train))
y_test_pred  = inverse(final_model.predict(X_test))

def compute_metrics(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
    }

train_metrics = compute_metrics(y_train_true, y_train_pred)
test_metrics  = compute_metrics(y_test_true,  y_test_pred)

with open(os.path.join(output_dir, "abc_ann_metrics.txt"), "w", encoding="utf-8") as f:
    f.write("ABC-ANN Model Evaluation (SeismicFoS)\n")
    f.write("=================================\n")
    f.write(f"Best Architecture: {best_layers}\n")
    f.write(f"Learning Rate: {best_lr}\n\n")
    f.write("Train Metrics:\n")
    for k, v in train_metrics.items():
        f.write(f"  {k}: {v:.6f}\n")
    f.write("\nTest Metrics:\n")
    for k, v in test_metrics.items():
        f.write(f"  {k}: {v:.6f}\n")

pd.DataFrame({
    "iteration": list(range(len(best_fitness_progress))),
    "best_r2": best_fitness_progress
}).to_csv(os.path.join(output_dir, "fitness_progress.csv"), index=False)

np.savetxt(os.path.join(output_dir, "train_actual.txt"),    y_train_true)
np.savetxt(os.path.join(output_dir, "train_predicted.txt"), y_train_pred)
np.savetxt(os.path.join(output_dir, "test_actual.txt"),     y_test_true)
np.savetxt(os.path.join(output_dir, "test_predicted.txt"),  y_test_pred)

print("\n✅ ABC-ANN (SeismicFoS) Training Completed")
print("Best Architecture:", best_layers)
print("Learning Rate:", best_lr)
print("Train R²: {:.4f}".format(train_metrics['R2']))
print("Test R² : {:.4f}".format(test_metrics['R2']))
print("All results saved in:", output_dir)


# %%
# ===== SECTION 3: PUBLICATION-READY PLOTS =====
hq_dir = os.path.join(output_dir, "d_abc")
os.makedirs(hq_dir, exist_ok=True)

plt.rcParams.update({
    'figure.dpi': 300, 'savefig.dpi': 300,
    'font.size': 14, 'axes.labelsize': 14, 'axes.titlesize': 16,
    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12,
    'lines.linewidth': 2, 'lines.markersize': 6, 'grid.alpha': 0.5,
})

def save_both(path_base):
    plt.tight_layout()
    plt.savefig(path_base + ".png", bbox_inches='tight')
    plt.savefig(path_base + ".pdf", bbox_inches='tight')
    plt.close()

# Fitness curve
plt.figure(figsize=(7.5, 5.5))
plt.plot(best_fitness_progress)
plt.xlabel("Iteration")
plt.ylabel("Best $R^2$")
plt.title("ABC Fitness over Iterations (SeismicFoS)")
plt.grid(True)
save_both(os.path.join(hq_dir, "abc_fitness_plot"))

# Scatter helper
def scatter_ap(y_true, y_pred, title, fname, xlabel="Actual SeismicFoS", ylabel="Predicted SeismicFoS"):
    r2  = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    plt.figure(figsize=(7.2, 6.2))
    plt.scatter(y_true, y_pred, alpha=0.7, s=40)
    mn = min(np.min(y_true), np.min(y_pred))
    mx = max(np.max(y_true), np.max(y_pred))
    pad = 0.02*(mx-mn) if mx > mn else 1.0
    lims = [mn - pad, mx + pad]
    plt.plot(lims, lims, linestyle='--')
    plt.xlim(lims); plt.ylim(lims)
    ax = plt.gca(); ax.set_aspect('equal', adjustable='box')
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.title(title); plt.grid(True)
    plt.text(0.02, 0.98, f"$R^2$ = {r2:.4f}\nRMSE = {rmse:.4f}",
             transform=ax.transAxes, va='top', ha='left',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    save_both(os.path.join(hq_dir, fname))

# Scatter plots
scatter_ap(y_train_true, y_train_pred, "Train Set: Actual vs Predicted", "abc_scatter_train")
scatter_ap(y_test_true,  y_test_pred,  "Test Set: Actual vs Predicted",  "abc_scatter_test")

# Line plots
def line_plot(y_true, y_pred, title, fname, ylabel="SeismicFoS"):
    plt.figure(figsize=(12, 5))
    idx = np.arange(len(y_true))
    plt.plot(idx, y_true, label="Actual", linewidth=2)
    plt.plot(idx, y_pred, label="Predicted", linewidth=2)
    plt.title(title); plt.xlabel("Sample Index"); plt.ylabel(ylabel)
    plt.legend(frameon=False); plt.grid(True)
    save_both(os.path.join(hq_dir, fname))

line_plot(y_train_true, y_train_pred, "Actual vs Predicted (Train)", "abc_lineplot_train")
line_plot(y_test_true,  y_test_pred,  "Actual vs Predicted (Test)",  "abc_lineplot_test")

print("High-quality figures saved in:", hq_dir)


