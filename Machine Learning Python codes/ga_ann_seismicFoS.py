# %%
# ===== SECTION 1: SETUP, DATA, GA SEARCH (REPRODUCIBLE) =====
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
SEED = 44
random.seed(SEED)
np.random.seed(SEED)

# === CONFIGURATION ===
csv_path = "C:/Users/vishn/OneDrive/Desktop/ML01/Vishnu_phd.csv"
output_dir = "C:/Users/vishn/OneDrive/Desktop/ML01/GA-ANN_s"
os.makedirs(output_dir, exist_ok=True)

# === LOAD AND PREPROCESS (SeismicFoS) ===
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

# === GA PARAMETERS ===
POP_SIZE = 20
GENERATIONS = 30
MUTATION_RATE = 0.2

def generate_individual():
    return [
        random.randint(8, 128),   # h1
        random.randint(8, 128),   # h2
        random.randint(0, 128),   # h3 (0=skip)
        round(random.uniform(0.0001, 0.01), 5)  # learning rate
    ]

def build_model(ind):
    layers = [n for n in ind[:3] if n > 0]
    lr = ind[3]
    return MLPRegressor(
        hidden_layer_sizes=tuple(layers),
        learning_rate_init=lr,
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=SEED
    )

def evaluate(ind):
    try:
        model = build_model(ind)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_inv = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()
        y_true_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
        return r2_score(y_true_inv, y_pred_inv)
    except Exception:
        return -np.inf

def mutate(ind):
    idx = random.randint(0, 3)
    if idx < 3:
        ind[idx] = random.randint(0 if idx == 2 else 8, 128)
    else:
        ind[idx] = round(random.uniform(0.0001, 0.01), 5)
    return ind

def crossover(p1, p2):
    point = random.randint(1, 3)
    return p1[:point] + p2[point:]

# === RUN GA ===
population = [generate_individual() for _ in range(POP_SIZE)]
fitness_history = []

for gen in range(GENERATIONS):
    scores = [evaluate(ind) for ind in population]
    fitness_history.append(max(scores))
    print(f"Generation {gen+1}: Max R² = {max(scores):.4f}")
    
    # select elites
    top_indices = np.argsort(scores)[-5:]
    top_individuals = [population[i] for i in top_indices]

    # make next population
    new_population = top_individuals[:]
    while len(new_population) < POP_SIZE:
        p1, p2 = random.sample(top_individuals, 2)
        child = crossover(p1, p2)
        if random.random() < MUTATION_RATE:
            child = mutate(child)
        new_population.append(child)
    
    population = new_population

# === FINAL BEST MODEL ===
best_individual = max(population, key=evaluate)
best_model = build_model(best_individual)
best_model.fit(X_train, y_train)


# %%
# ===== SECTION 2: METRICS, SAVES, PUBLICATION-READY PLOTS (PNG only) =====

# Save model and scalers
joblib.dump(best_model, os.path.join(output_dir, "ga_ann_best_model.joblib"))
joblib.dump(scaler_X, os.path.join(output_dir, "scaler_X.joblib"))
joblib.dump(scaler_y, os.path.join(output_dir, "scaler_y.joblib"))

# Inverse transforms and metrics
def inverse_y(y_scaled):
    return scaler_y.inverse_transform(np.asarray(y_scaled).reshape(-1, 1)).ravel()

y_train_true = inverse_y(y_train)
y_test_true  = inverse_y(y_test)
y_train_pred = inverse_y(best_model.predict(X_train))
y_test_pred  = inverse_y(best_model.predict(X_test))

def compute_metrics(y_true, y_pred):
    return {
        "R2":   r2_score(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
        "MSE":  float(mean_squared_error(y_true, y_pred)),
    }

train_metrics = compute_metrics(y_train_true, y_train_pred)
test_metrics  = compute_metrics(y_test_true,  y_test_pred)

# Save metrics (same layout as before)
with open(os.path.join(output_dir, "ga_ann_metrics.txt"), "w", encoding='utf-8') as f:
    f.write("GA-ANN Model Evaluation (SeismicFoS)\n")
    f.write("=================================\n")
    f.write(f"Best Architecture: {tuple([n for n in best_individual[:3] if n > 0])}\n")
    f.write(f"Learning Rate: {best_individual[3]}\n\n")
    f.write("Train Metrics:\n")
    for k, v in train_metrics.items():
        f.write(f"  {k}: {v:.6f}\n")
    f.write("\nTest Metrics:\n")
    for k, v in test_metrics.items():
        f.write(f"  {k}: {v:.6f}\n")

# Save fitness history & predictions
pd.DataFrame({"generation": list(range(1, len(fitness_history)+1)),
              "best_r2": fitness_history}).to_csv(
    os.path.join(output_dir, "fitness_progress.csv"), index=False
)
np.savetxt(os.path.join(output_dir, "train_actual.txt"),    y_train_true)
np.savetxt(os.path.join(output_dir, "train_predicted.txt"), y_train_pred)
np.savetxt(os.path.join(output_dir, "test_actual.txt"),     y_test_true)
np.savetxt(os.path.join(output_dir, "test_predicted.txt"),  y_test_pred)

# ===== High-quality plots (PNG only) =====
hq_dir = os.path.join(output_dir, "d_ga")
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

# Fitness curve
plt.figure(figsize=(7.5, 5.5))
plt.plot(fitness_history)
plt.xlabel("Generation")
plt.ylabel("Best $R^2$ (Test)")
plt.title("GA Fitness Progress (SeismicFoS)")
plt.grid(True)
save_png(os.path.join(hq_dir, "ga_fitness_progress.png"))

# Scatter helper: equal axes + embedded stats
def scatter_actual_vs_pred(y_true, y_pred, title, fname,
                           xlabel="Actual SeismicFoS", ylabel="Predicted SeismicFoS"):
    r2  = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    plt.figure(figsize=(7.2, 6.2))
    plt.scatter(y_true, y_pred, alpha=0.7, s=40)
    mn = min(np.min(y_true), np.min(y_pred))
    mx = max(np.max(y_true), np.max(y_pred))
    pad = 0.02 * (mx - mn) if mx > mn else 1.0
    lims = [mn - pad, mx + pad]
    plt.plot(lims, lims, linestyle='--')
    plt.xlim(lims); plt.ylim(lims)
    ax = plt.gca(); ax.set_aspect('equal', adjustable='box')
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.title(title); plt.grid(True)
    plt.text(0.02, 0.98, f"$R^2$ = {r2:.4f}\nRMSE = {rmse:.4f}",
             transform=ax.transAxes, va='top', ha='left',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    save_png(os.path.join(hq_dir, fname))

# Scatter plots
scatter_actual_vs_pred(y_train_true, y_train_pred,
                       "Train Set: Actual vs Predicted (SeismicFoS)",
                       "ga_scatter_train.png")
scatter_actual_vs_pred(y_test_true,  y_test_pred,
                       "Test Set: Actual vs Predicted (SeismicFoS)",
                       "ga_scatter_test.png")

# Line plots
def line_plot(y_true, y_pred, title, fname, ylabel="SeismicFoS"):
    plt.figure(figsize=(12, 5))
    idx = np.arange(len(y_true))
    plt.plot(idx, y_true, label="Actual", linewidth=2)
    plt.plot(idx, y_pred, label="Predicted", linewidth=2)
    plt.title(title); plt.xlabel("Sample Index"); plt.ylabel(ylabel)
    plt.legend(frameon=False); plt.grid(True)
    save_png(os.path.join(hq_dir, fname))

line_plot(y_train_true, y_train_pred,
          "Actual vs Predicted (Train, SeismicFoS)",
          "ga_line_train.png")
line_plot(y_test_true,  y_test_pred,
          "Actual vs Predicted (Test, SeismicFoS)",
          "ga_line_test.png")

print("\n🎯 GA-ANN (SeismicFoS) completed. Best config and model saved to:", output_dir)
print("High-quality figures saved in:", hq_dir)


