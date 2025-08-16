# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# ===== Reproducibility =====
SEED = 55
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)

# ===== Paths =====
csv_path = r"C:\Users\vishn\OneDrive\Documents\Machine Learning\Vishnu_phd.csv"
save_dir = r"C:\Users\vishn\OneDrive\Desktop\ML01\ML_traditional"
os.makedirs(save_dir, exist_ok=True)

# ===== Data Loading & Split =====
df = pd.read_csv(csv_path)
X = df.drop(columns=["FoS", "SeismicFoS"])
y_fos = df["FoS"]
y_seis = df["SeismicFoS"]

X_train, X_test, y_train_all, y_test_all = train_test_split(
    X, df[["FoS", "SeismicFoS"]], test_size=0.2, random_state=SEED
)
y_fos_train, y_seis_train = y_train_all["FoS"], y_train_all["SeismicFoS"]
y_fos_test,  y_seis_test  = y_test_all["FoS"],  y_test_all["SeismicFoS"]

# ===== Scaling =====
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Save the scaler (for future inference)
joblib.dump(scaler, os.path.join(save_dir, "scaler_X.joblib"), compress=3)

# ===== Models =====
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(),
    "DecisionTree": DecisionTreeRegressor(random_state=SEED),
    "RandomForest": RandomForestRegressor(random_state=SEED, n_jobs=1),
    "GradientBoosting": GradientBoostingRegressor(random_state=SEED),
    "SVR": SVR(),
    "KNN": KNeighborsRegressor()
}

# ===== Evaluation =====
def evaluate_all(model, X_tr, y_tr, X_te, y_te):
    y_tr_pred = model.predict(X_tr)
    y_te_pred = model.predict(X_te)
    return {
        "y_train_pred": y_tr_pred,
        "y_test_pred":  y_te_pred,
        "R2_train":   r2_score(y_tr, y_tr_pred),
        "R2_test":    r2_score(y_te, y_te_pred),
        "RMSE_train": float(np.sqrt(mean_squared_error(y_tr, y_tr_pred))),
        "RMSE_test":  float(np.sqrt(mean_squared_error(y_te, y_te_pred))),
        "MSE_train":  float(mean_squared_error(y_tr, y_tr_pred)),
        "MSE_test":   float(mean_squared_error(y_te, y_te_pred)),
        "MAE_train":  float(mean_absolute_error(y_tr, y_tr_pred)),
        "MAE_test":   float(mean_absolute_error(y_te, y_te_pred)),
    }

# ===== Publication-ready plotting defaults =====
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

# ===== Scatter Plot =====
def scatter_actual_vs_pred(y_true, y_pred, title, fname, out_dir,
                           xlabel="Actual", ylabel="Predicted"):
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)

    plt.figure(figsize=(7.2, 6.2))
    plt.scatter(y_true, y_pred, alpha=0.7, s=40)
    mn = float(min(np.min(y_true), np.min(y_pred)))
    mx = float(max(np.max(y_true), np.max(y_pred)))
    pad = 0.02 * (mx - mn) if mx > mn else 1.0
    lims = [mn - pad, mx + pad]
    plt.plot(lims, lims, linestyle='--')
    plt.xlim(lims); plt.ylim(lims)
    ax = plt.gca(); ax.set_aspect('equal', adjustable='box')
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.title(title); plt.grid(True)
    ax.text(0.02, 0.98, f"$R^2$ = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}",
            transform=ax.transAxes, va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    save_png(os.path.join(out_dir, fname))

# ===== Line Plot =====
def line_plot(y_true, y_pred, title, fname, out_dir, ylabel="Value"):
    plt.figure(figsize=(12, 5))
    idx = np.arange(len(y_true))
    plt.plot(idx, y_true, label="Actual", linewidth=2)
    plt.plot(idx, y_pred, label="Predicted", linewidth=2)
    plt.title(title); plt.xlabel("Sample Index"); plt.ylabel(ylabel)
    plt.legend(frameon=False); plt.grid(True)
    save_png(os.path.join(out_dir, fname))

# ===== Feature Importance with smaller numeric labels =====
def plot_feature_importance_pub_labeled(
    model,
    feature_names,
    title,
    out_dir,
    fname="feature_importance_labeled.png",
    normalize=True,
    decimals=2,
    horizontal=True
):
    if not hasattr(model, "feature_importances_"):
        return

    importances = np.asarray(model.feature_importances_)
    order = np.argsort(importances)[::-1]
    imp = importances[order]
    names = [feature_names[i] for i in order]

    total = imp.sum() if imp.sum() != 0 else 1.0
    imp_pct = (imp / total) * 100.0

    # Save CSV
    fi_df = pd.DataFrame({
        "feature": names,
        "importance": imp,
        "importance_pct": imp_pct
    })
    fi_df.to_csv(os.path.join(out_dir, "feature_importances.csv"), index=False)

    n = len(imp)
    if horizontal:
        plt.figure(figsize=(8, max(3.0, 0.4 * n)))
        y = np.arange(n)
        bars = plt.barh(y, imp)
        plt.gca().invert_yaxis()
        plt.yticks(y, names)
        plt.xlabel("Importance")
        plt.title(title)

        for i, b in enumerate(bars):
            val = imp_pct[i] if normalize else imp[i]
            suffix = "%" if normalize else ""
            txt = f"{val:.{decimals}f}{suffix}"
            x = b.get_width()
            ytxt = b.get_y() + b.get_height() / 2
            xtext = x + (0.015 * (imp.max() if imp.max() > 0 else 1.0))  # moved slightly away
            plt.text(xtext, ytxt, txt, va="center", ha="left", fontsize=10)  # reduced size
        save_png(os.path.join(out_dir, fname))
    else:
        plt.figure(figsize=(max(6.0, 0.5 * n), 4.0))
        x = np.arange(n)
        bars = plt.bar(x, imp)
        plt.xticks(x, names, rotation=60, ha='right')
        plt.ylabel("Importance")
        plt.title(title)

        for i, b in enumerate(bars):
            val = imp_pct[i] if normalize else imp[i]
            suffix = "%" if normalize else ""
            txt = f"{val:.{decimals}f}{suffix}"
            xtxt = b.get_x() + b.get_width() / 2
            ytxt = b.get_height()
            plt.text(xtxt, ytxt + (0.01 * imp.max()), txt, va="bottom", ha="center", fontsize=10)  # reduced size
        save_png(os.path.join(out_dir, fname))

# ===== Main: train, save, plot =====
metrics_master = os.path.join(save_dir, "model_evaluation_summary.txt")
with open(metrics_master, 'w', encoding='utf-8') as f_out:
    for name, model in models.items():
        for target, y_train, y_test in [("FoS", y_fos_train, y_fos_test),
                                        ("SeismicFoS", y_seis_train, y_seis_test)]:

            hq_dir = os.path.join(save_dir, f"{name}_{target}")
            os.makedirs(hq_dir, exist_ok=True)

            model.fit(X_train_scaled, y_train)
            preds = evaluate_all(model, X_train_scaled, y_train,
                                 X_test_scaled, y_test)

            joblib.dump(model, os.path.join(hq_dir, f"{name}_{target}_model.joblib"), compress=3)

            np.savetxt(os.path.join(hq_dir, "train_actual.txt"),    np.asarray(y_train))
            np.savetxt(os.path.join(hq_dir, "train_predicted.txt"), preds["y_train_pred"])
            np.savetxt(os.path.join(hq_dir, "test_actual.txt"),     np.asarray(y_test))
            np.savetxt(os.path.join(hq_dir, "test_predicted.txt"),  preds["y_test_pred"])
            pd.DataFrame({"y_true": y_train, "y_pred": preds["y_train_pred"]}).to_csv(
                os.path.join(hq_dir, "train_predictions.csv"), index=False)
            pd.DataFrame({"y_true": y_test, "y_pred": preds["y_test_pred"]}).to_csv(
                os.path.join(hq_dir, "test_predictions.csv"), index=False)

            metrics_path = os.path.join(hq_dir, f"{name}_{target}_metrics.txt")
            with open(metrics_path, "w", encoding='utf-8') as fm:
                fm.write(f"{name} Model Evaluation ({target})\n")
                fm.write("=" * 33 + "\n")
                fm.write("Train Metrics:\n")
                fm.write(f"  R2:   {preds['R2_train']:.6f}\n")
                fm.write(f"  RMSE: {preds['RMSE_train']:.6f}\n")
                fm.write(f"  MAE:  {preds['MAE_train']:.6f}\n")
                fm.write(f"  MSE:  {preds['MSE_train']:.6f}\n\n")
                fm.write("Test Metrics:\n")
                fm.write(f"  R2:   {preds['R2_test']:.6f}\n")
                fm.write(f"  RMSE: {preds['RMSE_test']:.6f}\n")
                fm.write(f"  MAE:  {preds['MAE_test']:.6f}\n")
                fm.write(f"  MSE:  {preds['MSE_test']:.6f}\n")

            f_out.write(f"Model: {name} | Target: {target}\n")
            for m in ["R2_train","R2_test","RMSE_train","RMSE_test","MSE_train","MSE_test","MAE_train","MAE_test"]:
                f_out.write(f"  {m}: {preds[m]:.6f}\n")
            f_out.write("\n")

            scatter_actual_vs_pred(
                y_train, preds["y_train_pred"],
                f"Train: Actual vs Predicted ({name} – {target})",
                "scatter_train.png", hq_dir,
                xlabel=f"Actual {target}", ylabel=f"Predicted {target}"
            )
            scatter_actual_vs_pred(
                y_test, preds["y_test_pred"],
                f"Test: Actual vs Predicted ({name} – {target})",
                "scatter_test.png", hq_dir,
                xlabel=f"Actual {target}", ylabel=f"Predicted {target}"
            )

            line_plot(
                y_train, preds["y_train_pred"],
                f"Actual vs Predicted – Train ({name} – {target})",
                "line_train.png", hq_dir, ylabel=target
            )
            line_plot(
                y_test, preds["y_test_pred"],
                f"Actual vs Predicted – Test ({name} – {target})",
                "line_test.png", hq_dir, ylabel=target
            )

            plot_feature_importance_pub_labeled(
                model, X.columns.tolist(),
                title=f"Feature Importance – {name} / {target}",
                out_dir=hq_dir,
                fname="feature_importance_labeled.png",
                normalize=True,
                decimals=2,
                horizontal=True
            )

print(f"\n✔ Completed. Master metrics at: {metrics_master}")
print(f"   Per-model outputs saved under: {save_dir}\\<Model>_<Target>")


# %%
import glob
import matplotlib.pyplot as plt

def merge_feature_importance_plots(base_dir, output_file="merged_feature_importance.png"):
    """
    Reads all feature_importances.csv files under base_dir and creates a merged comparative plot.
    Each model/target combination will be a separate bar in the grouped plot.
    """
    csv_files = glob.glob(os.path.join(base_dir, "*_*", "feature_importances.csv"))
    if not csv_files:
        print("No feature_importances.csv found. Run model training first.")
        return
    
    merged_df = pd.DataFrame()
    
    for csv_file in csv_files:
        df_fi = pd.read_csv(csv_file)
        model_target = os.path.basename(os.path.dirname(csv_file))
        df_fi["Model_Target"] = model_target
        merged_df = pd.concat([merged_df, df_fi], ignore_index=True)
    
    # Sort features by mean importance across all models
    feature_order = merged_df.groupby("feature")["importance"].mean().sort_values(ascending=False).index.tolist()
    
    # Pivot for grouped plot
    pivot_df = merged_df.pivot_table(index="feature", columns="Model_Target", values="importance_pct", fill_value=0)
    pivot_df = pivot_df.loc[feature_order]  # reorder by importance
    
    # Plot
    plt.figure(figsize=(max(8, len(pivot_df) * 0.6), max(5, len(pivot_df.columns) * 0.5)))
    pivot_df.plot(kind="bar", width=0.8)
    plt.ylabel("Importance (%)")
    plt.title("Feature Importance Comparison Across Models")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Model_Target", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    
    merged_path = os.path.join(base_dir, output_file)
    plt.savefig(merged_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✔ Merged feature importance plot saved at: {merged_path}")

# === Run after training ===
merge_feature_importance_plots(save_dir)


