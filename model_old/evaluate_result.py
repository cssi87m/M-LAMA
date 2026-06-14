import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import logging
from datetime import datetime
import os

# ==============================
# Logging setup
# ==============================
log_filename = f"logs/evaluation_report.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger("").addHandler(console)

# ==============================
# Load data
# ==============================
INPUT_CSV = "results/test_predictions_full.csv"
df = pd.read_csv(INPUT_CSV)

logging.info(f"📂 Loaded {len(df)} samples from {INPUT_CSV}")
logging.info(f"📋 Available columns: {list(df.columns)}")

# ==============================
# Define evaluation metrics
# ==============================
def compute_metrics(true, pred):
    mae = np.mean(np.abs(true - pred))
    mse = np.mean((true - pred) ** 2)
    rmse = np.sqrt(mse)
    pearson_corr, _ = pearsonr(true, pred) if len(true) > 1 else (np.nan, np.nan)
    spearman_corr, _ = spearmanr(true, pred) if len(true) > 1 else (np.nan, np.nan)
    return mae, mse, rmse, pearson_corr, spearman_corr

# ==============================
# Compute metrics per skill
# ==============================
skills = ["grammar", "vocabulary", "content", "fluency", "pronunciation"]
report = []
valid_skills = []

for skill in skills:
    true_col = f"{skill}_true"
    pred_col = f"{skill}_pred"

    if true_col not in df.columns or pred_col not in df.columns:
        logging.warning(f"⚠️ Missing columns for {skill}, skipping...")
        continue

    valid_skills.append(skill)
    true_vals = df[true_col].astype(float)
    pred_vals = df[pred_col].astype(float)

    mae, mse, rmse, pearson_corr, spearman_corr = compute_metrics(true_vals, pred_vals)

    report.append({
        "Skill": skill.capitalize(),
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "Pearson": pearson_corr,
        "Spearman": spearman_corr
    })

    logging.info(f"📊 {skill.capitalize()} Results:")
    logging.info(f"   MAE       : {mae:.4f}")
    logging.info(f"   MSE       : {mse:.4f}")
    logging.info(f"   RMSE      : {rmse:.4f}")
    logging.info(f"   Pearson r : {pearson_corr:.4f}")
    logging.info(f"   Spearman  : {spearman_corr:.4f}")
    logging.info("-" * 40)

# ==============================
# Compute overall metrics
# ==============================
if not valid_skills:
    logging.error("❌ No valid skill columns found. Cannot compute overall metrics.")
    logging.error(f"   Expected column patterns: {{skill}}_true and {{skill}}_pred")
    logging.error(f"   Where skill is one of: {skills}")
else:
    all_true = df[[f"{s}_true" for s in valid_skills]].to_numpy().flatten()
    all_pred = df[[f"{s}_pred" for s in valid_skills]].to_numpy().flatten()

    overall_mae, overall_mse, overall_rmse, overall_pearson, overall_spearman = compute_metrics(all_true, all_pred)

logging.info("🏁 Overall Performance Across All Skills:")
logging.info(f"   MAE       : {overall_mae:.4f}")
logging.info(f"   MSE       : {overall_mse:.4f}")
logging.info(f"   RMSE      : {overall_rmse:.4f}")
logging.info(f"   Pearson r : {overall_pearson:.4f}")
logging.info(f"   Spearman  : {overall_spearman:.4f}")
logging.info("=" * 60)

# ==============================
# Save summary CSV
# ==============================
if report:
    summary_path = "evaluation_summary_gemini_pro.csv"
    pd.DataFrame(report).to_csv(summary_path, index=False)
    logging.info(f"✅ Detailed summary saved to {summary_path}")
else:
    logging.warning("⚠️ No metrics computed, no summary file saved.")

logging.info(f"🧾 Full log written to {os.path.abspath(log_filename)}")