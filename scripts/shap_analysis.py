"""
SHAP (SHapley Additive exPlanations) Analysis for Customer Churn Prediction

Generates:
  1. SHAP Summary Plot (beeswarm) — Global feature importance with directionality
  2. SHAP Summary Bar Plot        — Mean absolute SHAP values
  3. SHAP Waterfall Plot          — Single prediction explanation
  4. SHAP Force Plot              — Single prediction force visualization (matplotlib)

How SHAP works:
  - SHAP values explain how each feature contributes to pushing a prediction
    away from the base value (average model output).
  - Positive SHAP = pushes toward churn | Negative SHAP = pushes toward no churn
  - The sum of all SHAP values + base value = final model prediction.
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import shap

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import load_data, preprocess_data

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'images')
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------------
# 1. Load data, model, and artifacts
# ------------------------------------------------------------------
print("=" * 70)
print("  SHAP ANALYSIS — CUSTOMER CHURN PREDICTION")
print("  (Understanding how each feature drives the model's decision)")
print("=" * 70)

print("\n[1/5] Loading data and preprocessing...")
df = load_data()
X, y, label_encoders, scaler = preprocess_data(df)

print(f"\n[2/5] Loading trained model...")
model_path = os.path.join(MODELS_DIR, "churn_model.pkl")
if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"Model not found at {model_path}. Run train_model.py first."
    )
model = joblib.load(model_path)
model_type = type(model).__name__
print(f"  Model loaded: {model_type}")

# ------------------------------------------------------------------
# 2. Create SHAP Explainer (automatic based on model type)
#     - TreeExplainer for RandomForest, XGBoost, LightGBM, etc.
#     - LinearExplainer for LogisticRegression
#     - KernelExplainer fallback for any other model
# ------------------------------------------------------------------
print("\n[3/5] Creating SHAP Explainer and computing SHAP values...")
print("  (This computes Shapley values for every feature x every sample)")
print("  Note: Computing on a subset of 100 samples for speed...")

# Use a subset for faster computation (100 samples is sufficient)
X_sample = X.sample(n=min(100, len(X)), random_state=42)

# Select the correct explainer based on model type
if model_type in ["RandomForestClassifier", "GradientBoostingClassifier", "XGBClassifier", "LGBMClassifier"]:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_values_churn = shap_values[1]
        expected_value = explainer.expected_value[1]
    else:
        shap_values_churn = shap_values
        expected_value = explainer.expected_value

elif model_type == "LogisticRegression":
    explainer = shap.LinearExplainer(model, X_sample, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_sample)
    shap_values_churn = shap_values
    expected_value = explainer.expected_value

else:
    # Fallback: KernelExplainer (model-agnostic)
    print(f"  Model type '{model_type}' not natively supported. Using KernelExplainer...")
    background = X.sample(n=min(50, len(X)), random_state=42)
    explainer = shap.KernelExplainer(model.predict_proba, background)
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_values_churn = shap_values[1]
    else:
        shap_values_churn = shap_values
    expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value

print(f"  SHAP values shape: {shap_values_churn.shape}")
print(f"  Base value (expected model output): {expected_value:.4f}")
print(f"  Interpretation: Base value is the average prediction across the dataset.")
print(f"  SHAP values explain deviations from this base for each prediction.")

# ------------------------------------------------------------------
# 3. SHAP Summary Plot (Beeswarm)
# ------------------------------------------------------------------
print("\n[4/5] Generating SHAP plots...")

# 3a. SHAP Beeswarm Summary Plot
print("\n  >> Generating SHAP Beeswarm Summary Plot...")
plt.figure(figsize=(12, 10))
shap.summary_plot(
    shap_values_churn, X_sample,
    plot_type="dot",
    show=False,
    max_display=15,
    color_bar_label="Feature Value (High to Low)"
)
plt.title("SHAP Summary Plot — Feature Impact on Customer Churn\n"
          "(Red = high feature value pushes churn | Blue = low feature value pushes no churn)",
          fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
beeswarm_path = os.path.join(OUTPUT_DIR, "shap_summary_beeswarm.png")
plt.savefig(beeswarm_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"     Saved to {beeswarm_path}")

# 3b. SHAP Summary Bar Plot (Mean |SHAP|)
print("\n  >> Generating SHAP Summary Bar Plot...")
plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_values_churn, X_sample,
    plot_type="bar",
    show=False,
    max_display=15,
    color="royalblue"
)
plt.title("Top 15 Features by Mean |SHAP Value|\n"
          "(Average magnitude of impact on churn prediction)",
          fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
bar_path = os.path.join(OUTPUT_DIR, "shap_summary_bar.png")
plt.savefig(bar_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"     Saved to {bar_path}")

# ------------------------------------------------------------------
# 4. SHAP Waterfall Plot — Explain a Single Prediction
# ------------------------------------------------------------------
print("\n  >> Generating SHAP Waterfall Plot (single customer explanation)...")

# Find an actual churner in the sample for a more interesting explanation
churner_idx = None
for idx in range(len(X_sample)):
    if y.loc[X_sample.index[idx]] == 1:
        churner_idx = idx
        break

if churner_idx is None:
    churner_idx = 0  # fallback

single_customer = X_sample.iloc[[churner_idx]]
actual_label = y.loc[X_sample.index[churner_idx]]
churn_prob = model.predict_proba(single_customer)[0][1]

print(f"  Explaining prediction for customer #{churner_idx + 1}:")
print(f"    Actual label : {'Churn' if actual_label == 1 else 'No Churn'}")
print(f"    Model predicts: {'Churn' if churn_prob >= 0.5 else 'No Churn'}")
print(f"    Churn probability: {churn_prob:.4f} ({churn_prob*100:.2f}%)")

plt.figure(figsize=(12, 8))
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values_churn[churner_idx],
        base_values=expected_value,
        data=X_sample.iloc[churner_idx].values,
        feature_names=X_sample.columns.tolist()
    ),
    show=False,
    max_display=15
)
plt.title(f"SHAP Waterfall Plot — Customer #{churner_idx + 1}\n"
          f"(Actual: {'Churn' if actual_label == 1 else 'No Churn'} | "
          f"Predicted: {'Churn' if churn_prob >= 0.5 else 'No Churn'} | "
          f"Probability: {churn_prob:.2%})",
          fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
waterfall_path = os.path.join(OUTPUT_DIR, "shap_waterfall.png")
plt.savefig(waterfall_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"     Saved to {waterfall_path}")

# ------------------------------------------------------------------
# 5. SHAP Force Plot — Visualize Feature Contributions
# ------------------------------------------------------------------
print("\n  >> Generating SHAP Force Plot...")

# matplotlib-based force plot (saves as static image)
plt.figure(figsize=(20, 4))
shap.force_plot(
    expected_value,
    shap_values_churn[churner_idx],
    X_sample.iloc[churner_idx],
    matplotlib=True,
    show=False,
    figsize=(20, 4),
    text_rotation=15
)
plt.title(f"SHAP Force Plot — Customer #{churner_idx + 1}\n"
          f"f(x) = {churn_prob:.3f} | Base = {expected_value:.3f}\n"
          f"Red features push toward churn | Blue features push toward no churn",
          fontsize=12, fontweight='bold', pad=15)
plt.tight_layout()
force_path = os.path.join(OUTPUT_DIR, "shap_force_plot.png")
plt.savefig(force_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"     Saved to {force_path}")

# ------------------------------------------------------------------
# 6. Print SHAP Values Table for the Explained Customer
# ------------------------------------------------------------------
print("\n[5/5] Detailed SHAP values for the explained customer:")
print("  " + "-" * 70)
print(f"  {'Feature':<35} {'Value':<15} {'SHAP Value':<12} {'Impact Direction'}")
print("  " + "-" * 70)

customer_shap = shap_values_churn[churner_idx]
customer_data = X_sample.iloc[churner_idx]

# Sort by absolute SHAP value
sorted_indices = np.argsort(np.abs(customer_shap))[::-1]

for idx in sorted_indices:
    feature = X_sample.columns[idx]
    value = customer_data[feature]
    shap_val = customer_shap[idx]
    direction = "+ → Churn" if shap_val > 0 else "- → No Churn"
    print(f"  {feature:<35} {value:<15.4f} {shap_val:<+12.4f} {direction}")

print("  " + "-" * 70)
print(f"\n  {'Total SHAP + Base = Prediction':<50} {expected_value + customer_shap.sum():.4f}")
print(f"  {'Base value (avg model output)':<50} {expected_value:.4f}")
print(f"  {'Model probability output':<50} {churn_prob:.4f}")

# ------------------------------------------------------------------
# 7. Key Insights Summary
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("  SHAP ANALYSIS — KEY INSIGHTS")
print("=" * 70)
print("""
  How to read SHAP plots:
  -------------------------------------------------------------

  1. BEESWARM PLOT
     - Each dot = one customer's SHAP value for a feature
     - Red = high feature value  |  Blue = low feature value
     - Right of center = pushes toward churn
     - Left of center = pushes toward no churn

  2. BAR PLOT
     - Average absolute SHAP value across all customers
     - Higher bar = feature has more impact on average

  3. WATERFALL PLOT
     - Explains ONE customer's prediction
     - Starts at base value, each bar adds/subtracts impact
     - Ends at final prediction f(x)

  4. FORCE PLOT
     - Red arrows push toward churn
     - Blue arrows push toward no churn
     - f(x) = final prediction score

  TYPICAL FINDINGS (Telco Churn Dataset):
  - Contract (Month-to-month) is usually the #1 driver of churn
  - Tenure: new customers (low tenure) are riskier
  - MonthlyCharges: higher charges = higher churn risk
  - InternetService (Fiber optic) increases churn likelihood
  - TechSupport (No) and OnlineSecurity (No) increase churn risk
""")

print("=" * 70)
print("  SHAP ANALYSIS COMPLETE — All plots saved to images/")
print("=" * 70)
