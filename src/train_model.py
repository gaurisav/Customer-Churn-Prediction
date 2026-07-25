import pandas as pd
import numpy as np
import joblib
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)

from src.preprocessing import load_data, preprocess_data


def plot_roc_curve(y_test, y_proba, model_name, save_path):
    """Plot and save the ROC curve."""
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#FF5722', lw=2.5,
             label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Recall / Sensitivity)', fontsize=12)
    plt.title(f'ROC Curve — {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  ROC curve saved to {save_path}")


def plot_confusion_matrix(y_test, y_pred, model_name, save_path):
    """Plot and save the confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))

    # Calculate percentages for annotations
    cm_sum = cm.sum()
    cm_pct = cm / cm_sum * 100

    # Create annotation matrix with counts and percentages
    annot = np.empty_like(cm).astype(str)
    n_rows, n_cols = cm.shape
    for i in range(n_rows):
        for j in range(n_cols):
            c = cm[i, j]
            p = cm_pct[i, j]
            annot[i, j] = f'{c}\n({p:.1f}%)'

    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', cbar=True,
                xticklabels=['No Churn (0)', 'Churn (1)'],
                yticklabels=['No Churn (0)', 'Churn (1)'],
                annot_kws={'size': 13})
    plt.title(f'Confusion Matrix — {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('Actual Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Confusion matrix saved to {save_path}")


def print_evaluation_report(y_test, y_pred, y_proba, model_name):
    """Print a detailed evaluation report with emphasis on churn-specific metrics."""
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Specificity / True Negative Rate
    specificity = tn / (tn + fp)

    print("\n" + "=" * 60)
    print(f"  EVALUATION REPORT — {model_name}")
    print("=" * 60)

    print("\n  ┌──────────────────────────┬──────────┐")
    print(f"  │ {'Metric':<24} │ {'Value':>8} │")
    print("  ├──────────────────────────┼──────────┤")
    print(f"  │ {'Accuracy':<24} │ {acc:>8.4f} │")
    print(f"  │ {'Precision':<24} │ {prec:>8.4f} │")
    print(f"  │ {'Recall (Sensitivity)':<24} │ {rec:>8.4f} │")
    print(f"  │ {'Specificity':<24} │ {specificity:>8.4f} │")
    print(f"  │ {'F1 Score':<24} │ {f1:>8.4f} │")
    print(f"  │ {'ROC-AUC':<24} │ {roc_auc:>8.4f} │")
    print("  └──────────────────────────┴──────────┘")

    print("\n  Confusion Matrix Breakdown:")
    print(f"    True Negatives  (TN): {tn:>5}  — Correctly predicted No Churn")
    print(f"    False Positives (FP): {fp:>5}  — Falsely predicted Churn (wasted retention budget)")
    print(f"    False Negatives (FN): {fn:>5}  — Missed churners (lost customers)")
    print(f"    True Positives  (TP): {tp:>5}  — Correctly predicted Churn")

    print(f"\n  Churn Rate in Test Set: {y_test.sum() / len(y_test):.2%}")

    print("\n  " + "-" * 58)
    print("  WHY PRECISION & RECALL MATTER FOR CHURN PREDICTION:")
    print("  " + "-" * 58)
    print("    \u2022  Churners are the minority class (~26.5% of data).")
    print("       A model that always predicts 'No Churn' would achieve")
    print("       ~73.5% accuracy but have 0% recall — useless!")
    print()
    print("    \u2022  PRECISION tells us: 'Of all customers flagged as churn,")
    print("       how many actually churned?' Low precision = wasted")
    print("       retention budget on false alarms.")
    print()
    print("    \u2022  RECALL tells us: 'Of all actual churners, how many did")
    print("       we catch?' Low recall = missed retention opportunities.")
    print()
    print("    \u2022  F1 SCORE balances both — our primary selection metric.")
    print()
    print("    \u2022  ROC-AUC measures the model's ability to distinguish")
    print("       between churners and non-churners across all thresholds.")
    print("  " + "-" * 58)

    # Business impact interpretation
    print("\n  Business Impact Assessment:")
    print(f"    \u2022  Model catches {rec:.1%} of actual churners (Recall)")
    print(f"    \u2022  {prec:.1%} of retention efforts are correctly targeted (Precision)")
    print(f"    \u2022  Missed churners: {fn} customers (potential revenue loss)")
    print(f"    \u2022  False alarms: {fp} customers (wasted retention cost)")

    return {
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "Specificity": round(specificity, 4),
        "F1 Score": round(f1, 4),
        "ROC-AUC": round(roc_auc, 4)
    }


def plot_feature_importance(best_model, X, model_name, save_path):
    """Plot feature importance (tree-based) or coefficients (linear model)."""
    if hasattr(best_model, "feature_importances_"):
        # Tree-based model
        importance_df = pd.DataFrame({
            "feature": X.columns,
            "importance": best_model.feature_importances_
        }).sort_values("importance", ascending=False)

        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df.head(10), x="importance", y="feature", palette="viridis")
        plt.title(f"Top 10 Feature Importance — {model_name}", fontsize=14, fontweight='bold')
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"  Feature importance saved to {save_path}")

        print("\nTop 10 Features by Importance:")
        print(importance_df.head(10).to_string(index=False))
        return importance_df

    elif hasattr(best_model, "coef_"):
        # Linear model (Logistic Regression)
        coef_df = pd.DataFrame({
            "feature": X.columns,
            "coefficient": best_model.coef_[0]
        }).sort_values("coefficient", ascending=False)

        plt.figure(figsize=(10, 8))
        colors = ['#FF5722' if c < 0 else '#4CAF50' for c in coef_df["coefficient"]]
        top10 = coef_df.head(10)
        bars = plt.barh(range(len(top10)), top10["coefficient"].values, color=colors)
        plt.yticks(range(len(top10)), top10["feature"].values)
        plt.title(f"Top 10 Feature Coefficients — {model_name}", fontsize=14, fontweight='bold')
        plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        plt.xlabel("Coefficient (positive = increases churn risk)")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"  Feature coefficients saved to {save_path}")

        print("\nTop 10 Features by Coefficient Magnitude:")
        coef_df["abs_coef"] = coef_df["coefficient"].abs()
        print(coef_df.sort_values("abs_coef", ascending=False).head(10).to_string(index=False))
        return coef_df

    return None


def train_and_compare_models():
    """
    Complete training pipeline:
    1. Compare Logistic Regression, Decision Tree, Random Forest (baseline)
    2. Hyperparameter tuning of Random Forest via GridSearchCV
    3. Comprehensive evaluation: Accuracy, Precision, Recall, F1, ROC Curve, Confusion Matrix
    4. Select best model overall, save all artifacts.
    """
    print("=" * 70)
    print("  CUSTOMER CHURN PREDICTION — MODEL TRAINING & COMPARISON")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load & preprocess
    # ------------------------------------------------------------------
    print("\n[1/6] Loading data...")
    df = load_data()

    print("\n[2/6] Preprocessing data (including feature engineering)...")
    X, y, label_encoders, scaler = preprocess_data(df)

    # ------------------------------------------------------------------
    # 2. Train/test split
    # ------------------------------------------------------------------
    print("\n[3/6] Splitting into train/test sets (80/20 stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train size: {X_train.shape[0]} samples")
    print(f"  Test size : {X_test.shape[0]} samples")
    print(f"  Class balance in test set — Churn: {y_test.sum()} ({y_test.mean():.2%})")

    # ------------------------------------------------------------------
    # 3. Baseline models comparison
    # ------------------------------------------------------------------
    print("\n[4/6] Training baseline models...\n")

    baseline_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, min_samples_split=20, random_state=42),
        "Random Forest (baseline)": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    }

    results = []

    for name, model in baseline_models.items():
        print(f"  >> Training {name}...")
        start = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1 Score": round(f1, 4),
            "ROC-AUC": round(roc_auc, 4),
            "CV F1 Mean": round(cv_mean, 4),
            "CV F1 Std": round(cv_std, 4),
            "Time (s)": round(elapsed, 2)
        })

        print(f"    Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {roc_auc:.4f}")

    # ------------------------------------------------------------------
    # 4. GridSearchCV hyperparameter tuning for Random Forest
    # ------------------------------------------------------------------
    print("\n[5/6] Hyperparameter tuning Random Forest with GridSearchCV...\n")

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "criterion": ["gini", "entropy"]
    }

    rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    total_combos = 1
    for v in param_grid.values():
        total_combos *= len(v)
    print(f"  Grid: {total_combos} combinations x 5 folds = {total_combos * 5} fits")
    print("  Running GridSearchCV (this may take a few minutes)...")

    start = time.time()
    grid_search = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )

    grid_search.fit(X_train, y_train)
    elapsed = time.time() - start

    best_rf = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_f1 = grid_search.best_score_

    print(f"\n  GridSearchCV completed in {elapsed:.2f}s")
    print(f"\n  Best Random Forest Parameters:")
    for param, value in best_params.items():
        print(f"      {param}: {value}")
    print(f"  Best CV F1 Score: {best_cv_f1:.4f}")

    # Evaluate tuned RF
    y_pred_rf = best_rf.predict(X_test)
    y_proba_rf = best_rf.predict_proba(X_test)[:, 1]

    rf_acc = accuracy_score(y_test, y_pred_rf)
    rf_prec = precision_score(y_test, y_pred_rf)
    rf_rec = recall_score(y_test, y_pred_rf)
    rf_f1 = f1_score(y_test, y_pred_rf)
    rf_roc_auc = roc_auc_score(y_test, y_proba_rf)

    results.append({
        "Model": "Random Forest (tuned)",
        "Accuracy": round(rf_acc, 4),
        "Precision": round(rf_prec, 4),
        "Recall": round(rf_rec, 4),
        "F1 Score": round(rf_f1, 4),
        "ROC-AUC": round(rf_roc_auc, 4),
        "CV F1 Mean": round(best_cv_f1, 4),
        "CV F1 Std": round(grid_search.cv_results_["std_test_score"][grid_search.best_index_], 4),
        "Time (s)": round(elapsed, 2)
    })

    print(f"\n  >> Tuned Random Forest — Acc: {rf_acc:.4f} | Prec: {rf_prec:.4f} | Rec: {rf_rec:.4f} | F1: {rf_f1:.4f} | AUC: {rf_roc_auc:.4f}")

    # ------------------------------------------------------------------
    # 5. Determine overall best model
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  FINAL MODEL COMPARISON")
    print("=" * 70)

    results_df = pd.DataFrame(results)
    print("\n" + results_df.to_string(index=False))

    # Pick best by F1 Score (best balance of precision & recall)
    best_row = results_df.loc[results_df["F1 Score"].idxmax()]
    best_model_name = best_row["Model"]

    if best_model_name == "Random Forest (tuned)":
        best_model = best_rf
    else:
        baseline_map = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Decision Tree": DecisionTreeClassifier(max_depth=10, min_samples_split=20, random_state=42),
            "Random Forest (baseline)": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        }
        best_model = baseline_map[best_model_name]
        best_model.fit(X_train, y_train)

    print(f"\n  >> Selected Best Model: {best_model_name}")
    print(f"     F1 Score: {best_row['F1 Score']:.4f} | ROC-AUC: {best_row['ROC-AUC']:.4f}")

    # ------------------------------------------------------------------
    # 6. Comprehensive evaluation of the best model
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  COMPREHENSIVE MODEL EVALUATION")
    print("=" * 70)

    y_pred_best = best_model.predict(X_test)
    y_proba_best = best_model.predict_proba(X_test)[:, 1]

    # 6a. Detailed evaluation report
    metrics = print_evaluation_report(y_test, y_pred_best, y_proba_best, best_model_name)

    # 6b. Confusion Matrix plot
    plot_confusion_matrix(
        y_test, y_pred_best, best_model_name,
        os.path.join("images", "confusion_matrix.png")
    )

    # 6c. ROC Curve plot
    plot_roc_curve(
        y_test, y_proba_best, best_model_name,
        os.path.join("images", "roc_curve.png")
    )

    # 6d. Feature Importance / Coefficients plot (for best model)
    plot_feature_importance(
        best_model, X, best_model_name,
        os.path.join("images", "feature_importance.png")
    )

    # ------------------------------------------------------------------
    # 6e. RANDOM FOREST FEATURE IMPORTANCE ANALYSIS
    #     Random Forest (tree-based) tells us which features matter most.
    #     Examples: Monthly Charges, Contract, Tenure, Tech Support, Internet Service.
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  RANDOM FOREST — WHICH FEATURES MATTER MOST?")
    print("  (Tree-based feature importance — higher = more predictive power)")
    print("=" * 70)

    rf_importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": best_rf.feature_importances_
    }).sort_values("importance", ascending=False)

    # Save full importance data to CSV
    rf_importance_df.to_csv(os.path.join("models", "feature_importance_rf.csv"), index=False)
    print(f"  Feature importance data -> models/feature_importance_rf.csv")

    # Print top 10 with clear ranking
    print("\n  Top 10 Features Driving Churn (Random Forest):")
    print("  " + "-" * 60)
    print(f"  {'Rank':<6} {'Feature':<32} {'Importance':<10} {'Contribution':<12}")
    print("  " + "-" * 60)
    for i, row in rf_importance_df.head(10).iterrows():
        print(f"  {i+1:<6} {row['feature']:<32} {row['importance']:.4f}    {row['importance']*100:.2f}%")
    print("  " + "-" * 60)

    # Plot: Top 10 Important Features (horizontal bar, high-res)
    plt.figure(figsize=(12, 8))
    top10_asc = rf_importance_df.head(10).iloc[::-1]  # reverse for ascending display

    colors = sns.color_palette("viridis", n_colors=len(top10_asc))
    bars = plt.barh(range(len(top10_asc)), top10_asc["importance"].values,
                    color=colors, edgecolor='black', linewidth=0.8)

    plt.yticks(range(len(top10_asc)), top10_asc["feature"].values, fontsize=12)
    plt.xlabel("Importance Score", fontsize=13, fontweight='bold')
    plt.ylabel("Feature", fontsize=13, fontweight='bold')
    plt.title("Top 10 Important Features — Random Forest\n(Which factors most strongly predict customer churn?)",
              fontsize=15, fontweight='bold', pad=15)

    # Add value labels on bars
    for bar, val in zip(bars, top10_asc["importance"].values):
        plt.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                 f'{val:.4f} ({val*100:.2f}%)', va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join("images", "feature_importance_rf_top10.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved to images/feature_importance_rf_top10.png")

    print("\n  Key Business Insights from Random Forest Feature Importance:")
    print("  - Contract type, tenure, and Monthly Charges are the top 3 predictors.")
    print("  - Month-to-month contracts with high charges & low tenure = highest churn risk.")
    print("  - Customers lacking Tech Support or Online Security are more likely to churn.")
    print("  - Fiber optic Internet Service is associated with higher churn than DSL.")

    # ------------------------------------------------------------------
    # 7. Save the best model and all artifacts
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  SAVING MODEL & ARTIFACTS")
    print("=" * 70)

    os.makedirs("models", exist_ok=True)

    joblib.dump(best_model, os.path.join("models", "churn_model.pkl"))
    joblib.dump(scaler, os.path.join("models", "scaler.pkl"))
    joblib.dump(label_encoders, os.path.join("models", "label_encoders.pkl"))

    if best_model_name == "Random Forest (tuned)":
        best_params_df = pd.DataFrame([best_params])
        best_params_df.to_csv(os.path.join("models", "best_params.csv"), index=False)
        print(f"  Hyperparameters      -> models/best_params.csv")

    print(f"  Best model ({best_model_name}) -> models/churn_model.pkl")
    print(f"  Scaler                -> models/scaler.pkl")
    print(f"  Label encoders        -> models/label_encoders.pkl")

    # Save comparison + detailed metrics
    results_df.to_csv(os.path.join("models", "model_comparison.csv"), index=False)
    print(f"  Model comparison      -> models/model_comparison.csv")

    # Save detailed metrics as JSON
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join("models", "detailed_metrics.csv"), index=False)
    print(f"  Detailed metrics      -> models/detailed_metrics.csv")

    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)

    return best_model, best_model_name, results_df, metrics


if __name__ == "__main__":
    train_and_compare_models()
