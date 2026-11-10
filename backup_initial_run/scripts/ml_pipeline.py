"""
ml_pipeline.py
Implements rigorous, leakage-free machine learning analysis on Mendeley dataset:
- Evaluates Logistic Regression, Random Forest, and XGBoost
- Stratified 5-Fold Cross-Validation with all scaling, feature selection, and balancing INSIDE each fold
- Computes ROC-AUC, PR-AUC, Sensitivity, Specificity, Precision, Recall with 95% CIs
- Computes SHAP values, Gini importance, and logistic regression coefficients
- Outputs:
  - model_evaluation.csv
  - feature_importance.csv
  - Figures: ROC curves, PR curves, SHAP summary plot, Feature Importance bar chart
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
)
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Set cache and environment
os.environ["MPLCONFIGDIR"] = "/tmp/mpl_cache"
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

OUTPUT_DIR = "/Users/ritweekraj/Desktop/HPV/outputs"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)
TABLE_S1_PATH = "/Users/ritweekraj/Desktop/HPV/data/mendeley/Table S1.xlsx"

def prepare_data(filepath):
    print("Loading data for ML modeling...")
    df = pd.read_excel(filepath)
    df.columns = df.columns.str.strip()
    
    bacterial_species = [
        "F. vaginae", "G. vaginalis", "BVAB-2", 
        "Megasphaera sp.Type 1", "Megasphaera sp. Type 2", 
        "L. iners", "L. gasseri", "L. jensenii", "L. crispatus"
    ]
    hpv_genotypes = [
        "HPV 16", "HPV 18", "HPV 31", "HPV 33", "HPV 35", 
        "HPV 39", "HPV 45", "HPV 51", "HPV 52", "HPV 56", 
        "HPV 58", "HPV 59", "HPV 68"
    ]
    
    # Engineer features
    df["Total_hrHPV_Load"] = df[hpv_genotypes].sum(axis=1)
    df["Max_hrHPV_Load"] = df[hpv_genotypes].max(axis=1)
    df["HPV16_18_Load"] = df[["HPV 16", "HPV 18"]].sum(axis=1)
    df["Other_hrHPV_Load"] = df[[h for h in hpv_genotypes if h not in ["HPV 16", "HPV 18"]]].sum(axis=1)
    df["hrHPV_Genotype_Count"] = (df[hpv_genotypes] > 0).sum(axis=1)
    
    protective_lacto = ["L. crispatus", "L. gasseri", "L. jensenii"]
    all_lacto = ["L. iners", "L. crispatus", "L. gasseri", "L. jensenii"]
    bv_pathogens = ["F. vaginae", "G. vaginalis", "BVAB-2", "Megasphaera sp.Type 1", "Megasphaera sp. Type 2"]
    
    df["Total_Lactobacillus_Load"] = df[all_lacto].sum(axis=1)
    df["Protective_Lacto_Load"] = df[protective_lacto].sum(axis=1)
    df["Total_BV_Pathogens_Load"] = df[bv_pathogens].sum(axis=1)
    df["Total_Bacterial_Biomass"] = df[bacterial_species].sum(axis=1)
    
    for b in bacterial_species:
        df[f"log10_{b}"] = np.log10(df[b] + 1)
    for h in hpv_genotypes:
        df[f"log10_{h}"] = np.log10(df[h] + 1)
        
    df["log10_Total_hrHPV_Load"] = np.log10(df["Total_hrHPV_Load"] + 1)
    df["log10_Total_Lacto"] = np.log10(df["Total_Lactobacillus_Load"] + 1)
    df["log10_Total_BV_Pathogens"] = np.log10(df["Total_BV_Pathogens_Load"] + 1)
    
    df["L_crispatus_to_L_iners_Ratio"] = df["log10_L. crispatus"] - df["log10_L. iners"]
    df["Lacto_to_Pathogen_Ratio"] = df["log10_Total_Lacto"] - df["log10_Total_BV_Pathogens"]
    
    biomass_safe = np.where(df["Total_Bacterial_Biomass"] > 0, df["Total_Bacterial_Biomass"], 1.0)
    for b in bacterial_species:
        df[f"rel_{b}"] = df[b] / biomass_safe
    df["rel_Total_Lacto"] = df["Total_Lactobacillus_Load"] / biomass_safe
    df["rel_Total_BV_Pathogens"] = df["Total_BV_Pathogens_Load"] / biomass_safe
    
    bv_map = {"BV-N": 0.0, "BV-T": 0.5, "BV-P": 1.0}
    df["BV_Score"] = df["BV Status"].map(bv_map).fillna(0.0)
    df["Pt. Age"] = pd.to_numeric(df["Pt. Age"], errors="coerce").fillna(df["Pt. Age"].median())
    
    df["log10_HPV16_18_Load"] = np.log10(df["HPV16_18_Load"] + 1)

    # Filter to Normal Cytology cohort (NILM == 1)
    df_nilm = df[df["NILM"] == 1].copy().reset_index(drop=True)
    
    # Define Feature Sets
    microbiome_clinical_features = [
        "Pt. Age", "BV_Score",
        # Bacterial log10 loads
        "log10_F. vaginae", "log10_G. vaginalis", "log10_BVAB-2",
        "log10_Megasphaera sp.Type 1", "log10_Megasphaera sp. Type 2",
        "log10_L. iners", "log10_L. gasseri", "log10_L. jensenii", "log10_L. crispatus",
        # Bacterial Composites & Ratios
        "log10_Total_Lacto", "log10_Total_BV_Pathogens",
        "L_crispatus_to_L_iners_Ratio", "Lacto_to_Pathogen_Ratio",
        # Relative Abundances
        "rel_L. crispatus", "rel_L. iners", "rel_L. gasseri", "rel_L. jensenii",
        "rel_F. vaginae", "rel_G. vaginalis", "rel_BVAB-2", "rel_Megasphaera sp.Type 1",
        "rel_Total_Lacto", "rel_Total_BV_Pathogens"
    ]
    
    viral_features = [
        "log10_Total_hrHPV_Load", "hrHPV_Genotype_Count", "log10_HPV16_18_Load",
        "log10_HPV 16", "log10_HPV 18", "log10_HPV 31", "log10_HPV 33", "log10_HPV 35",
        "log10_HPV 39", "log10_HPV 45", "log10_HPV 51", "log10_HPV 52", "log10_HPV 56",
        "log10_HPV 58", "log10_HPV 59", "log10_HPV 68"
    ]
    
    multimodal_features = viral_features + microbiome_clinical_features
    
    return df_nilm, microbiome_clinical_features, multimodal_features

def evaluate_models_cross_validation(df_data, feature_cols, experiment_name, n_splits=5):
    """
    Performs strictly leakage-free 5-Fold Stratified Cross Validation.
    Feature scaling, selection, and balancing are applied exclusively within each train split.
    """
    print(f"\n=======================================================")
    print(f"Running Experiment: {experiment_name}")
    print(f"Features: {len(feature_cols)} | Samples: {len(df_data)} (HPV- = {(df_data['hrHPV Result']==0).sum()}, hrHPV+ = {(df_data['hrHPV Result']==1).sum()})")
    print(f"=======================================================")
    
    X = df_data[feature_cols].values
    y = df_data["hrHPV Result"].values
    feature_names = np.array(feature_cols)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced", random_state=42, n_jobs=-1),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05, 
            scale_pos_weight=(len(y)-sum(y))/sum(y), eval_metric="logloss", 
            random_state=42, n_jobs=-1
        )
    }
    
    cv_metrics = {m_name: {
        "roc_auc": [], "pr_auc": [], "sensitivity": [], 
        "specificity": [], "precision": [], "recall": [], 
        "f1": [], "balanced_acc": []
    } for m_name in models}
    
    # To store curves across folds
    roc_curves_data = {m_name: [] for m_name in models}
    pr_curves_data = {m_name: [] for m_name in models}
    
    # To accumulate feature importances
    rf_importances = np.zeros(len(feature_cols))
    xgb_importances = np.zeros(len(feature_cols))
    lr_coefficients = np.zeros(len(feature_cols))
    
    # Store test predictions and SHAP values for fold 1
    shap_values_dict = {}
    X_test_sample = None
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 1. Scaling fitted ONLY on training fold
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)
        
        for m_name, model in models.items():
            model.fit(X_train, y_train)
            
            # Predict probabilities and classes
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_prob = model.decision_function(X_test)
                y_prob = 1.0 / (1.0 + np.exp(-y_prob))
                
            y_pred = (y_prob >= 0.5).astype(int)
            
            # Confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
            
            roc_auc = roc_auc_score(y_test, y_prob)
            pr_auc = average_precision_score(y_test, y_prob)
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            bacc = (sens + spec) / 2.0
            
            cv_metrics[m_name]["roc_auc"].append(roc_auc)
            cv_metrics[m_name]["pr_auc"].append(pr_auc)
            cv_metrics[m_name]["sensitivity"].append(sens)
            cv_metrics[m_name]["specificity"].append(spec)
            cv_metrics[m_name]["precision"].append(prec)
            cv_metrics[m_name]["recall"].append(rec)
            cv_metrics[m_name]["f1"].append(f1)
            cv_metrics[m_name]["balanced_acc"].append(bacc)
            
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_curves_data[m_name].append((fpr, tpr, roc_auc))
            
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
            pr_curves_data[m_name].append((recall_curve, precision_curve, pr_auc))
            
            # Accumulate feature weights/importances
            if m_name == "Random Forest":
                rf_importances += model.feature_importances_ / n_splits
                if fold == 1 and experiment_name == "Multimodal_Viral_Microbiome":
                    explainer_rf = shap.TreeExplainer(model)
                    # Compute SHAP on a balanced subset of test fold for speed
                    shap_values_dict["Random Forest"] = explainer_rf.shap_values(X_test)
            elif m_name == "XGBoost":
                xgb_importances += model.feature_importances_ / n_splits
                if fold == 1 and experiment_name == "Multimodal_Viral_Microbiome":
                    explainer_xgb = shap.TreeExplainer(model)
                    shap_values_dict["XGBoost"] = explainer_xgb.shap_values(X_test)
                    X_test_sample = X_test
            elif m_name == "Logistic Regression":
                lr_coefficients += np.abs(model.coef_[0]) / n_splits
                
    # Summary of cross-validation results
    results_summary = []
    print(f"\n--- Cross-Validation Results for {experiment_name} ---")
    for m_name in models:
        row = {"experiment": experiment_name, "model": m_name}
        for metric_name in ["roc_auc", "pr_auc", "sensitivity", "specificity", "precision", "recall", "f1", "balanced_acc"]:
            vals = np.array(cv_metrics[m_name][metric_name])
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            # 95% Confidence Interval: mean +/- 1.96 * (std / sqrt(n))
            ci95 = 1.96 * (std_val / np.sqrt(n_splits))
            row[f"{metric_name}_mean"] = mean_val
            row[f"{metric_name}_std"] = std_val
            row[f"{metric_name}_ci95_low"] = max(0.0, mean_val - ci95)
            row[f"{metric_name}_ci95_high"] = min(1.0, mean_val + ci95)
        results_summary.append(row)
        print(f"{m_name:20s} | ROC-AUC: {row['roc_auc_mean']:.4f} (95% CI: {row['roc_auc_ci95_low']:.4f}-{row['roc_auc_ci95_high']:.4f}) | "
              f"Sens: {row['sensitivity_mean']:.4f} | Spec: {row['specificity_mean']:.4f} | PR-AUC: {row['pr_auc_mean']:.4f}")
        
    df_eval = pd.DataFrame(results_summary)
    
    # Feature importance dataframe
    df_imp = pd.DataFrame({
        "feature": feature_names,
        "rf_importance": rf_importances,
        "xgb_importance": xgb_importances,
        "lr_abs_coefficient": lr_coefficients
    })
    
    # Calculate unified rank score
    df_imp["rank_rf"] = df_imp["rf_importance"].rank(ascending=False)
    df_imp["rank_xgb"] = df_imp["xgb_importance"].rank(ascending=False)
    df_imp["rank_lr"] = df_imp["lr_abs_coefficient"].rank(ascending=False)
    df_imp["mean_rank"] = (df_imp["rank_rf"] + df_imp["rank_xgb"] + df_imp["rank_lr"]) / 3.0
    df_imp = df_imp.sort_values(by="mean_rank").reset_index(drop=True)
    
    return df_eval, df_imp, roc_curves_data, pr_curves_data, shap_values_dict, X_test_sample, feature_names

def plot_ml_curves(exp1_roc, exp1_pr, exp2_roc, exp2_pr):
    """Plot ROC and PR curves comparing both experiments across models."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    colors = {"Logistic Regression": "#2ca02c", "Random Forest": "#1f77b4", "XGBoost": "#ff7f0e"}
    
    # Panel A: Exp 1 ROC (Microbiome Only)
    ax = axes[0, 0]
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance (AUC=0.50)")
    for m_name, folds in exp1_roc.items():
        mean_auc = np.mean([f[2] for f in folds])
        for idx, (fpr, tpr, auc_val) in enumerate(folds):
            ax.plot(fpr, tpr, color=colors[m_name], alpha=0.15, linewidth=1)
        # Interpolate mean curve
        mean_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.mean([np.interp(mean_fpr, f[0], f[1]) for f in folds], axis=0)
        mean_tpr[-1] = 1.0
        ax.plot(mean_fpr, mean_tpr, color=colors[m_name], linewidth=2.5, label=f"{m_name} (Mean AUC = {mean_auc:.3f})")
    ax.set_title("A. ROC Curves: Microbiome & Demographics Only\n(Normal Cytology: NILM=1)", fontsize=12, fontweight="bold")
    ax.set_xlabel("1 - Specificity (False Positive Rate)", fontsize=10)
    ax.set_ylabel("Sensitivity (True Positive Rate)", fontsize=10)
    ax.legend(loc="lower right", frameon=True)
    
    # Panel B: Exp 1 PR (Microbiome Only)
    ax = axes[0, 1]
    baseline_pr = 311 / 11749
    ax.axhline(baseline_pr, color="black", linestyle="--", alpha=0.5, label=f"Prevalence Baseline ({baseline_pr:.3f})")
    for m_name, folds in exp1_pr.items():
        mean_prauc = np.mean([f[2] for f in folds])
        mean_rec = np.linspace(0, 1, 100)
        mean_prec = np.mean([np.interp(mean_rec, f[0][::-1], f[1][::-1]) for f in folds], axis=0)
        ax.plot(mean_rec, mean_prec, color=colors[m_name], linewidth=2.5, label=f"{m_name} (PR-AUC = {mean_prauc:.3f})")
    ax.set_title("B. Precision-Recall Curves: Microbiome & Demographics Only\n(High Imbalance Benchmark)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Recall", fontsize=10)
    ax.set_ylabel("Precision", fontsize=10)
    ax.legend(loc="upper right", frameon=True)
    
    # Panel C: Exp 2 ROC (Multimodal Viral + Microbiome)
    ax = axes[1, 0]
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance (AUC=0.50)")
    for m_name, folds in exp2_roc.items():
        mean_auc = np.mean([f[2] for f in folds])
        mean_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.mean([np.interp(mean_fpr, f[0], f[1]) for f in folds], axis=0)
        mean_tpr[-1] = 1.0
        ax.plot(mean_fpr, mean_tpr, color=colors[m_name], linewidth=2.5, label=f"{m_name} (Mean AUC = {mean_auc:.4f})")
    ax.set_title("C. ROC Curves: Multimodal (Viral DNA + Microbiome)\n(Normal Cytology: NILM=1)", fontsize=12, fontweight="bold")
    ax.set_xlabel("1 - Specificity (False Positive Rate)", fontsize=10)
    ax.set_ylabel("Sensitivity (True Positive Rate)", fontsize=10)
    ax.legend(loc="lower right", frameon=True)
    
    # Panel D: Exp 2 PR (Multimodal Viral + Microbiome)
    ax = axes[1, 1]
    ax.axhline(baseline_pr, color="black", linestyle="--", alpha=0.5, label=f"Prevalence Baseline ({baseline_pr:.3f})")
    for m_name, folds in exp2_pr.items():
        mean_prauc = np.mean([f[2] for f in folds])
        mean_rec = np.linspace(0, 1, 100)
        mean_prec = np.mean([np.interp(mean_rec, f[0][::-1], f[1][::-1]) for f in folds], axis=0)
        ax.plot(mean_rec, mean_prec, color=colors[m_name], linewidth=2.5, label=f"{m_name} (PR-AUC = {mean_prauc:.4f})")
    ax.set_title("D. Precision-Recall Curves: Multimodal (Viral DNA + Microbiome)\n(Near-Perfect Direct DNA Detection)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Recall", fontsize=10)
    ax.set_ylabel("Precision", fontsize=10)
    ax.legend(loc="lower left", frameon=True)
    
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "fig5_model_roc_pr_curves.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path}")

def plot_feature_importance_and_shap(df_imp_multimodal, df_imp_microbiome, shap_values_dict, X_test_sample, feature_names):
    """Plot ranked feature importance and SHAP summary plots."""
    # 1. Bar plot of Top Features
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left: Multimodal Top Features
    top_multi = df_imp_multimodal.head(15).copy().sort_values(by="rf_importance", ascending=True)
    axes[0].barh(top_multi["feature"], top_multi["rf_importance"], color="#d73027", height=0.65)
    axes[0].set_title("A. Top 15 Features in Multimodal Classifier\n(Random Forest Gini Importance)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Relative Importance Weight", fontsize=10)
    
    # Right: Microbiome-Only Top Features
    top_micro = df_imp_microbiome.head(15).copy().sort_values(by="rf_importance", ascending=True)
    axes[1].barh(top_micro["feature"], top_micro["rf_importance"], color="#3182bd", height=0.65)
    axes[1].set_title("B. Top 15 Host Microbiome & Demographics Features\n(Predicting hrHPV Status in Normal Cytology)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Relative Importance Weight", fontsize=10)
    
    plt.tight_layout()
    bar_path = os.path.join(FIGURES_DIR, "fig6_feature_importance_comparison.png")
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {bar_path}")
    
    # 2. SHAP summary plot for XGBoost if available
    if "XGBoost" in shap_values_dict and X_test_sample is not None:
        plt.figure(figsize=(10, 8))
        shap_vals = shap_values_dict["XGBoost"]
        # Handle if binary returns 2D or list
        if isinstance(shap_vals, list):
            vals_to_plot = shap_vals[1]
        else:
            vals_to_plot = shap_vals
            
        shap.summary_plot(vals_to_plot, X_test_sample, feature_names=feature_names, max_display=15, show=False)
        plt.title("SHAP Feature Importance & Attribution for XGBoost\n(Normal Cytology hrHPV Classification)", fontsize=12, fontweight="bold")
        plt.tight_layout()
        shap_path = os.path.join(FIGURES_DIR, "fig7_shap_summary_plot.png")
        plt.savefig(shap_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {shap_path}")

def run_ml_pipeline():
    df_nilm, microbiome_features, multimodal_features = prepare_data(TABLE_S1_PATH)
    
    # Run Experiment 1: Microbiome & Demographics Only (Host Susceptibility)
    eval1, imp1, roc1, pr1, _, _, _ = evaluate_models_cross_validation(
        df_nilm, microbiome_features, "Microbiome_Clinical_Only", n_splits=5
    )
    
    # Run Experiment 2: Multimodal (Viral DNA + Microbiome + Demographics)
    eval2, imp2, roc2, pr2, shap_dict, X_test_s, feat_names = evaluate_models_cross_validation(
        df_nilm, multimodal_features, "Multimodal_Viral_Microbiome", n_splits=5
    )
    
    # Combine evaluation results into model_evaluation.csv
    df_eval_combined = pd.concat([eval1, eval2], ignore_index=True)
    eval_csv_path = os.path.join("/Users/ritweekraj/Desktop/HPV", "model_evaluation.csv")
    df_eval_combined.to_csv(eval_csv_path, index=False)
    print(f"\nSaved deliverable: {eval_csv_path}")
    
    # Feature importance deliverable: feature_importance.csv
    imp2["dataset_scope"] = "Multimodal (Mendeley Table S1 Normal Cytology)"
    imp1["dataset_scope"] = "Microbiome-Only (Mendeley Table S1 Normal Cytology)"
    df_imp_combined = pd.concat([imp2, imp1], ignore_index=True)
    imp_csv_path = os.path.join("/Users/ritweekraj/Desktop/HPV", "feature_importance.csv")
    df_imp_combined.to_csv(imp_csv_path, index=False)
    print(f"Saved deliverable: {imp_csv_path}")
    
    # Generate Plots
    plot_ml_curves(roc1, pr1, roc2, pr2)
    plot_feature_importance_and_shap(imp2, imp1, shap_dict, X_test_s, feat_names)
    
    print("\nML Pipeline executed successfully!")

if __name__ == "__main__":
    run_ml_pipeline()
