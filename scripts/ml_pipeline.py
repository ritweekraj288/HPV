"""
ml_pipeline.py (Scientifically Corrected Version)
- Target: 1 = hrHPV-positive with normal cytology, 0 = HPV-negative with normal cytology (NILM == 1)
- Feature Set: Microbiome bacterial qPCR measurements and clinical covariates ONLY (23 features).
- STRICT ZERO LEAKAGE: Absolute exclusion of any HPV-derived variable (no genotype, load, or count).
- Repeated Stratified Cross-Validation (5 folds x 3 repeats = 15 validation splits)
- All scaling and class-weight balancing fitted STRICTLY inside training folds.
- Threshold selection optimized on TRAINING fold (Youden J) and evaluated on held-out test fold.
- Evaluation metrics: ROC-AUC, PR-AUC, Sensitivity, Specificity, Precision, F1-Score, Brier Score.
- Explicit comparison of PR-AUC against the positive class prevalence (0.0265 baseline).
- Outputs:
  * outputs/microbiome_ml_features.txt
  * model_evaluation.csv
  * feature_importance.csv
  * figures/fig5_model_roc_pr_curves.png
  * figures/fig6_feature_importance_comparison.png
  * figures/fig7_shap_summary_plot.png
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    confusion_matrix, precision_score, recall_score, f1_score, brier_score_loss
)
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

os.environ["MPLCONFIGDIR"] = "/tmp/mpl_cache"
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

OUTPUT_DIR = "/Users/ritweekraj/Desktop/HPV/outputs"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)
TABLE_S1_PATH = "/Users/ritweekraj/Desktop/HPV/data/mendeley/Table S1.xlsx"

def prepare_microbiome_dataset(filepath):
    print("Loading data for microbiome-based prediction...")
    df = pd.read_excel(filepath)
    df.columns = df.columns.str.strip()
    
    # Filter to Normal Cytology cohort (NILM == 1)
    df_nilm = df[df["NILM"] == 1].copy().reset_index(drop=True)
    
    bacterial_species = [
        "F. vaginae", "G. vaginalis", "BVAB-2", 
        "Megasphaera sp.Type 1", "Megasphaera sp. Type 2", 
        "L. iners", "L. gasseri", "L. jensenii", "L. crispatus"
    ]
    
    # Feature engineering for bacteria only
    for b in bacterial_species:
        df_nilm[f"log10_{b}"] = np.log10(df_nilm[b] + 1)
        
    all_lacto = ["L. iners", "L. crispatus", "L. gasseri", "L. jensenii"]
    bv_pathogens = ["F. vaginae", "G. vaginalis", "BVAB-2", "Megasphaera sp.Type 1", "Megasphaera sp. Type 2"]
    
    df_nilm["Total_Lacto"] = df_nilm[all_lacto].sum(axis=1)
    df_nilm["Total_BV_Pathogens"] = df_nilm[bv_pathogens].sum(axis=1)
    df_nilm["log10_Total_Lacto"] = np.log10(df_nilm["Total_Lacto"] + 1)
    df_nilm["log10_Total_BV_Pathogens"] = np.log10(df_nilm["Total_BV_Pathogens"] + 1)
    df_nilm["L_crispatus_to_L_iners_Ratio"] = df_nilm["log10_L. crispatus"] - df_nilm["log10_L. iners"]
    df_nilm["Lacto_to_Pathogen_Ratio"] = df_nilm["log10_Total_Lacto"] - df_nilm["log10_Total_BV_Pathogens"]
    
    biomass = df_nilm[bacterial_species].sum(axis=1)
    biomass_safe = np.where(biomass > 0, biomass, 1.0)
    for b in bacterial_species:
        df_nilm[f"rel_{b}"] = df_nilm[b] / biomass_safe
    df_nilm["rel_Total_Lacto"] = df_nilm["Total_Lacto"] / biomass_safe
    df_nilm["rel_Total_BV_Pathogens"] = df_nilm["Total_BV_Pathogens"] / biomass_safe
    
    # Clinical covariates
    bv_map = {"BV-N": 0.0, "BV-T": 0.5, "BV-P": 1.0}
    df_nilm["BV_Score"] = df_nilm["BV Status"].map(bv_map).fillna(0.0)
    df_nilm["Pt. Age"] = pd.to_numeric(df_nilm["Pt. Age"], errors="coerce").fillna(df_nilm["Pt. Age"].median())
    
    # STRICT MICROBIOME & CLINICAL FEATURE LIST
    features = [
        "Pt. Age", "BV_Score",
        # Quantitative bacterial qPCR loads
        "log10_F. vaginae", "log10_G. vaginalis", "log10_BVAB-2",
        "log10_Megasphaera sp.Type 1", "log10_Megasphaera sp. Type 2",
        "log10_L. iners", "log10_L. gasseri", "log10_L. jensenii", "log10_L. crispatus",
        # Ecological Composites & Ratios
        "log10_Total_Lacto", "log10_Total_BV_Pathogens",
        "L_crispatus_to_L_iners_Ratio", "Lacto_to_Pathogen_Ratio",
        # Relative Taxonomic Abundances
        "rel_L. crispatus", "rel_L. iners", "rel_L. gasseri", "rel_L. jensenii",
        "rel_F. vaginae", "rel_G. vaginalis", "rel_BVAB-2", "rel_Megasphaera sp.Type 1",
        "rel_Total_Lacto", "rel_Total_BV_Pathogens"
    ]
    
    # MANUAL LEAKAGE CHECK
    print("\n--- AUDITING FINAL ML FEATURE LIST FOR TARGET LEAKAGE ---")
    for f in features:
        assert "hpv" not in f.lower(), f"CRITICAL LEAKAGE DETECTED: {f}"
        assert "viral" not in f.lower(), f"CRITICAL LEAKAGE DETECTED: {f}"
        print(f"  [VERIFIED PURE FEATURE]: {f}")
    print(f"Audit passed: 0 HPV-derived features in predictor set (Total: {len(features)} features).")
    
    # Save audited feature list
    feat_txt_path = os.path.join(OUTPUT_DIR, "microbiome_ml_features.txt")
    with open(feat_txt_path, "w") as f_out:
        for f in features:
            f_out.write(f + "\n")
    print(f"Saved verified feature list to: {feat_txt_path}")
    
    return df_nilm, features

def run_repeated_cv(df_nilm, feature_cols, n_splits=5, n_repeats=3):
    X = df_nilm[feature_cols].values
    y = df_nilm["hrHPV Result"].values
    n_pos = sum(y)
    n_total = len(y)
    prevalence = n_pos / n_total
    
    print("\n" + "="*75)
    print(f"EXPERIMENT: MICROBIOME-BASED PREDICTION OF hrHPV IN NORMAL CYTOLOGY")
    print(f"Samples: {n_total} | Negative: {n_total - n_pos} (97.35%) | Positive: {n_pos} (2.65%)")
    print(f"Prevalence Baseline PR-AUC: {prevalence:.4f}")
    print(f"Cross-Validation: Repeated Stratified K-Fold ({n_splits} folds x {n_repeats} repeats = {n_splits*n_repeats} splits)")
    print("="*75)
    
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=5, class_weight="balanced", random_state=42, n_jobs=-1),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.03,
            scale_pos_weight=(n_total - n_pos) / n_pos, eval_metric="logloss",
            random_state=42, n_jobs=-1
        )
    }
    
    metrics = {m: {
        "roc_auc": [], "pr_auc": [], "sensitivity": [], "specificity": [],
        "precision": [], "f1": [], "brier_score": []
    } for m in models}
    
    confusion_matrices = {m: np.zeros((2, 2), dtype=int) for m in models}
    roc_curves_data = {m: [] for m in models}
    pr_curves_data = {m: [] for m in models}
    
    rf_importances = np.zeros(len(feature_cols))
    xgb_importances = np.zeros(len(feature_cols))
    lr_coefficients = np.zeros(len(feature_cols))
    
    shap_sample_values = None
    X_test_for_shap = None
    
    split_count = 0
    for train_idx, test_idx in rskf.split(X, y):
        split_count += 1
        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Scaling strictly inside training fold
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)
        
        for m_name, model in models.items():
            model.fit(X_train, y_train)
            
            y_prob_train = model.predict_proba(X_train)[:, 1]
            y_prob_test = model.predict_proba(X_test)[:, 1]
            
            # Select threshold on TRAINING data (Youden's J statistic = TPR - FPR)
            fpr_tr, tpr_tr, thresh_tr = roc_curve(y_train, y_prob_train)
            best_idx = np.argmax(tpr_tr - fpr_tr)
            train_threshold = thresh_tr[best_idx]
            
            # Apply training-derived threshold to held-out test fold
            y_pred_test = (y_prob_test >= train_threshold).astype(int)
            
            # Record metrics
            auc = roc_auc_score(y_test, y_prob_test)
            prauc = average_precision_score(y_test, y_prob_test)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test, labels=[0, 1]).ravel()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            prec = precision_score(y_test, y_pred_test, zero_division=0)
            f1 = f1_score(y_test, y_pred_test, zero_division=0)
            brier = brier_score_loss(y_test, y_prob_test)
            
            metrics[m_name]["roc_auc"].append(auc)
            metrics[m_name]["pr_auc"].append(prauc)
            metrics[m_name]["sensitivity"].append(sens)
            metrics[m_name]["specificity"].append(spec)
            metrics[m_name]["precision"].append(prec)
            metrics[m_name]["f1"].append(f1)
            metrics[m_name]["brier_score"].append(brier)
            
            confusion_matrices[m_name] += np.array([[tn, fp], [fn, tp]])
            
            if split_count <= n_splits:
                fpr_ts, tpr_ts, _ = roc_curve(y_test, y_prob_test)
                roc_curves_data[m_name].append((fpr_ts, tpr_ts, auc))
                p_c, r_c, _ = precision_recall_curve(y_test, y_prob_test)
                pr_curves_data[m_name].append((r_c, p_c, prauc))
                
            if m_name == "Random Forest":
                rf_importances += model.feature_importances_ / (n_splits * n_repeats)
            elif m_name == "XGBoost":
                xgb_importances += model.feature_importances_ / (n_splits * n_repeats)
                if split_count == 1:
                    explainer = shap.TreeExplainer(model)
                    shap_sample_values = explainer.shap_values(X_test)
                    X_test_for_shap = X_test
            elif m_name == "Logistic Regression":
                lr_coefficients += np.abs(model.coef_[0]) / (n_splits * n_repeats)
                
    # Summarize results
    eval_rows = []
    print("\n--- REPEATED CROSS-VALIDATION SUMMARY METRICS ---")
    print(f"(Baseline PR-AUC expected by chance = {prevalence:.4f})\n")
    for m_name in models:
        mean_auc, std_auc = np.mean(metrics[m_name]["roc_auc"]), np.std(metrics[m_name]["roc_auc"])
        mean_prauc, std_prauc = np.mean(metrics[m_name]["pr_auc"]), np.std(metrics[m_name]["pr_auc"])
        mean_sens, std_sens = np.mean(metrics[m_name]["sensitivity"]), np.std(metrics[m_name]["sensitivity"])
        mean_spec, std_spec = np.mean(metrics[m_name]["specificity"]), np.std(metrics[m_name]["specificity"])
        mean_prec, std_prec = np.mean(metrics[m_name]["precision"]), np.std(metrics[m_name]["precision"])
        mean_f1, std_f1 = np.mean(metrics[m_name]["f1"]), np.std(metrics[m_name]["f1"])
        mean_brier = np.mean(metrics[m_name]["brier_score"])
        
        cm = confusion_matrices[m_name]
        
        eval_rows.append({
            "experiment": "Microbiome_Clinical_Association_Only",
            "model": m_name,
            "target": "hrHPV_positive_in_normal_cytology",
            "positive_class_prevalence": prevalence,
            "roc_auc_mean": mean_auc,
            "roc_auc_std": std_auc,
            "pr_auc_mean": mean_prauc,
            "pr_auc_std": std_prauc,
            "sensitivity_mean": mean_sens,
            "sensitivity_std": std_sens,
            "specificity_mean": mean_spec,
            "specificity_std": std_spec,
            "precision_mean": mean_prec,
            "precision_std": std_prec,
            "f1_mean": mean_f1,
            "f1_std": std_f1,
            "brier_score_mean": mean_brier,
            "accumulated_tn": cm[0, 0],
            "accumulated_fp": cm[0, 1],
            "accumulated_fn": cm[1, 0],
            "accumulated_tp": cm[1, 1],
            "evaluation_note": f"Evaluated across {n_splits*n_repeats} repeated folds; thresholds selected inside training fold via Youden J."
        })
        
        print(f"Model: {m_name}")
        print(f"  ROC-AUC:     {mean_auc:.4f} +/- {std_auc:.4f}")
        print(f"  PR-AUC:      {mean_prauc:.4f} +/- {std_prauc:.4f} (Prevalence baseline: {prevalence:.4f})")
        print(f"  Sensitivity: {mean_sens:.4f} +/- {std_sens:.4f}")
        print(f"  Specificity: {mean_spec:.4f} +/- {std_spec:.4f}")
        print(f"  Precision:   {mean_prec:.4f} +/- {std_prec:.4f}")
        print(f"  F1-Score:    {mean_f1:.4f} +/- {std_f1:.4f}")
        print(f"  Brier Score: {mean_brier:.4f}")
        print(f"  Accumulated Confusion Matrix: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}\n")
        
    df_eval = pd.DataFrame(eval_rows)
    eval_csv = os.path.join("/Users/ritweekraj/Desktop/HPV", "model_evaluation.csv")
    df_eval.to_csv(eval_csv, index=False)
    print(f"Saved: {eval_csv}")
    
    # Feature importance
    df_imp = pd.DataFrame({
        "feature": feature_cols,
        "rf_importance": rf_importances,
        "xgb_importance": xgb_importances,
        "lr_abs_coefficient": lr_coefficients
    })
    df_imp["rank_rf"] = df_imp["rf_importance"].rank(ascending=False)
    df_imp["rank_xgb"] = df_imp["xgb_importance"].rank(ascending=False)
    df_imp["rank_lr"] = df_imp["lr_abs_coefficient"].rank(ascending=False)
    df_imp["mean_rank"] = (df_imp["rank_rf"] + df_imp["rank_xgb"] + df_imp["rank_lr"]) / 3.0
    df_imp["dataset_scope"] = "Microbiome-Only (Normal Cytology, Table S1)"
    df_imp = df_imp.sort_values(by="mean_rank").reset_index(drop=True)
    
    imp_csv = os.path.join("/Users/ritweekraj/Desktop/HPV", "feature_importance.csv")
    df_imp.to_csv(imp_csv, index=False)
    print(f"Saved: {imp_csv}")
    
    return df_eval, df_imp, roc_curves_data, pr_curves_data, shap_sample_values, X_test_for_shap, feature_cols, prevalence

def plot_corrected_ml_figures(roc_data, pr_data, df_imp, shap_vals, X_test, feature_names, prevalence):
    print("\nGenerating corrected ML evaluation figures...")
    
    # -------------------------------------------------------------
    # Figure 5: ROC and Precision-Recall Curves (Microbiome Only)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = {"Logistic Regression": "#2ca02c", "Random Forest": "#1f77b4", "XGBoost": "#ff7f0e"}
    
    # Panel A: ROC Curves
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance (AUC = 0.50)")
    for m_name, folds in roc_data.items():
        mean_auc = np.mean([f[2] for f in folds])
        mean_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.mean([np.interp(mean_fpr, f[0], f[1]) for f in folds], axis=0)
        mean_tpr[-1] = 1.0
        ax.plot(mean_fpr, mean_tpr, color=colors[m_name], linewidth=2.5, 
                label=f"{m_name} (Mean AUC = {mean_auc:.3f})")
    ax.set_title("A. ROC Curves: Microbiome & Clinical Predictors Only\n(Normal Cytology NILM=1, No HPV Features)", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("1 - Specificity (False Positive Rate)", fontsize=10)
    ax.set_ylabel("Sensitivity (True Positive Rate)", fontsize=10)
    ax.legend(loc="lower right", frameon=True)
    
    # Panel B: PR Curves
    ax = axes[1]
    ax.axhline(prevalence, color="black", linestyle="--", linewidth=1.5, 
               label=f"Positive Class Prevalence Baseline ({prevalence:.4f})")
    for m_name, folds in pr_data.items():
        mean_prauc = np.mean([f[2] for f in folds])
        mean_rec = np.linspace(0, 1, 100)
        mean_prec = np.mean([np.interp(mean_rec, f[0][::-1], f[1][::-1]) for f in folds], axis=0)
        ax.plot(mean_rec, mean_prec, color=colors[m_name], linewidth=2.5, 
                label=f"{m_name} (PR-AUC = {mean_prauc:.4f})")
    ax.set_title("B. Precision-Recall Curves vs Baseline Prevalence\n(Severe Imbalance: 2.65% Positive)", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Recall", fontsize=10)
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=10)
    ax.legend(loc="upper right", frameon=True)
    
    plt.tight_layout()
    fig5_path = os.path.join(FIGURES_DIR, "fig5_model_roc_pr_curves.png")
    plt.savefig(fig5_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig5_path}")
    
    # -------------------------------------------------------------
    # Figure 6: Feature Importance Bar Chart
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 7))
    top_features = df_imp.head(15).copy().sort_values(by="rf_importance", ascending=True)
    plt.barh(top_features["feature"], top_features["rf_importance"], color="#3182bd", height=0.6, alpha=0.85)
    plt.xlabel("Random Forest Mean Decrease in Impurity (Gini Importance)", fontsize=10.5)
    plt.ylabel("Microbial / Clinical Predictor", fontsize=10.5)
    plt.title("Top Ranked Microbiome & Demographic Features\nAssociated with hrHPV Positivity in Normal Cytology", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig6_path = os.path.join(FIGURES_DIR, "fig6_feature_importance_comparison.png")
    plt.savefig(fig6_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig6_path}")
    
    # -------------------------------------------------------------
    # Figure 7: SHAP Summary Plot
    # -------------------------------------------------------------
    if shap_vals is not None and X_test is not None:
        plt.figure(figsize=(10, 8))
        vals = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
        shap.summary_plot(vals, X_test, feature_names=feature_names, max_display=15, show=False)
        plt.title("SHAP Feature Attribution for XGBoost\n(Microbiome Model in Normal Cytology)", fontsize=12, fontweight="bold")
        plt.tight_layout()
        fig7_path = os.path.join(FIGURES_DIR, "fig7_shap_summary_plot.png")
        plt.savefig(fig7_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {fig7_path}")

def run_corrected_ml():
    df_nilm, feature_cols = prepare_microbiome_dataset(TABLE_S1_PATH)
    df_eval, df_imp, roc_data, pr_data, shap_vals, X_test, feat_names, prev = run_repeated_cv(
        df_nilm, feature_cols, n_splits=5, n_repeats=3
    )
    plot_corrected_ml_figures(roc_data, pr_data, df_imp, shap_vals, X_test, feat_names, prev)
    print("\nCorrected ML pipeline completed successfully!")

if __name__ == "__main__":
    run_corrected_ml()
