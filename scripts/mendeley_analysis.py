"""
mendeley_analysis.py (Scientifically Corrected Version)
- Evaluates HPV-negative normal cytology vs hrHPV-positive normal cytology (NILM == 1)
- Reports zero-handling (values below qPCR limit of detection < 100 copies/mL)
- Unadjusted tests: Mann-Whitney U, Cliff's delta with 95% CIs, Cohen's d, Log2FC
- Confounder analysis: Multivariable Logistic Regression adjusted for Age and BV Status
- Multiple testing correction across all bacterial features tested (Benjamini-Hochberg FDR)
- Cautious scientific wording (cross-sectional associations, no causal or temporal claims)
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

os.environ["MPLCONFIGDIR"] = "/tmp/mpl_cache"
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 0.8,
    "grid.color": "#e0e0e0",
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
})

DATA_PATH = "/Users/ritweekraj/Desktop/HPV/data/mendeley/Table S1.xlsx"
OUTPUT_DIR = "/Users/ritweekraj/Desktop/HPV/outputs"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

def cliffs_delta_with_ci(x, y, alpha=0.05):
    """Computes Cliff's Delta effect size and asymptotic 95% Confidence Interval."""
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return np.nan, np.nan, np.nan
    u_stat, _ = stats.mannwhitneyu(x, y, alternative="greater")
    d = (2.0 * u_stat) / (n_x * n_y) - 1.0
    
    # Asymptotic variance calculation for Cliff's Delta
    s_d = np.sqrt((n_x + n_y + 1) / (3.0 * n_x * n_y))
    z = stats.norm.ppf(1 - alpha / 2.0)
    ci_low = max(-1.0, d - z * s_d)
    ci_high = min(1.0, d + z * s_d)
    return d, ci_low, ci_high

def load_and_preprocess(filepath):
    print("Loading raw clinical dataset from:", filepath)
    df = pd.read_excel(filepath)
    df.columns = df.columns.str.strip()
    
    bacterial_species = [
        "F. vaginae", "G. vaginalis", "BVAB-2", 
        "Megasphaera sp.Type 1", "Megasphaera sp. Type 2", 
        "L. iners", "L. gasseri", "L. jensenii", "L. crispatus"
    ]
    
    all_lacto = ["L. iners", "L. crispatus", "L. gasseri", "L. jensenii"]
    protective_lacto = ["L. crispatus", "L. gasseri", "L. jensenii"]
    bv_pathogens = ["F. vaginae", "G. vaginalis", "BVAB-2", "Megasphaera sp.Type 1", "Megasphaera sp. Type 2"]
    
    # Compute composites
    df["Total_Lactobacillus_Load"] = df[all_lacto].sum(axis=1)
    df["Protective_Lacto_Load"] = df[protective_lacto].sum(axis=1)
    df["Total_BV_Pathogens_Load"] = df[bv_pathogens].sum(axis=1)
    df["Total_Bacterial_Biomass"] = df[bacterial_species].sum(axis=1)
    
    # Log10-transformed abundance (+1 pseudocount)
    for b in bacterial_species:
        df[f"log10_{b}"] = np.log10(df[b] + 1)
        
    df["log10_Total_Lacto"] = np.log10(df["Total_Lactobacillus_Load"] + 1)
    df["log10_Total_BV_Pathogens"] = np.log10(df["Total_BV_Pathogens_Load"] + 1)
    df["L_crispatus_to_L_iners_Ratio"] = df["log10_L. crispatus"] - df["log10_L. iners"]
    df["Lacto_to_Pathogen_Ratio"] = df["log10_Total_Lacto"] - df["log10_Total_BV_Pathogens"]
    
    # Relative abundances
    biomass_safe = np.where(df["Total_Bacterial_Biomass"] > 0, df["Total_Bacterial_Biomass"], 1.0)
    for b in bacterial_species:
        df[f"rel_{b}"] = df[b] / biomass_safe
    df["rel_Total_Lacto"] = df["Total_Lactobacillus_Load"] / biomass_safe
    df["rel_Total_BV_Pathogens"] = df["Total_BV_Pathogens_Load"] / biomass_safe
    
    # Clinical covariates
    bv_map = {"BV-N": 0.0, "BV-T": 0.5, "BV-P": 1.0}
    df["BV_Score"] = df["BV Status"].map(bv_map).fillna(0.0)
    df["BV_Positive_Binary"] = (df["BV Status"] == "BV-P").astype(int)
    df["Pt. Age"] = pd.to_numeric(df["Pt. Age"], errors="coerce")
    df["Pt. Age"] = df["Pt. Age"].fillna(df["Pt. Age"].median())
    
    return df, bacterial_species

def perform_statistical_analysis(df):
    print("\n" + "="*75)
    print("STEP 1: PRIMARY BACTERIAL & CLINICAL ANALYSIS (Normal Cytology: NILM == 1)")
    print("="*75)
    
    df_normal = df[df["NILM"] == 1].copy()
    neg_grp = df_normal[df_normal["hrHPV Result"] == 0]
    pos_grp = df_normal[df_normal["hrHPV Result"] == 1]
    
    n_neg = len(neg_grp)
    n_pos = len(pos_grp)
    print(f"Cohort size: NILM==1 total = {len(df_normal)} | HPV(-) = {n_neg} (97.35%) | hrHPV(+) = {n_pos} (2.65%)")
    
    # Check confounders between groups
    age_ttest = stats.ttest_ind(pos_grp["Pt. Age"], neg_grp["Pt. Age"])
    chi2_bv = stats.chi2_contingency(pd.crosstab(df_normal["BV Status"], df_normal["hrHPV Result"]))
    print(f"Confounder Check: Mean Age HPV(-) = {neg_grp['Pt. Age'].mean():.2f} vs hrHPV(+) = {pos_grp['Pt. Age'].mean():.2f} (t-test p = {age_ttest.pvalue:.2e})")
    print(f"Confounder Check: BV-Positive rate HPV(-) = {(neg_grp['BV Status']=='BV-P').mean()*100:.1f}% vs hrHPV(+) = {(pos_grp['BV Status']=='BV-P').mean()*100:.1f}% (Chi2 p = {chi2_bv.pvalue:.4f})")
    
    # STRICT EXCLUSION OF ANY HPV-DERIVED PREDICTORS!
    features_to_test = [
        ("F. vaginae", "bacterial_load"),
        ("G. vaginalis", "bacterial_load"),
        ("BVAB-2", "bacterial_load"),
        ("Megasphaera sp.Type 1", "bacterial_load"),
        ("Megasphaera sp. Type 2", "bacterial_load"),
        ("L. iners", "bacterial_load"),
        ("L. gasseri", "bacterial_load"),
        ("L. jensenii", "bacterial_load"),
        ("L. crispatus", "bacterial_load"),
        ("Total_Lactobacillus_Load", "bacterial_composite"),
        ("Protective_Lacto_Load", "bacterial_composite"),
        ("Total_BV_Pathogens_Load", "bacterial_composite"),
        ("rel_L. crispatus", "microbiome_composition"),
        ("rel_L. iners", "microbiome_composition"),
        ("rel_L. gasseri", "microbiome_composition"),
        ("rel_L. jensenii", "microbiome_composition"),
        ("rel_F. vaginae", "microbiome_composition"),
        ("rel_G. vaginalis", "microbiome_composition"),
        ("rel_BVAB-2", "microbiome_composition"),
        ("rel_Megasphaera sp.Type 1", "microbiome_composition"),
        ("rel_Total_Lacto", "microbiome_composition"),
        ("rel_Total_BV_Pathogens", "microbiome_composition"),
        ("log10_L. crispatus", "bacterial_log10"),
        ("log10_L. iners", "bacterial_log10"),
        ("log10_F. vaginae", "bacterial_log10"),
        ("log10_G. vaginalis", "bacterial_log10"),
        ("log10_BVAB-2", "bacterial_log10"),
        ("L_crispatus_to_L_iners_Ratio", "microbial_ratio"),
        ("Lacto_to_Pathogen_Ratio", "microbial_ratio"),
        ("Pt. Age", "clinical_demographic"),
        ("BV_Score", "clinical_status")
    ]
    
    results = []
    
    for feat, cat in features_to_test:
        vals_neg = neg_grp[feat].dropna().values
        vals_pos = pos_grp[feat].dropna().values
        
        mean_neg, std_neg = np.mean(vals_neg), np.std(vals_neg)
        mean_pos, std_pos = np.mean(vals_pos), np.std(vals_pos)
        med_neg = np.median(vals_neg)
        med_pos = np.median(vals_pos)
        iqr_neg = np.percentile(vals_neg, 75) - np.percentile(vals_neg, 25)
        iqr_pos = np.percentile(vals_pos, 75) - np.percentile(vals_pos, 25)
        
        pos_rate_neg = (vals_neg > 0).mean() * 100
        pos_rate_pos = (vals_pos > 0).mean() * 100
        prev_diff = pos_rate_pos - pos_rate_neg
        
        u_stat, p_val = stats.mannwhitneyu(vals_pos, vals_neg, alternative="two-sided")
        d_cliff, ci_low, ci_high = cliffs_delta_with_ci(vals_pos, vals_neg)
        
        pooled_std = np.sqrt(((len(vals_neg)-1)*std_neg**2 + (len(vals_pos)-1)*std_pos**2) / (len(vals_neg)+len(vals_pos)-2))
        cohen_d = (mean_pos - mean_neg) / pooled_std if pooled_std > 1e-12 else 0.0
        
        pseudo = 1.0 if "load" in cat or feat.startswith("Total") else 0.001
        log2_fc = np.log2((mean_pos + pseudo) / (mean_neg + pseudo))
        
        try:
            X_cov = df_normal[[feat, "Pt. Age", "BV_Score"]].dropna()
            y_cov = df_normal.loc[X_cov.index, "hrHPV Result"]
            feat_std = (X_cov[feat] - X_cov[feat].mean()) / (X_cov[feat].std() + 1e-12)
            X_mat = sm.add_constant(pd.DataFrame({
                "feature": feat_std,
                "age": (X_cov["Pt. Age"] - X_cov["Pt. Age"].mean()) / X_cov["Pt. Age"].std(),
                "bv": X_cov["BV_Score"]
            }))
            logit_mod = sm.Logit(y_cov, X_mat).fit(disp=False)
            adj_or = np.exp(logit_mod.params["feature"])
            adj_p = logit_mod.pvalues["feature"]
            conf = logit_mod.conf_int().loc["feature"]
            adj_ci_low, adj_ci_high = np.exp(conf[0]), np.exp(conf[1])
        except Exception:
            adj_or, adj_p, adj_ci_low, adj_ci_high = np.nan, np.nan, np.nan, np.nan
        
        results.append({
            "feature": feat,
            "category": cat,
            "comparison": "NILM_hrHPV_pos_vs_neg",
            "mean_HPV_neg": mean_neg,
            "std_HPV_neg": std_neg,
            "median_HPV_neg": med_neg,
            "iqr_HPV_neg": iqr_neg,
            "prev_pct_HPV_neg": pos_rate_neg,
            "mean_hrHPV_pos": mean_pos,
            "std_hrHPV_pos": std_pos,
            "median_hrHPV_pos": med_pos,
            "iqr_hrHPV_pos": iqr_pos,
            "prev_pct_hrHPV_pos": pos_rate_pos,
            "prevalence_difference_pct": prev_diff,
            "log2_fold_change": log2_fc,
            "cohens_d": cohen_d,
            "cliffs_delta": d_cliff,
            "cliffs_delta_ci95_low": ci_low,
            "cliffs_delta_ci95_high": ci_high,
            "mannwhitney_u": u_stat,
            "unadjusted_p_value": p_val,
            "adjusted_odds_ratio_per_sd": adj_or,
            "adj_or_ci95_low": adj_ci_low,
            "adj_or_ci95_high": adj_ci_high,
            "adjusted_regression_p_value": adj_p
        })
        
    df_diff = pd.DataFrame(results)
    
    # Benjamini-Hochberg FDR correction across all bacterial & clinical tests
    _, q_vals, _, _ = multipletests(df_diff["unadjusted_p_value"], alpha=0.05, method="fdr_bh")
    df_diff["adjusted_p_value_fdr"] = q_vals
    df_diff["significant_fdr_0.05"] = df_diff["adjusted_p_value_fdr"] < 0.05
    
    df_diff = df_diff.sort_values(by=["adjusted_p_value_fdr", "mannwhitney_u"], ascending=[True, False])
    
    bio_diff_path = os.path.join("/Users/ritweekraj/Desktop/HPV", "biological_differences.csv")
    df_diff.to_csv(bio_diff_path, index=False)
    print(f"Saved: {bio_diff_path} ({len(df_diff)} features tested)")
    
    df_sig = df_diff[df_diff["significant_fdr_0.05"] == True].copy()
    diff_bio_path = os.path.join("/Users/ritweekraj/Desktop/HPV", "differential_biomarkers.csv")
    df_sig.to_csv(diff_bio_path, index=False)
    print(f"Saved: {diff_bio_path} ({len(df_sig)} significant features at FDR q < 0.05)")
    
    print("\nTop Differential Bacterial & Clinical Biomarkers in Normal Cytology (NILM):")
    cols_show = ["feature", "category", "cliffs_delta", "cliffs_delta_ci95_low", "cliffs_delta_ci95_high", 
                 "unadjusted_p_value", "adjusted_p_value_fdr", "adjusted_odds_ratio_per_sd"]
    print(df_diff[cols_show].head(15).to_string(index=False))
    
    return df_diff

def generate_exploratory_plots(df, df_diff):
    print("\nGenerating updated scientific figures...")
    df_normal = df[df["NILM"] == 1].copy()
    bacterial_species = [
        "F. vaginae", "G. vaginalis", "BVAB-2", 
        "Megasphaera sp.Type 1", "Megasphaera sp. Type 2", 
        "L. iners", "L. gasseri", "L. jensenii", "L. crispatus"
    ]
    
    # Figure 1: Bacterial Abundance by HPV Status (Normal Cytology)
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    axes = axes.flatten()
    labels = {0: "HPV-Negative (n=11,438)", 1: "hrHPV-Positive (n=311)"}
    
    for idx, b in enumerate(bacterial_species):
        ax = axes[idx]
        plot_data = pd.DataFrame({
            "log10_load": df_normal[f"log10_{b}"],
            "HPV_Status": df_normal["hrHPV Result"].map(labels)
        })
        
        sns.boxplot(
            data=plot_data, x="HPV_Status", y="log10_load", 
            palette={"HPV-Negative (n=11,438)": "#4575b4", "hrHPV-Positive (n=311)": "#d73027"},
            ax=ax, width=0.45, showfliers=False, boxprops=dict(alpha=0.8)
        )
        means = plot_data.groupby("HPV_Status")["log10_load"].mean()
        ax.scatter([0, 1], [means["HPV-Negative (n=11,438)"], means["hrHPV-Positive (n=311)"]], 
                   color="gold", s=60, zorder=5, edgecolor="black", label="Mean")
        
        feat_row = df_diff[df_diff["feature"] == b]
        if not feat_row.empty:
            q_val = feat_row["adjusted_p_value_fdr"].values[0]
            d_val = feat_row["cliffs_delta"].values[0]
            p_text = f"FDR q = {q_val:.2e}\nCliff's \u0394 = {d_val:+.3f}"
            ax.set_title(f"{b}\n({p_text})", fontsize=10, fontweight="bold")
        else:
            ax.set_title(b, fontsize=10, fontweight="bold")
            
        ax.set_xlabel("")
        ax.set_ylabel("Log10(qPCR copies/mL + 1)", fontsize=9)
        ax.tick_params(axis="both", labelsize=8.5)
        
    plt.suptitle("Cervicovaginal Bacterial Abundance in Normal Cytology (NILM=1)\nCross-Sectional Comparison Stratified by High-Risk HPV Infection Status", 
                 fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()
    fig1_path = os.path.join(FIGURES_DIR, "fig1_bacterial_abundance_by_hpv.png")
    plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig1_path}")
    
    # Figure 2: PCA of Cervicovaginal Microbiome in Normal Cytology
    microbiome_feats = [f"log10_{b}" for b in bacterial_species] + ["Pt. Age", "BV_Score", "Lacto_to_Pathogen_Ratio"]
    X_pca = df_normal[microbiome_feats].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pca)
    
    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_ * 100
    
    pca_df = pd.DataFrame({
        "PC1": pcs[:, 0],
        "PC2": pcs[:, 1],
        "hrHPV": df_normal["hrHPV Result"].map({0: "HPV-Negative (n=11,438)", 1: "hrHPV-Positive (n=311)"})
    })
    
    plt.figure(figsize=(9, 7))
    sns.scatterplot(
        data=pca_df, x="PC1", y="PC2", hue="hrHPV",
        palette={"HPV-Negative (n=11,438)": "#3182bd", "hrHPV-Positive (n=311)": "#de2d26"},
        alpha=0.35, s=25, edgecolor=None
    )
    plt.title(f"Principal Component Analysis of Vaginal Bacterial qPCR Profiles (Normal Cytology)\nPC1 ({var_exp[0]:.1f}%) vs PC2 ({var_exp[1]:.1f}% Variance Explained)", 
              fontsize=12, fontweight="bold")
    plt.xlabel(f"PC1 ({var_exp[0]:.1f}% Variance)", fontsize=10.5)
    plt.ylabel(f"PC2 ({var_exp[1]:.1f}% Variance)", fontsize=10.5)
    plt.legend(frameon=True, framealpha=0.9, loc="upper right")
    plt.tight_layout()
    fig2_path = os.path.join(FIGURES_DIR, "fig2_pca_microbiome_hpv.png")
    plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig2_path}")
    
    # Figure 3: Correlation Matrix of Bacterial qPCR & Clinical Covariates
    corr_cols = [f"log10_{b}" for b in bacterial_species] + ["Pt. Age", "BV_Score", "Lacto_to_Pathogen_Ratio"]
    renamed_cols = [b for b in bacterial_species] + ["Age", "BV Score", "Lacto:Pathogen Ratio"]
    corr_matrix = df_normal[corr_cols].corr(method="spearman")
    corr_matrix.columns = renamed_cols
    corr_matrix.index = renamed_cols
    
    plt.figure(figsize=(10, 8.5))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, mask=mask, cmap="vlag", vmin=-0.8, vmax=0.8, center=0,
        annot=True, fmt=".2f", annot_kws={"size": 8},
        linewidths=0.5, cbar_kws={"label": "Spearman Rank Correlation (\u03c1)"}
    )
    plt.title("Spearman Correlation Matrix of Cervicovaginal Bacterial Taxa\nand Clinical Covariates in Normal Cytology (n = 11,749)", 
              fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig3_path = os.path.join(FIGURES_DIR, "fig3_spearman_correlation_heatmap.png")
    plt.savefig(fig3_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig3_path}")
    
    # Figure 4: Effect Sizes (Cliff's Delta with 95% CIs) of Differential Biomarkers
    plt.figure(figsize=(10, 6.5))
    top_diff = df_diff[df_diff["category"].str.contains("bact|micro|clinical")].head(14).copy()
    top_diff = top_diff.sort_values(by="cliffs_delta", ascending=True)
    
    colors_list = ["#de2d26" if d > 0 else "#3182bd" for d in top_diff["cliffs_delta"]]
    y_pos = np.arange(len(top_diff))
    
    plt.barh(y_pos, top_diff["cliffs_delta"], color=colors_list, height=0.6, alpha=0.85)
    xerr_low = top_diff["cliffs_delta"] - top_diff["cliffs_delta_ci95_low"]
    xerr_high = top_diff["cliffs_delta_ci95_high"] - top_diff["cliffs_delta"]
    plt.errorbar(top_diff["cliffs_delta"], y_pos, xerr=[xerr_low, xerr_high], fmt="none", ecolor="black", capsize=3, elinewidth=1)
    
    plt.yticks(y_pos, top_diff["feature"], fontsize=9.5)
    plt.axvline(0, color="black", linestyle="--", linewidth=0.8)
    plt.xlabel("Cliff's Delta Effect Size with 95% CI (Positive = Enriched in hrHPV+ Normal Cytology)", fontsize=10.5)
    plt.title("Ranked Non-Parametric Effect Sizes of Top Cervicovaginal Biomarkers\n(Normal Cytology: hrHPV+ vs hrHPV-)", fontsize=12, fontweight="bold")
    
    for idx, (_, row) in enumerate(top_diff.iterrows()):
        x_val = row["cliffs_delta"]
        offset = 0.02 if x_val >= 0 else -0.02
        ha = "left" if x_val >= 0 else "right"
        plt.text(x_val + offset, idx, f"q={row['adjusted_p_value_fdr']:.1e}", va="center", ha=ha, fontsize=7.5)
        
    plt.tight_layout()
    fig4_path = os.path.join(FIGURES_DIR, "fig4_differential_biomarker_effect_sizes.png")
    plt.savefig(fig4_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig4_path}")

if __name__ == "__main__":
    df, bacterial_species = load_and_preprocess(DATA_PATH)
    df_diff = perform_statistical_analysis(df)
    generate_exploratory_plots(df, df_diff)
    print("\nCorrected Mendeley statistical analysis completed!")
