"""
mendeley_analysis.py
Performs complete exploratory data analysis, statistical comparisons, effect size calculations,
FDR corrections, and PCA on the Mendeley dataset (Table S1, n=15,607).
Outputs:
- biological_differences.csv
- differential_biomarkers.csv
- Figures: PCA, bacterial abundance distributions, correlation heatmaps
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Set cache and style
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

def load_and_preprocess(filepath):
    print("Loading raw data from:", filepath)
    df = pd.read_excel(filepath)
    df.columns = df.columns.str.strip()
    
    # 1. Feature Engineering
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
    
    # Check that columns exist
    for b in bacterial_species:
        assert b in df.columns, f"Missing {b}"
    for h in hpv_genotypes:
        assert h in df.columns, f"Missing {h}"
        
    # Total hrHPV Viral Load and metrics
    df["Total_hrHPV_Load"] = df[hpv_genotypes].sum(axis=1)
    df["Max_hrHPV_Load"] = df[hpv_genotypes].max(axis=1)
    df["HPV16_18_Load"] = df[["HPV 16", "HPV 18"]].sum(axis=1)
    df["Other_hrHPV_Load"] = df[[h for h in hpv_genotypes if h not in ["HPV 16", "HPV 18"]]].sum(axis=1)
    df["hrHPV_Genotype_Count"] = (df[hpv_genotypes] > 0).sum(axis=1)
    
    # Bacterial group metrics
    protective_lacto = ["L. crispatus", "L. gasseri", "L. jensenii"]
    all_lacto = ["L. iners", "L. crispatus", "L. gasseri", "L. jensenii"]
    bv_pathogens = ["F. vaginae", "G. vaginalis", "BVAB-2", "Megasphaera sp.Type 1", "Megasphaera sp. Type 2"]
    
    df["Total_Lactobacillus_Load"] = df[all_lacto].sum(axis=1)
    df["Protective_Lacto_Load"] = df[protective_lacto].sum(axis=1)
    df["Total_BV_Pathogens_Load"] = df[bv_pathogens].sum(axis=1)
    df["Total_Bacterial_Biomass"] = df[bacterial_species].sum(axis=1)
    
    # Log10-transformed abundance (+1 pseudocount)
    for b in bacterial_species:
        df[f"log10_{b}"] = np.log10(df[b] + 1)
    for h in hpv_genotypes:
        df[f"log10_{h}"] = np.log10(df[h] + 1)
        
    df["log10_Total_hrHPV_Load"] = np.log10(df["Total_hrHPV_Load"] + 1)
    df["log10_Total_Lacto"] = np.log10(df["Total_Lactobacillus_Load"] + 1)
    df["log10_Total_BV_Pathogens"] = np.log10(df["Total_BV_Pathogens_Load"] + 1)
    
    # Dysbiosis and protective balance metrics
    df["L_crispatus_to_L_iners_Ratio"] = df["log10_L. crispatus"] - df["log10_L. iners"]
    df["Lacto_to_Pathogen_Ratio"] = df["log10_Total_Lacto"] - df["log10_Total_BV_Pathogens"]
    
    # Relative abundances (avoid division by 0)
    biomass_safe = np.where(df["Total_Bacterial_Biomass"] > 0, df["Total_Bacterial_Biomass"], 1.0)
    for b in bacterial_species:
        df[f"rel_{b}"] = df[b] / biomass_safe
    df["rel_Total_Lacto"] = df["Total_Lactobacillus_Load"] / biomass_safe
    df["rel_Total_BV_Pathogens"] = df["Total_BV_Pathogens_Load"] / biomass_safe
    
    # Encode BV Status: BV-N=0, BV-T=0.5, BV-P=1
    bv_map = {"BV-N": 0.0, "BV-T": 0.5, "BV-P": 1.0}
    df["BV_Score"] = df["BV Status"].map(bv_map).fillna(0.0)
    
    # Age numeric
    df["Pt. Age"] = pd.to_numeric(df["Pt. Age"], errors="coerce").fillna(df["Pt. Age"].median())
    
    # Abnormal cytology flag: NILM == 0
    df["Abnormal_Cytology"] = (df["NILM"] == 0).astype(int)
    
    return df, bacterial_species, hpv_genotypes

def cliffs_delta(x, y):
    """Compute Cliff's Delta effect size between two groups."""
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return np.nan
    # Use vectorized comparison for speed with large arrays or binned
    # For speed on large datasets, use rank-based calculation:
    # d = (2 * U) / (n_x * n_y) - 1 where U is Mann-Whitney U for x > y
    u_stat, _ = stats.mannwhitneyu(x, y, alternative="greater")
    d = (2.0 * u_stat) / (n_x * n_y) - 1.0
    return d

def perform_statistical_analysis(df):
    """
    Primary comparison: HPV-negative Normal Cytology vs High-Risk HPV-positive Normal Cytology.
    Secondary comparison: HPV-positive Normal Cytology vs HPV-positive Abnormal Cytology.
    """
    print("\n" + "="*70)
    print("STEP 1: PRIMARY BIOLOGICAL ANALYSIS (Normal Cytology: NILM == 1)")
    print("="*70)
    
    df_normal = df[df["NILM"] == 1].copy()
    neg_grp = df_normal[df_normal["hrHPV Result"] == 0]
    pos_grp = df_normal[df_normal["hrHPV Result"] == 1]
    
    n_neg = len(neg_grp)
    n_pos = len(pos_grp)
    print(f"Cohort size: NILM==1 total = {len(df_normal)} | HPV(-) = {n_neg} | hrHPV(+) = {n_pos}")
    
    features_to_test = [
        # Bacterial raw loads
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
        # Relative abundances
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
        # Log10-transformed & balance indices
        ("log10_L. crispatus", "bacterial_log10"),
        ("log10_L. iners", "bacterial_log10"),
        ("log10_F. vaginae", "bacterial_log10"),
        ("log10_G. vaginalis", "bacterial_log10"),
        ("L_crispatus_to_L_iners_Ratio", "microbial_ratio"),
        ("Lacto_to_Pathogen_Ratio", "microbial_ratio"),
        # Clinical & Viral markers
        ("Pt. Age", "clinical_demographic"),
        ("BV_Score", "clinical_status"),
        ("Total_hrHPV_Load", "viral_load"),
        ("Max_hrHPV_Load", "viral_load"),
        ("HPV16_18_Load", "viral_load"),
        ("Other_hrHPV_Load", "viral_load"),
        ("hrHPV_Genotype_Count", "viral_diversity"),
        ("HPV 16", "viral_genotype"),
        ("HPV 18", "viral_genotype"),
        ("HPV 31", "viral_genotype"),
        ("HPV 35", "viral_genotype"),
        ("HPV 45", "viral_genotype"),
        ("HPV 51", "viral_genotype"),
        ("HPV 52", "viral_genotype"),
        ("HPV 56", "viral_genotype"),
        ("HPV 58", "viral_genotype"),
        ("HPV 59", "viral_genotype"),
    ]
    
    results = []
    
    for feat, cat in features_to_test:
        vals_neg = neg_grp[feat].dropna().values
        vals_pos = pos_grp[feat].dropna().values
        
        # Summary statistics
        mean_neg, std_neg = np.mean(vals_neg), np.std(vals_neg)
        mean_pos, std_pos = np.mean(vals_pos), np.std(vals_pos)
        med_neg = np.median(vals_neg)
        med_pos = np.median(vals_pos)
        iqr_neg = np.percentile(vals_neg, 75) - np.percentile(vals_neg, 25)
        iqr_pos = np.percentile(vals_pos, 75) - np.percentile(vals_pos, 25)
        
        # Positivity prevalence (> 0)
        pos_rate_neg = (vals_neg > 0).mean() * 100
        pos_rate_pos = (vals_pos > 0).mean() * 100
        prevalence_diff = pos_rate_pos - pos_rate_neg
        
        # Mann-Whitney U test
        u_stat, p_val = stats.mannwhitneyu(vals_pos, vals_neg, alternative="two-sided")
        
        # Cliff's Delta effect size
        d_cliff = cliffs_delta(vals_pos, vals_neg)
        
        # Cohen's d (pooled std)
        pooled_std = np.sqrt(((len(vals_neg)-1)*std_neg**2 + (len(vals_pos)-1)*std_pos**2) / (len(vals_neg)+len(vals_pos)-2))
        cohen_d = (mean_pos - mean_neg) / pooled_std if pooled_std > 1e-12 else 0.0
        
        # Log2 fold change (adding small pseudocount based on scale)
        pseudo = 1.0 if "load" in cat or feat.startswith("HPV") or feat.startswith("Total") else 0.001
        log2_fc = np.log2((mean_pos + pseudo) / (mean_neg + pseudo))
        
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
            "prevalence_difference_pct": prevalence_diff,
            "log2_fold_change": log2_fc,
            "cohens_d": cohen_d,
            "cliffs_delta": d_cliff,
            "mannwhitney_u": u_stat,
            "p_value": p_val
        })
        
    df_diff = pd.DataFrame(results)
    
    # False Discovery Rate adjustment (Benjamini-Hochberg)
    _, q_vals, _, _ = multipletests(df_diff["p_value"], alpha=0.05, method="fdr_bh")
    df_diff["adjusted_p_value_fdr"] = q_vals
    df_diff["significant_fdr_0.05"] = df_diff["adjusted_p_value_fdr"] < 0.05
    
    # Sort by significance and effect size
    df_diff = df_diff.sort_values(by=["adjusted_p_value_fdr", "mannwhitney_u"], ascending=[True, False])
    
    # Save biological_differences.csv
    bio_diff_path = os.path.join("/Users/ritweekraj/Desktop/HPV", "biological_differences.csv")
    df_diff.to_csv(bio_diff_path, index=False)
    print(f"Saved: {bio_diff_path} ({len(df_diff)} features)")
    
    # Significant subset: differential_biomarkers.csv
    df_sig = df_diff[df_diff["significant_fdr_0.05"] == True].copy()
    diff_bio_path = os.path.join("/Users/ritweekraj/Desktop/HPV", "differential_biomarkers.csv")
    df_sig.to_csv(diff_bio_path, index=False)
    print(f"Saved: {diff_bio_path} ({len(df_sig)} significant features at FDR < 0.05)")
    
    # Print top differential biomarkers
    print("\nTop 15 Most Significant Biological Differences (NILM Normal Cytology):")
    cols_to_show = ["feature", "category", "log2_fold_change", "cliffs_delta", "p_value", "adjusted_p_value_fdr"]
    print(df_diff[cols_to_show].head(15).to_string(index=False))
    
    # SECONDARY ANALYSIS: Normal vs Abnormal Cytology among hrHPV(+)
    print("\n" + "="*70)
    print("STEP 2: PROGRESSION BIOMARKER ANALYSIS (hrHPV(+) Normal vs Abnormal Cytology)")
    print("="*70)
    df_hrhpv = df[df["hrHPV Result"] == 1].copy()
    hpv_norm = df_hrhpv[df_hrhpv["NILM"] == 1]
    hpv_abn = df_hrhpv[df_hrhpv["NILM"] == 0]
    print(f"hrHPV(+) Cohort: Normal Cytology (NILM=1) = {len(hpv_norm)} | Abnormal Cytology (NILM=0) = {len(hpv_abn)}")
    
    prog_results = []
    for feat, cat in features_to_test:
        v_norm = hpv_norm[feat].dropna().values
        v_abn = hpv_abn[feat].dropna().values
        
        u_stat, p_val = stats.mannwhitneyu(v_abn, v_norm, alternative="two-sided")
        d_cliff = cliffs_delta(v_abn, v_norm)
        mean_norm, mean_abn = np.mean(v_norm), np.mean(v_abn)
        pseudo = 1.0 if "load" in cat or feat.startswith("HPV") or feat.startswith("Total") else 0.001
        log2_fc = np.log2((mean_abn + pseudo) / (mean_norm + pseudo))
        
        prog_results.append({
            "feature": feat,
            "category": cat,
            "comparison": "hrHPV_abnormal_vs_normal_cytology",
            "mean_hrHPV_normal": mean_norm,
            "mean_hrHPV_abnormal": mean_abn,
            "log2_fold_change": log2_fc,
            "cliffs_delta": d_cliff,
            "mannwhitney_u": u_stat,
            "p_value": p_val
        })
    df_prog = pd.DataFrame(prog_results)
    _, q_prog, _, _ = multipletests(df_prog["p_value"], alpha=0.05, method="fdr_bh")
    df_prog["adjusted_p_value_fdr"] = q_prog
    df_prog["significant_fdr_0.05"] = df_prog["adjusted_p_value_fdr"] < 0.05
    df_prog = df_prog.sort_values(by="adjusted_p_value_fdr")
    
    prog_path = os.path.join(OUTPUT_DIR, "cytological_progression_biomarkers.csv")
    df_prog.to_csv(prog_path, index=False)
    print(f"Saved: {prog_path} ({df_prog['significant_fdr_0.05'].sum()} significant progression biomarkers)")
    print("\nTop 10 Progression Biomarkers (hrHPV+ Normal vs Abnormal):")
    print(df_prog[["feature", "log2_fold_change", "cliffs_delta", "p_value", "adjusted_p_value_fdr"]].head(10).to_string(index=False))
    
    return df_diff, df_prog

def generate_exploratory_plots(df, df_diff):
    """Generate high-resolution scientific plots for PCA, bacterial abundance, and correlations."""
    print("\nGenerating exploratory figures...")
    
    df_normal = df[df["NILM"] == 1].copy()
    bacterial_species = [
        "F. vaginae", "G. vaginalis", "BVAB-2", 
        "Megasphaera sp.Type 1", "Megasphaera sp. Type 2", 
        "L. iners", "L. gasseri", "L. jensenii", "L. crispatus"
    ]
    
    # -------------------------------------------------------------
    # Figure 1: Bacterial Abundance by HPV Status (Normal Cytology)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    axes = axes.flatten()
    palette = {0: "#2b5c8f", 1: "#d95f02"}
    labels = {0: "HPV-Negative (n=11,438)", 1: "hrHPV-Positive (n=311)"}
    
    for idx, b in enumerate(bacterial_species):
        ax = axes[idx]
        plot_data = pd.DataFrame({
            "log10_load": df_normal[f"log10_{b}"],
            "HPV_Status": df_normal["hrHPV Result"].map(labels)
        })
        
        # Use boxplot with fliers off for clean visibility + jittered strip
        sns.boxplot(
            data=plot_data, x="HPV_Status", y="log10_load", 
            palette={"HPV-Negative (n=11,438)": "#4575b4", "hrHPV-Positive (n=311)": "#d73027"},
            ax=ax, width=0.45, showfliers=False, boxprops=dict(alpha=0.8)
        )
        # Overlay mean indicator
        means = plot_data.groupby("HPV_Status")["log10_load"].mean()
        ax.scatter([0, 1], [means["HPV-Negative (n=11,438)"], means["hrHPV-Positive (n=311)"]], 
                   color="gold", s=60, zorder=5, edgecolor="black", label="Mean")
        
        # Annotate p-value
        feat_row = df_diff[df_diff["feature"] == b]
        if not feat_row.empty:
            q_val = feat_row["adjusted_p_value_fdr"].values[0]
            fc = feat_row["log2_fold_change"].values[0]
            p_text = f"FDR q = {q_val:.2e}\nLog2FC = {fc:+.2f}"
            ax.set_title(f"{b}\n({p_text})", fontsize=11, fontweight="bold")
        else:
            ax.set_title(b, fontsize=11, fontweight="bold")
            
        ax.set_xlabel("")
        ax.set_ylabel("Log10(copies/mL + 1)", fontsize=9)
        ax.tick_params(axis="both", labelsize=9)
        
    plt.suptitle("Cervicovaginal Bacterial Abundance in Normal Cytology (NILM=1)\nStratified by High-Risk HPV Infection Status", 
                 fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    fig1_path = os.path.join(FIGURES_DIR, "fig1_bacterial_abundance_by_hpv.png")
    plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig1_path}")
    
    # -------------------------------------------------------------
    # Figure 2: Principal Component Analysis (PCA) of Cervicovaginal Microbiome
    # -------------------------------------------------------------
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
        "hrHPV": df_normal["hrHPV Result"].map({0: "HPV-Negative (NILM)", 1: "hrHPV-Positive (NILM)"})
    })
    
    plt.figure(figsize=(9, 7))
    sns.scatterplot(
        data=pca_df, x="PC1", y="PC2", hue="hrHPV",
        palette={"HPV-Negative (NILM)": "#3182bd", "hrHPV-Positive (NILM)": "#de2d26"},
        alpha=0.4, s=25, edgecolor=None
    )
    plt.title(f"Principal Component Analysis of Host Microbiome & Demographics\nPC1 ({var_exp[0]:.1f}%) vs PC2 ({var_exp[1]:.1f}% Variance Explained)", 
              fontsize=13, fontweight="bold")
    plt.xlabel(f"Principal Component 1 ({var_exp[0]:.1f}%)", fontsize=11)
    plt.ylabel(f"Principal Component 2 ({var_exp[1]:.1f}%)", fontsize=11)
    plt.legend(frameon=True, framealpha=0.9, loc="upper right")
    plt.tight_layout()
    fig2_path = os.path.join(FIGURES_DIR, "fig2_pca_microbiome_hpv.png")
    plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig2_path}")
    
    # -------------------------------------------------------------
    # Figure 3: Microbial Correlation & Interaction Heatmap
    # -------------------------------------------------------------
    corr_cols = [f"log10_{b}" for b in bacterial_species] + [
        "log10_Total_hrHPV_Load", "hrHPV_Genotype_Count", "Pt. Age", "BV_Score", "Abnormal_Cytology"
    ]
    renamed_cols = [b for b in bacterial_species] + [
        "Total hrHPV Load", "HPV Genotype Count", "Age", "BV Score", "Abnormal Cytology"
    ]
    corr_matrix = df[corr_cols].corr(method="spearman")
    corr_matrix.columns = renamed_cols
    corr_matrix.index = renamed_cols
    
    plt.figure(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, mask=mask, cmap="vlag", vmin=-0.8, vmax=0.8, center=0,
        annot=True, fmt=".2f", annot_kws={"size": 8},
        linewidths=0.5, cbar_kws={"label": "Spearman Correlation (\u03c1)"}
    )
    plt.title("Spearman Correlation Matrix of Cervicovaginal Microbiota,\nViral Load, and Host Clinical Parameters (n = 15,607)", 
              fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig3_path = os.path.join(FIGURES_DIR, "fig3_spearman_correlation_heatmap.png")
    plt.savefig(fig3_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig3_path}")
    
    # -------------------------------------------------------------
    # Figure 4: Differential Effect Size Volcano / Bar Plot
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    top_diff = df_diff.head(15).copy()
    top_diff["neg_log_p"] = -np.log10(top_diff["adjusted_p_value_fdr"] + 1e-300)
    top_diff = top_diff.sort_values(by="cliffs_delta", ascending=True)
    
    colors = ["#de2d26" if d > 0 else "#3182bd" for d in top_diff["cliffs_delta"]]
    bars = plt.barh(top_diff["feature"], top_diff["cliffs_delta"], color=colors, height=0.65)
    plt.axvline(0, color="black", linestyle="--", linewidth=0.8)
    plt.xlabel("Cliff's Delta Effect Size (Positive = Enriched in hrHPV+ Normal Cytology)", fontsize=11)
    plt.ylabel("Biological Feature", fontsize=11)
    plt.title("Ranked Effect Sizes of Top Differential Biomarkers\n(Normal Cytology: hrHPV+ vs hrHPV-)", fontsize=13, fontweight="bold")
    
    # Annotate FDR
    for bar, (_, row) in zip(bars, top_diff.iterrows()):
        x_pos = bar.get_width()
        offset = 0.01 if x_pos >= 0 else -0.01
        ha = "left" if x_pos >= 0 else "right"
        plt.text(x_pos + offset, bar.get_y() + bar.get_height()/2.0, 
                 f"q={row['adjusted_p_value_fdr']:.1e}", va="center", ha=ha, fontsize=8)
                 
    plt.tight_layout()
    fig4_path = os.path.join(FIGURES_DIR, "fig4_differential_biomarker_effect_sizes.png")
    plt.savefig(fig4_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig4_path}")

if __name__ == "__main__":
    df, bacterial_species, hpv_genotypes = load_and_preprocess(DATA_PATH)
    df_diff, df_prog = perform_statistical_analysis(df)
    generate_exploratory_plots(df, df_diff)
    print("\nEDA and statistical analysis completed successfully!")
