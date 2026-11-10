"""
gse_analysis.py
Analyzes host gene expression in cervical swabs from GEO dataset GSE75132:
- 41 cervical swab samples:
  * Category 0: Normal (HPV-negative, no disease, n=11)
  * Category 1: Persistent HPV16, no progression (n=10)
  * Category 2: Persistent HPV16, progressed to CIN3+ (n=20)
- Mapped target genes: CDKN2A (p16INK4A), MKI67 (Ki-67), TMEM45A, SERPINB5 (Maspin)
- Performs statistical comparisons, Log2FC, Mann-Whitney U, t-test, FDR correction
- Produces:
  * outputs/gse75132_differential_expression.csv
  * outputs/figures/fig8_host_gene_expression_gse75132.png
  * outputs/figures/fig9_host_gene_heatmap_gse75132.png
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import GEOparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

os.environ["MPLCONFIGDIR"] = "/tmp/mpl_cache"
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

OUTPUT_DIR = "/Users/ritweekraj/Desktop/HPV/outputs"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)
GSE_PATH = "/Users/ritweekraj/Desktop/HPV/data/geo/GSE75132_family.soft.gz"

def analyze_gse75132():
    print("Loading and analyzing GSE75132...")
    gse = GEOparse.get_GEO(filepath=GSE_PATH)
    
    # Probes to analyze
    probe_mapping = {
        "207039_at": ("CDKN2A", "CDKN2A (p16INK4a, probe 207039_at)"),
        "209644_x_at": ("CDKN2A", "CDKN2A (probe 209644_x_at)"),
        "211156_at": ("CDKN2A", "CDKN2A (probe 211156_at)"),
        "212020_s_at": ("MKI67", "MKI67 (Ki-67, probe 212020_s_at)"),
        "212021_s_at": ("MKI67", "MKI67 (probe 212021_s_at)"),
        "212022_s_at": ("MKI67", "MKI67 (probe 212022_s_at)"),
        "212023_s_at": ("MKI67", "MKI67 (probe 212023_s_at)"),
        "219410_at": ("TMEM45A", "TMEM45A (probe 219410_at)"),
        "1555551_at": ("SERPINB5", "SERPINB5 (Maspin, probe 1555551_at)"),
        "204855_at": ("SERPINB5", "SERPINB5 (Maspin, probe 204855_at)")
    }
    
    samples_data = []
    for gsm_name, gsm in gse.gsms.items():
        meta = gsm.metadata
        char = meta.get("characteristics_ch1", [])
        cat = None
        for c in char:
            if "category" in c:
                cat = int(c.split(":")[-1].strip().split()[0])
                break
        title = meta.get("title", [""])[0]
        tbl = gsm.table.set_index("ID_REF")
        
        row = {"GSM": gsm_name, "title": title, "category": cat}
        for probe_id in probe_mapping:
            if probe_id in tbl.index:
                row[probe_id] = float(tbl.loc[probe_id, "VALUE"])
        samples_data.append(row)
        
    df_samples = pd.DataFrame(samples_data)
    
    # Add mapped gene means (average across probes for the same gene)
    for gene in ["CDKN2A", "MKI67", "TMEM45A", "SERPINB5"]:
        gene_probes = [p for p, (g, _) in probe_mapping.items() if g == gene]
        df_samples[f"{gene}_mean"] = df_samples[gene_probes].mean(axis=1)
        
    # Group designations:
    # 0 = HPV-Negative Normal (n=11)
    # 1 = HPV16 Non-progressor (n=10)
    # 2 = HPV16 Progressor (n=20)
    # 1+2 = All HPV16-positive (n=30)
    cat0 = df_samples[df_samples["category"] == 0]
    cat1 = df_samples[df_samples["category"] == 1]
    cat2 = df_samples[df_samples["category"] == 2]
    cat_hpv16 = df_samples[df_samples["category"].isin([1, 2])]
    
    print(f"Sample breakdown: Normal (HPV-) = {len(cat0)} | HPV16 Non-Prog = {len(cat1)} | HPV16 Prog CIN3+ = {len(cat2)}")
    
    features_to_test = list(probe_mapping.keys()) + [f"{g}_mean" for g in ["CDKN2A", "MKI67", "TMEM45A", "SERPINB5"]]
    
    results = []
    for feat in features_to_test:
        gene_name = probe_mapping[feat][0] if feat in probe_mapping else feat.split("_")[0]
        desc = probe_mapping[feat][1] if feat in probe_mapping else f"{gene_name} (Composite Mean)"
        
        # Contrast 1: HPV-Negative Normal vs ALL HPV16-Positive
        v0 = cat0[feat].values
        v_hpv = cat_hpv16[feat].values
        
        # Student t-test & Mann-Whitney
        t_stat1, p_t1 = stats.ttest_ind(v_hpv, v0)
        u_stat1, p_u1 = stats.mannwhitneyu(v_hpv, v0, alternative="two-sided")
        log2_fc1 = np.mean(v_hpv) - np.mean(v0) # GEO values are already normalized log2 intensity
        
        # Contrast 2: HPV16 Non-progressor (cat 1) vs HPV16 Progressor to CIN3+ (cat 2)
        v1 = cat1[feat].values
        v2 = cat2[feat].values
        t_stat2, p_t2 = stats.ttest_ind(v2, v1)
        u_stat2, p_u2 = stats.mannwhitneyu(v2, v1, alternative="two-sided")
        log2_fc2 = np.mean(v2) - np.mean(v1)
        
        results.append({
            "feature": feat,
            "gene": gene_name,
            "description": desc,
            # Normal vs HPV16+
            "mean_HPV_neg": np.mean(v0),
            "mean_HPV16_pos": np.mean(v_hpv),
            "log2_FC_HPV16_vs_Normal": log2_fc1,
            "p_val_HPV16_vs_Normal": p_u1,
            "p_t_test_HPV16_vs_Normal": p_t1,
            # Non-progressor vs Progressor
            "mean_HPV16_non_prog": np.mean(v1),
            "mean_HPV16_prog_CIN3": np.mean(v2),
            "log2_FC_Prog_vs_NonProg": log2_fc2,
            "p_val_Prog_vs_NonProg": p_u2,
            "p_t_test_Prog_vs_NonProg": p_t2
        })
        
    df_res = pd.DataFrame(results)
    _, q1, _, _ = multipletests(df_res["p_val_HPV16_vs_Normal"], method="fdr_bh")
    _, q2, _, _ = multipletests(df_res["p_val_Prog_vs_NonProg"], method="fdr_bh")
    df_res["FDR_HPV16_vs_Normal"] = q1
    df_res["FDR_Prog_vs_NonProg"] = q2
    
    out_csv = os.path.join(OUTPUT_DIR, "gse75132_differential_expression.csv")
    df_res.to_csv(out_csv, index=False)
    print(f"Saved differential expression results: {out_csv}")
    print("\nHost Gene Expression Differences (GSE75132):")
    cols_display = ["gene", "feature", "mean_HPV_neg", "mean_HPV16_pos", "log2_FC_HPV16_vs_Normal", "p_val_HPV16_vs_Normal", "FDR_HPV16_vs_Normal"]
    print(df_res[cols_display].to_string(index=False))
    
    # -------------------------------------------------------------
    # Figure 8: Host Gene Expression Boxplots (CDKN2A, MKI67, TMEM45A, SERPINB5)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.flatten()
    genes = ["CDKN2A", "MKI67", "TMEM45A", "SERPINB5"]
    group_labels = {0: "HPV(-) Normal\n(n=11)", 1: "HPV16+ Non-Prog\n(n=10)", 2: "HPV16+ CIN3+ Prog\n(n=20)"}
    df_samples["Group"] = df_samples["category"].map(group_labels)
    group_order = [group_labels[0], group_labels[1], group_labels[2]]
    palette = {group_labels[0]: "#4575b4", group_labels[1]: "#fdae61", group_labels[2]: "#d73027"}
    
    for idx, g in enumerate(genes):
        ax = axes[idx]
        col_name = f"{g}_mean"
        sns.boxplot(data=df_samples, x="Group", y=col_name, order=group_order, palette=palette, ax=ax, width=0.45, boxprops=dict(alpha=0.85))
        sns.stripplot(data=df_samples, x="Group", y=col_name, order=group_order, color="black", size=6, jitter=0.2, alpha=0.6, ax=ax)
        
        # P-values from df_res
        row = df_res[df_res["feature"] == col_name]
        if not row.empty:
            fc1 = row["log2_FC_HPV16_vs_Normal"].values[0]
            p1 = row["p_val_HPV16_vs_Normal"].values[0]
            p2 = row["p_val_Prog_vs_NonProg"].values[0]
            ax.set_title(f"{g} Expression\n(HPV16 vs Normal: \u0394={fc1:+.2f}, p={p1:.3e} | Prog vs NonProg: p={p2:.3f})", 
                         fontsize=11, fontweight="bold")
        else:
            ax.set_title(f"{g} Expression", fontsize=11, fontweight="bold")
            
        ax.set_xlabel("")
        ax.set_ylabel("Normalized Log2 Expression", fontsize=10)
        ax.tick_params(axis="both", labelsize=9)
        
    plt.suptitle("Host Epithelial & Oncogenic Biomarker Expression in Cervical Swabs (GEO GSE75132)\nComparing HPV-Negative Normal vs HPV16-Infected vs CIN3+ Disease Progression", 
                 fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()
    fig8_path = os.path.join(FIGURES_DIR, "fig8_host_gene_expression_gse75132.png")
    plt.savefig(fig8_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig8_path}")
    
    # -------------------------------------------------------------
    # Figure 9: Heatmap of Cervical Swab Gene Expression Profile
    # -------------------------------------------------------------
    plt.figure(figsize=(12, 6))
    heatmap_data = df_samples.sort_values(by="category")[["GSM", "category"] + [f"{g}_mean" for g in genes]].copy()
    heatmap_matrix = heatmap_data[[f"{g}_mean" for g in genes]].T
    heatmap_matrix.columns = [f"{row['GSM']} ({row['category']})" for _, row in heatmap_data.iterrows()]
    heatmap_matrix.index = ["CDKN2A (p16INK4a)", "MKI67 (Ki-67)", "TMEM45A", "SERPINB5 (Maspin)"]
    
    # Standardize rows (Z-score)
    heatmap_z = heatmap_matrix.apply(lambda x: (x - x.mean()) / x.std(), axis=1)
    
    sns.heatmap(heatmap_z, cmap="coolwarm", center=0, cbar_kws={"label": "Expression Z-Score"}, linewidths=0.5)
    plt.title("Z-Score Standardized Host Biomarker Expression Across 41 Cervical Swab Samples\n(0 = HPV- Normal [first 11 cols] | 1 = HPV16 Non-Prog | 2 = HPV16 CIN3+ Progressors)", 
              fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig9_path = os.path.join(FIGURES_DIR, "fig9_host_gene_heatmap_gse75132.png")
    plt.savefig(fig9_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig9_path}")
    
    return df_res

if __name__ == "__main__":
    analyze_gse75132()
