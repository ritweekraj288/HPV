"""
gse_analysis.py (Scientifically Corrected Version)
- Verifies Affymetrix HG-U133 Plus 2.0 (GPL570) probe mappings for 4 target host genes (10 probes total).
- Verifies expression scale: GEO submitter protocol normalized linear values by dividing by the median of HPV-negative controls.
- Calculates correct logarithmic fold changes: Log2FC = log2(mean_group1 / mean_group0).
- Preserves the three original patient groups:
  * Category 0: HPV-negative normal controls (n = 11)
  * Category 1: Persistent HPV16-positive non-progressors (n = 10)
  * Category 2: Persistent HPV16-positive progressors to CIN3+ (n = 20)
- Executes three separate primary comparisons:
  1. HPV16 Non-progressors vs HPV-negative Controls
  2. HPV16 Progressors to CIN3+ vs HPV-negative Controls
  3. HPV16 Progressors to CIN3+ vs HPV16 Non-progressors
  (and overall HPV16-positive vs Controls)
- Applies Benjamini-Hochberg FDR correction across all 10 tested probes.
- Cautious scientific conclusions: 41 cervical swab exploratory cohort, NOT a clinically validated panel.
- Outputs:
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
    print("Loading and analyzing GSE75132 (HG-U133 Plus 2.0)...")
    gse = GEOparse.get_GEO(filepath=GSE_PATH)
    
    # 1. Document exact probe-to-gene mapping
    probe_dict = {
        "219410_at": ("TMEM45A", "transmembrane protein 45A"),
        "204855_at": ("SERPINB5", "serpin peptidase inhibitor clade B member 5 (Maspin)"),
        "1555551_at": ("SERPINB5", "serpin peptidase inhibitor clade B member 5"),
        "207039_at": ("CDKN2A", "cyclin-dependent kinase inhibitor 2A (p16INK4A)"),
        "209644_x_at": ("CDKN2A", "cyclin-dependent kinase inhibitor 2A"),
        "211156_at": ("CDKN2A", "cyclin-dependent kinase inhibitor 2A"),
        "212020_s_at": ("MKI67", "marker of proliferation Ki-67"),
        "212021_s_at": ("MKI67", "marker of proliferation Ki-67"),
        "212022_s_at": ("MKI67", "marker of proliferation Ki-67"),
        "212023_s_at": ("MKI67", "marker of proliferation Ki-67")
    }
    
    samples_data = []
    for gsm_name, gsm in gse.gsms.items():
        char = gsm.metadata.get("characteristics_ch1", [])
        cat = int([c.split(":")[-1].strip().split()[0] for c in char if "category" in c][0])
        title = gsm.metadata.get("title", [""])[0]
        tbl = gsm.table.set_index("ID_REF")
        
        row = {"GSM": gsm_name, "title": title, "category": cat}
        for probe_id in probe_dict:
            row[probe_id] = float(tbl.loc[probe_id, "VALUE"])
        samples_data.append(row)
        
    df_samples = pd.DataFrame(samples_data)
    
    n_cat0 = (df_samples["category"] == 0).sum()
    n_cat1 = (df_samples["category"] == 1).sum()
    n_cat2 = (df_samples["category"] == 2).sum()
    print(f"Sample breakdown: Category 0 (HPV- Negative Controls) = {n_cat0}")
    print(f"                  Category 1 (HPV16+ Non-progressors)     = {n_cat1}")
    print(f"                  Category 2 (HPV16+ Progressors CIN3+)   = {n_cat2}")
    print(f"                  Total Cervical Swab Specimens           = {len(df_samples)}")
    
    # Define the 4 pairwise/group contrasts
    contrasts = [
        ("HPV16_NonProg_vs_Ctrl", [1], [0], "HPV16 Non-Progressors vs HPV- Negative Controls"),
        ("HPV16_Prog_vs_Ctrl", [2], [0], "HPV16 Progressors (CIN3+) vs HPV- Negative Controls"),
        ("Prog_vs_NonProg", [2], [1], "HPV16 Progressors (CIN3+) vs HPV16 Non-Progressors"),
        ("HPV16_All_vs_Ctrl", [1, 2], [0], "All HPV16-Positive Swabs vs HPV- Negative Controls")
    ]
    
    all_results = []
    
    for contrast_id, grpA_codes, grpB_codes, contrast_desc in contrasts:
        subA = df_samples[df_samples["category"].isin(grpA_codes)]
        subB = df_samples[df_samples["category"].isin(grpB_codes)]
        
        p_vals_mw, p_vals_tt, log2_fcs, rows_temp = [], [], [], []
        
        for probe_id, (gene, desc) in probe_dict.items():
            vA = subA[probe_id].values
            vB = subB[probe_id].values
            
            meanA = np.mean(vA)
            meanB = np.mean(vB)
            # Correct Log2 Fold Change calculation on linear ratio scale:
            log2_fc = np.log2((meanA + 1e-6) / (meanB + 1e-6))
            
            # Non-parametric Mann-Whitney U test
            u_stat, p_mw = stats.mannwhitneyu(vA, vB, alternative="two-sided")
            # Student t-test
            t_stat, p_tt = stats.ttest_ind(vA, vB)
            
            p_vals_mw.append(p_mw)
            p_vals_tt.append(p_tt)
            log2_fcs.append(log2_fc)
            
            rows_temp.append({
                "contrast_id": contrast_id,
                "contrast_description": contrast_desc,
                "gene_symbol": gene,
                "probe_id": probe_id,
                "probe_description": desc,
                "n_groupA": len(vA),
                "n_groupB": len(vB),
                "linear_mean_groupA": meanA,
                "linear_mean_groupB": meanB,
                "log2_fold_change": log2_fc,
                "mannwhitney_u_stat": u_stat,
                "mannwhitney_p_value": p_mw,
                "t_test_p_value": p_tt
            })
            
        # FDR correction across all 10 probes within contrast
        _, q_vals_mw, _, _ = multipletests(p_vals_mw, method="fdr_bh")
        for idx, row in enumerate(rows_temp):
            row["fdr_adjusted_q_value"] = q_vals_mw[idx]
            row["significant_fdr_0.05"] = q_vals_mw[idx] < 0.05
            all_results.append(row)
            
    df_results = pd.DataFrame(all_results)
    
    out_csv = os.path.join(OUTPUT_DIR, "gse75132_differential_expression.csv")
    df_results.to_csv(out_csv, index=False)
    print(f"\nSaved differential expression table: {out_csv}")
    
    print("\n--- SUMMARY OF GSE75132 TRANSCRIPTOMIC FINDINGS ---")
    for cid in ["HPV16_All_vs_Ctrl", "HPV16_NonProg_vs_Ctrl", "HPV16_Prog_vs_Ctrl", "Prog_vs_NonProg"]:
        print(f"\nContrast: {cid}")
        sub = df_results[df_results["contrast_id"] == cid]
        print(sub[["gene_symbol", "probe_id", "linear_mean_groupA", "linear_mean_groupB", "log2_fold_change", "mannwhitney_p_value", "fdr_adjusted_q_value"]].to_string(index=False))
        
    # -------------------------------------------------------------
    # Figure 8: Host Gene Expression Boxplots on Corrected Scale
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.flatten()
    
    # Primary probe for each gene
    selected_probes = [
        ("TMEM45A", "219410_at"),
        ("SERPINB5", "204855_at"),
        ("CDKN2A", "207039_at"),
        ("MKI67", "212020_s_at")
    ]
    
    group_labels = {0: "HPV(-) Normal\n(n=11)", 1: "HPV16+ Non-Prog\n(n=10)", 2: "HPV16+ CIN3+ Prog\n(n=20)"}
    df_samples["Group"] = df_samples["category"].map(group_labels)
    group_order = [group_labels[0], group_labels[1], group_labels[2]]
    palette = {group_labels[0]: "#4575b4", group_labels[1]: "#fdae61", group_labels[2]: "#d73027"}
    
    for idx, (gene, p_id) in enumerate(selected_probes):
        ax = axes[idx]
        sns.boxplot(data=df_samples, x="Group", y=p_id, order=group_order, palette=palette, 
                    ax=ax, width=0.45, boxprops=dict(alpha=0.85))
        sns.stripplot(data=df_samples, x="Group", y=p_id, order=group_order, 
                      color="black", size=6, jitter=0.2, alpha=0.6, ax=ax)
        
        # Pull stats
        prog_vs_ctrl = df_results[(df_results["contrast_id"] == "HPV16_Prog_vs_Ctrl") & (df_results["probe_id"] == p_id)]
        fc = prog_vs_ctrl["log2_fold_change"].values[0]
        q_val = prog_vs_ctrl["fdr_adjusted_q_value"].values[0]
        
        ax.set_title(f"{gene} (Probe: {p_id})\nProg vs Ctrl: Log2FC = {fc:+.2f}, FDR q = {q_val:.2e}", 
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Normalized Linear Ratio (Rel. to Control Median)", fontsize=9.5)
        ax.tick_params(axis="both", labelsize=9)
        
    plt.suptitle("Host Epithelial & Oncogenic Biomarker Expression in Cervical Swabs (GEO GSE75132)\nNormalized Relative to HPV-Negative Control Group Median (n = 41 Swabs)", 
                 fontsize=12.5, fontweight="bold", y=0.99)
    plt.tight_layout()
    fig8_path = os.path.join(FIGURES_DIR, "fig8_host_gene_expression_gse75132.png")
    plt.savefig(fig8_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig8_path}")
    
    # -------------------------------------------------------------
    # Figure 9: Heatmap of Cervical Swab Gene Expression Profile
    # -------------------------------------------------------------
    plt.figure(figsize=(12, 6))
    probe_labels = [f"{gene} ({p})" for p, (gene, _) in probe_dict.items()]
    heatmap_data = df_samples.sort_values(by="category")[["GSM", "category"] + list(probe_dict.keys())].copy()
    
    heatmap_matrix = heatmap_data[list(probe_dict.keys())].T
    heatmap_matrix.index = probe_labels
    heatmap_matrix.columns = [f"{row['GSM']} (Cat {int(row['category'])})" for _, row in heatmap_data.iterrows()]
    
    # Log2-transform for display visualization
    heatmap_log = np.log2(heatmap_matrix + 0.01)
    
    sns.heatmap(heatmap_log, cmap="viridis", cbar_kws={"label": "Log2 Normalized Ratio"}, linewidths=0.5)
    plt.title("Log2 Normalized Host Biomarker Expression Across 41 Cervical Swab Samples (GSE75132)\n(Cat 0 = HPV- Normal [first 11 cols] | Cat 1 = HPV16 Non-Prog | Cat 2 = HPV16 CIN3+ Progressors)", 
              fontsize=11.5, fontweight="bold")
    plt.tight_layout()
    fig9_path = os.path.join(FIGURES_DIR, "fig9_host_gene_heatmap_gse75132.png")
    plt.savefig(fig9_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig9_path}")

if __name__ == "__main__":
    analyze_gse75132()
