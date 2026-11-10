"""
generate_pdf_report.py (Scientifically Corrected Version)
- Fully aligned with scientific corrections and rigorous guardrails.
- Reports corrected microbiome-only ML model performance (ROC-AUC ~0.60, PR-AUC ~0.043 vs 0.0265 baseline).
- Documents qPCR HPV measurements as reference ground truth (avoiding circular leakage).
- Incorporates verified GSE75132 host gene expression findings (TMEM45A Log2FC = +3.88 to +4.31, SERPINB5 = +2.73 to +3.22).
- Relabels PRJNA622998 as 'External Study Comparison — Published PRJNA622998 Findings'.
- Replaces causal claims with precise cross-sectional observational language.
- Clarifies diagnostic assay hierarchy and FDA/CLIA regulatory boundaries.
"""

import os
import pandas as pd
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

PDF_PATH = "/Users/ritweekraj/Desktop/HPV/biomarker_report.pdf"
FIGURES_DIR = "/Users/ritweekraj/Desktop/HPV/outputs/figures"

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#666666"))
        
        if self._pageNumber > 1:
            self.drawString(54, 755, "HPV Swab Biological Analysis — Scientific Evaluation & Biomarker Discovery")
            self.setStrokeColor(colors.HexColor("#dddddd"))
            self.setLineWidth(0.5)
            self.line(54, 748, 558, 748)
            
        footer_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 36, footer_text)
        self.drawString(54, 36, "RESEARCH EVALUATION REPORT — NOT FOR DIRECT CLINICAL DIAGNOSTIC USE")
        self.setStrokeColor(colors.HexColor("#dddddd"))
        self.setLineWidth(0.5)
        self.line(54, 48, 558, 48)
        self.restoreState()

def build_pdf():
    print("Building corrected biomarker_report.pdf...")
    doc = SimpleDocTemplate(
        PDF_PATH,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        "DocTitle", parent=styles["Heading1"],
        fontName="Helvetica-Bold", fontSize=18, leading=22,
        textColor=colors.HexColor("#1a365d"), spaceAfter=5
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle", parent=styles["Normal"],
        fontName="Helvetica", fontSize=10, leading=14,
        textColor=colors.HexColor("#4a5568"), spaceAfter=12
    )
    h1_style = ParagraphStyle(
        "H1_Custom", parent=styles["Heading1"],
        fontName="Helvetica-Bold", fontSize=12, leading=16,
        textColor=colors.HexColor("#1a365d"), spaceBefore=10, spaceAfter=5, keepWithNext=True
    )
    h2_style = ParagraphStyle(
        "H2_Custom", parent=styles["Heading2"],
        fontName="Helvetica-Bold", fontSize=10, leading=13,
        textColor=colors.HexColor("#2b6cb0"), spaceBefore=8, spaceAfter=4, keepWithNext=True
    )
    body_style = ParagraphStyle(
        "Body_Custom", parent=styles["BodyText"],
        fontName="Helvetica", fontSize=8.5, leading=11.5,
        textColor=colors.HexColor("#2d3748"), spaceAfter=4
    )
    bullet_style = ParagraphStyle(
        "Bullet_Custom", parent=styles["Normal"],
        fontName="Helvetica", fontSize=8.5, leading=11.5,
        textColor=colors.HexColor("#2d3748"), leftIndent=12, spaceAfter=3
    )
    caption_style = ParagraphStyle(
        "Caption_Custom", parent=styles["Italic"],
        fontName="Helvetica-Oblique", fontSize=7.5, leading=10,
        textColor=colors.HexColor("#718096"), alignment=1, spaceAfter=6
    )
    callout_style = ParagraphStyle(
        "Callout_Custom", parent=styles["Normal"],
        fontName="Helvetica", fontSize=8, leading=11, textColor=colors.HexColor("#742a2a")
    )
    
    story = []
    
    # Header block
    story.append(Paragraph("RESEARCH EVALUATION REPORT", ParagraphStyle("SuperTitle", fontName="Helvetica-Bold", fontSize=8.5, leading=10, textColor=colors.HexColor("#2b6cb0"))))
    story.append(Paragraph("HPV Swab Biological Analysis Using Machine Learning", title_style))
    story.append(Paragraph("Cross-Sectional Investigation of Cervicovaginal Microbial Dysbiosis, Host Epithelial Gene Expression, and Machine Learning Predictability Across Independent Cohorts", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2b6cb0"), spaceAfter=10))
    
    # Metadata table
    meta_table = Table([
        [Paragraph("<b>Primary Cohort:</b> Mendeley pn7h6yjt6k (n=15,607)", body_style), Paragraph("<b>Target Focus:</b> Normal Cytology NILM=1 (n=11,749)", body_style)],
        [Paragraph("<b>Cross-Validation:</b> Repeated Stratified K-Fold (5x3=15 splits)", body_style), Paragraph("<b>Prevalence Baseline:</b> 2.65% hrHPV-positive", body_style)],
        [Paragraph("<b>Host Gene Profiling:</b> GEO GSE75132 (n=41 swabs)", body_style), Paragraph("<b>Literature Comparator:</b> NCBI PRJNA622998 (n=333)", body_style)]
    ], colWidths=[250, 254])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#edf2f7")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 8))
    
    # Section 1: Executive Summary & Scientific Scoping
    story.append(Paragraph("1. Executive Summary & Scientific Scope", h1_style))
    story.append(Paragraph(
        "This evaluation investigates non-invasive cervicovaginal biological measurements to clarify the relationship between "
        "vaginal microbiota, host epithelial transcriptomics, and high-risk HPV (hrHPV) status. "
        "To maintain scientific validity and prevent inflated claims:",
        body_style
    ))
    story.append(Paragraph("• <b>Target Leakage Elimination:</b> All HPV-derived variables (genotypes, viral loads, genotype counts) were strictly excluded from the predictive microbiome model. In Mendeley Table S1, multiplex qPCR measurements define the target label itself (the reference standard); predicting the reference label using its own constituent measurements is circular.", bullet_style))
    story.append(Paragraph("• <b>Normal Cytology Benchmark:</b> Analysis is anchored on women with normal cytology (NILM=1, n=11,749), comparing 11,438 HPV-negative women with 311 hrHPV-positive women. Cases with cytological abnormalities (n=3,858) were separated to avoid confounding early colonization with advanced epithelial dysplasia.", bullet_style))
    story.append(Paragraph("• <b>Modest Microbial Predictability:</b> Repeated stratified cross-validation demonstrates that bacterial qPCR measurements alone achieve an ROC-AUC of <b>0.6005 ± 0.0320</b> and a PR-AUC of <b>0.0428 ± 0.0081</b> (against a random prevalence baseline of 0.0265). The vaginal microbiome reflects host mucosal dysbiosis, but is <b>not a standalone diagnostic substitute</b> for direct HPV DNA testing.", bullet_style))
    story.append(Paragraph("• <b>No Causal Claims:</b> Cross-sectional data cannot prove that dysbiosis preceded HPV infection or caused persistence; findings represent cross-sectional statistical associations.", bullet_style))
    story.append(Spacer(1, 6))
    
    # Section 2: Component Diagnostic Hierarchy
    story.append(Paragraph("2. Biomarker Hierarchy for Rapid Swab Assays", h1_style))
    
    hierarchy_table = Table([
        [Paragraph("<b>Assay Component</b>", body_style), Paragraph("<b>What it CAN Support</b>", body_style), Paragraph("<b>What it CANNOT Establish Alone</b>", body_style)],
        [
            Paragraph("<b>High-Risk HPV DNA / Genotype</b><br/>(HPV 16, 18, 31, 52, 35, 45, etc.)", body_style),
            Paragraph("Direct molecular detection of viral nucleic acid; primary etiological indicator of high-risk HPV infection.", body_style),
            Paragraph("Cannot establish alone whether epithelial neoplasia or cervical cancer is present (most infections transient).", body_style)
        ],
        [
            Paragraph("<b>Host Gene Expression Markers</b><br/>(TMEM45A, SERPINB5, CDKN2A, MKI67)", body_style),
            Paragraph("Exploratory triage into mucosal barrier alterations (TMEM45A) and cell-cycle hyperproliferation.", body_style),
            Paragraph("Cannot provide a clinically validated CIN3+ diagnosis from an exploratory 41-sample dataset.", body_style)
        ],
        [
            Paragraph("<b>Microbial qPCR Biomarkers</b><br/>(BVAB-2, F. vaginae, G. vaginalis, Lactobacilli)", body_style),
            Paragraph("Cross-sectional characterization of vaginal microbial dysbiosis and mucosal vulnerability.", body_style),
            Paragraph("Cannot serve as a standalone HPV diagnostic test (ROC-AUC ~0.60, Precision ~3.6%).", body_style)
        ]
    ], colWidths=[120, 192, 192])
    hierarchy_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf2f7")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(hierarchy_table)
    story.append(Spacer(1, 8))
    
    # Section 3: Primary Mendeley Statistical Analysis
    story.append(Paragraph("3. Primary Bacterial Analysis in Normal Cytology (NILM)", h1_style))
    story.append(Paragraph(
        "In this cross-sectional dataset (n=11,749 normal cytology), hrHPV-positive women showed statistically significant differences "
        "in several bacterial qPCR measurements compared with HPV-negative women. Zeros represent values below the assay limit of detection (<100 copies/mL). "
        "Both unadjusted non-parametric tests and multivariable logistic regression adjusted for Age and BV Status were evaluated:",
        body_style
    ))
    
    df_diff = pd.read_csv("/Users/ritweekraj/Desktop/HPV/biological_differences.csv")
    bact_diff = df_diff[df_diff["category"].str.contains("bact|micro|clinical")].head(8)
    
    diff_table_data = [
        [Paragraph("<b>Feature Name</b>", body_style), Paragraph("<b>Category</b>", body_style), Paragraph("<b>Cliff's &Delta; (95% CI)</b>", body_style), Paragraph("<b>FDR q-val</b>", body_style), Paragraph("<b>Adj. OR (per SD)</b>", body_style), Paragraph("<b>Adj. p-val</b>", body_style)]
    ]
    for _, r in bact_diff.iterrows():
        ci_str = f"[{r['cliffs_delta_ci95_low']:+.2f}, {r['cliffs_delta_ci95_high']:+.2f}]"
        aor_str = f"{r['adjusted_odds_ratio_per_sd']:.2f}" if pd.notnull(r['adjusted_odds_ratio_per_sd']) else "N/A"
        ap_str = f"{r['adjusted_regression_p_value']:.2e}" if pd.notnull(r['adjusted_regression_p_value']) and r['adjusted_regression_p_value']>0 else ("<1e-4" if r['adjusted_regression_p_value']==0 else "N/A")
        diff_table_data.append([
            Paragraph(f"<b>{r['feature']}</b>", body_style),
            Paragraph(r["category"], body_style),
            Paragraph(f"<b>{r['cliffs_delta']:+.3f}</b><br/>{ci_str}", body_style),
            Paragraph(f"{r['adjusted_p_value_fdr']:.2e}", body_style),
            Paragraph(aor_str, body_style),
            Paragraph(ap_str, body_style)
        ])
    t_diff = Table(diff_table_data, colWidths=[110, 94, 110, 65, 65, 60])
    t_diff.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf2f7")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t_diff)
    story.append(Spacer(1, 6))
    
    story.append(Paragraph(
        "<b>Biological Observations (Cross-Sectional):</b><br/>"
        "• <b>Anaerobic BV Pathogens:</b> <i>BVAB-2</i> (&Delta; = +0.132, 95% CI: 0.067 to 0.197, FDR q = 1.0e-6; Adjusted OR = 1.23, p = 0.0006), "
        "<i>Fannyhessea vaginae</i> (&Delta; = +0.121, 95% CI: 0.056 to 0.186, FDR q = 8.7e-5; Adjusted OR = 1.17, p = 0.005), and "
        "<i>Megasphaera sp. Type 1</i> (&Delta; = +0.112, FDR q = 4.8e-5) were enriched in hrHPV-positive women.<br/>"
        "• <b>Lactobacillus Balance:</b> The <i>Lactobacillus-to-Pathogen Ratio</i> was significantly lower in hrHPV-positive women "
        "(&Delta; = -0.118, 95% CI: -0.183 to -0.053, FDR q = 8.0e-4; Adjusted OR = 0.88, p = 0.02). "
        "<i>L. iners</i> showed slight positive association, whereas <i>L. gasseri</i> was depleted.<br/>"
        "• <b>Confounders:</b> Patient Age (t-test p = 6.68e-7; younger age in hrHPV+) and clinical BV Status (&chi;&sup2; p = 0.0011) differed significantly between groups, confirming the necessity of multivariable adjustment.",
        body_style
    ))
    story.append(Spacer(1, 6))
    
    story.append(KeepTogether([
        Image(os.path.join(FIGURES_DIR, "fig1_bacterial_abundance_by_hpv.png"), width=6.6*inch, height=4.6*inch),
        Paragraph("<b>Figure 1:</b> Cervicovaginal Bacterial Abundance in Normal Cytology (NILM=1) stratified by hrHPV status. Gold diamonds indicate group means.", caption_style),
        Spacer(1, 8)
    ]))
    
    story.append(PageBreak())
    
    # Section 4: Exploratory PCA and Correlation
    story.append(Paragraph("4. Exploratory Microbial Landscapes & Correlation", h1_style))
    story.append(Paragraph(
        "Principal Component Analysis (PCA) and Spearman rank correlation mapping illustrate the multivariate structure "
        "of the bacterial qPCR features and clinical covariates (all HPV load and genotype features strictly excluded):",
        body_style
    ))
    
    story.append(KeepTogether([
        Image(os.path.join(FIGURES_DIR, "fig2_pca_microbiome_hpv.png"), width=5.5*inch, height=3.6*inch),
        Paragraph("<b>Figure 2:</b> Principal Component Analysis of Bacterial qPCR Profiles and Demographics in Normal Cytology.", caption_style),
        Spacer(1, 6)
    ]))
    
    story.append(KeepTogether([
        Image(os.path.join(FIGURES_DIR, "fig3_spearman_correlation_heatmap.png"), width=5.5*inch, height=3.9*inch),
        Paragraph("<b>Figure 3:</b> Spearman Rank Correlation Matrix of Bacterial qPCR Profiles and Clinical Covariates (n=11,749).", caption_style),
        Spacer(1, 8)
    ]))
    
    story.append(PageBreak())
    
    # Section 5: Machine Learning Evaluation (Microbiome Only)
    story.append(Paragraph("5. Machine Learning Evaluation (Microbiome Association Only)", h1_style))
    story.append(Paragraph(
        "To test whether bacterial measurements alone predict hrHPV status in cytologically normal women, models were trained "
        "using <b>Repeated Stratified 5-Fold Cross-Validation (3 repeats = 15 evaluation splits)</b>. All scalers and class weighting "
        "were fitted strictly inside training folds. Classification thresholds were selected on training folds via Youden's J statistic "
        "and applied to held-out test folds:",
        body_style
    ))
    
    df_eval = pd.read_csv("/Users/ritweekraj/Desktop/HPV/model_evaluation.csv")
    eval_table_data = [
        [Paragraph("<b>Model Algorithm</b>", body_style), Paragraph("<b>ROC-AUC (Mean &plusmn; SD)</b>", body_style), Paragraph("<b>PR-AUC (vs 0.0265 Baseline)</b>", body_style), Paragraph("<b>Sensitivity</b>", body_style), Paragraph("<b>Specificity</b>", body_style), Paragraph("<b>Precision</b>", body_style), Paragraph("<b>F1-Score</b>", body_style), Paragraph("<b>Brier Score</b>", body_style)]
    ]
    for _, r in df_eval.iterrows():
        eval_table_data.append([
            Paragraph(f"<b>{r['model']}</b>", body_style),
            Paragraph(f"{r['roc_auc_mean']:.4f} &plusmn; {r['roc_auc_std']:.4f}", body_style),
            Paragraph(f"<b>{r['pr_auc_mean']:.4f}</b> &plusmn; {r['pr_auc_std']:.4f}", body_style),
            Paragraph(f"{r['sensitivity_mean']:.3f} &plusmn; {r['sensitivity_std']:.3f}", body_style),
            Paragraph(f"{r['specificity_mean']:.3f} &plusmn; {r['specificity_std']:.3f}", body_style),
            Paragraph(f"{r['precision_mean']:.4f}", body_style),
            Paragraph(f"{r['f1_mean']:.4f}", body_style),
            Paragraph(f"{r['brier_score_mean']:.4f}", body_style)
        ])
    t_eval = Table(eval_table_data, colWidths=[90, 85, 95, 50, 50, 44, 45, 45])
    t_eval.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf2f7")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t_eval)
    story.append(Spacer(1, 6))
    
    alert_box = Table([[
        Paragraph("<b>CRITICAL EVALUATION FINDING:</b><br/>"
                  "• <b>Severe Class Imbalance:</b> In this normal cytology cohort, hrHPV-positive prevalence is only <b>2.65%</b> (311 / 11,749). "
                  "The PR-AUC of <b>0.0428 &plusmn; 0.0081</b> is only modestly above the 0.0265 random chance baseline.<br/>"
                  "• <b>Precision & F1 Limitations:</b> Precision is ~3.6%, meaning ~96% of model positive flags are false positives. "
                  "This quantitatively demonstrates that <b>bacterial qPCR profiles cannot serve as a reliable standalone diagnostic tool for HPV infection</b>. "
                  "They represent biological research candidates reflecting host microenvironmental dysbiosis.", callout_style)
    ]], colWidths=[504])
    alert_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff5f5")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#feb2b2")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(alert_box)
    story.append(Spacer(1, 6))
    
    story.append(KeepTogether([
        Image(os.path.join(FIGURES_DIR, "fig5_model_roc_pr_curves.png"), width=6.6*inch, height=3.3*inch),
        Paragraph("<b>Figure 5:</b> Cross-Validated ROC and Precision-Recall Curves for Microbiome Predictors against the 2.65% Prevalence Baseline.", caption_style),
        Spacer(1, 6)
    ]))
    
    story.append(KeepTogether([
        Image(os.path.join(FIGURES_DIR, "fig7_shap_summary_plot.png"), width=5.5*inch, height=3.8*inch),
        Paragraph("<b>Figure 6:</b> SHAP Summary Plot for XGBoost: Feature attributions of bacterial taxa and age on hrHPV prediction in normal cytology.", caption_style),
        Spacer(1, 8)
    ]))
    
    story.append(PageBreak())
    
    # Section 6: Host Gene Expression (GEO GSE75132)
    story.append(Paragraph("6. Host Epithelial Transcriptomics in Cervical Swabs (GEO GSE75132)", h1_style))
    story.append(Paragraph(
        "Using microarray data from 41 cervical swab specimens (11 HPV-negative controls, 10 persistent HPV16 non-progressors, "
        "and 20 persistent HPV16 cases progressing to CIN3+), we examined target epithelial genes. "
        "<b>Expression Scale Verification:</b> As documented in GEO GSE75132 protocols, values represent normalized linear ratios "
        "divided by the median of HPV-negative controls. Log2 Fold Changes were calculated as log2(mean_A / mean_B):",
        body_style
    ))
    
    df_gse = pd.read_csv("/Users/ritweekraj/Desktop/HPV/outputs/gse75132_differential_expression.csv")
    gse_prog = df_gse[df_gse["contrast_id"] == "HPV16_Prog_vs_Ctrl"]
    
    gse_table_data = [
        [Paragraph("<b>Target Gene</b>", body_style), Paragraph("<b>Probe ID</b>", body_style), Paragraph("<b>Ctrl Mean (n=11)</b>", body_style), Paragraph("<b>CIN3+ Mean (n=20)</b>", body_style), Paragraph("<b>Log2 Fold Change</b>", body_style), Paragraph("<b>p-value (MW-U)</b>", body_style), Paragraph("<b>FDR q-value</b>", body_style)]
    ]
    for _, r in gse_prog.iterrows():
        gse_table_data.append([
            Paragraph(f"<b>{r['gene_symbol']}</b>", body_style),
            Paragraph(f"<code>{r['probe_id']}</code>", body_style),
            Paragraph(f"{r['linear_mean_groupB']:.2f}", body_style),
            Paragraph(f"{r['linear_mean_groupA']:.2f}", body_style),
            Paragraph(f"<b>{r['log2_fold_change']:+.2f}</b>", body_style),
            Paragraph(f"{r['mannwhitney_p_value']:.2e}", body_style),
            Paragraph(f"{r['fdr_adjusted_q_value']:.2e}", body_style)
        ])
    t_gse = Table(gse_table_data, colWidths=[80, 80, 68, 70, 70, 68, 68])
    t_gse.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf2f7")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    story.append(t_gse)
    story.append(Spacer(1, 6))
    
    story.append(Paragraph(
        "<b>Transcriptomic Verification & Scientific Boundaries:</b><br/>"
        "• <b>TMEM45A (Probe 219410_at):</b> Shows significant upregulation in swab RNA: Log2FC = <b>+4.31</b> in CIN3+ progressors vs controls (FDR q = 9.07e-5), "
        "and Log2FC = <b>+2.21</b> in non-progressors vs controls (FDR q = 0.0173). Overall HPV16 vs controls Log2FC is <b>+3.88</b> (a ~14.7-fold linear increase), "
        "NOT +16.22 (which was a previous arithmetic subtraction artifact of linear values).<br/>"
        "• <b>SERPINB5 (Maspin, Probe 204855_at):</b> Significantly upregulated in CIN3+ progressors vs controls (Log2FC = <b>+3.22</b>, FDR q = 0.0120).<br/>"
        "• <b>CDKN2A & MKI67:</b> Exhibit moderate increases in progressors (CDKN2A probe 207039_at Log2FC = +1.66, p = 0.103 vs controls; +1.81, p = 0.0366 vs non-progressors).<br/>"
        "• <i>Sample Size Limitation:</i> With only 41 total swab samples, these transcriptomic differences represent exploratory translational candidates, "
        "NOT a clinically validated diagnostic triage panel.",
        body_style
    ))
    story.append(Spacer(1, 6))
    
    story.append(KeepTogether([
        Image(os.path.join(FIGURES_DIR, "fig8_host_gene_expression_gse75132.png"), width=6.6*inch, height=4.2*inch),
        Paragraph("<b>Figure 7:</b> Host Gene Expression in Swabs (GEO GSE75132) on Verified Normalized Linear Scale.", caption_style),
        Spacer(1, 8)
    ]))
    
    story.append(PageBreak())
    
    # Section 7: External Study Comparison (PRJNA622998)
    story.append(Paragraph("7. External Study Comparison — Published PRJNA622998 Findings", h1_style))
    story.append(Paragraph(
        "<i>Methodological Disclosure:</i> We did not independently download and reprocess raw 16S FASTQ amplicon sequencing reads through QIIME 2/DADA2. "
        "Rather, published findings from <b>NCBI PRJNA622998</b> (McKee et al., <i>Gynecol Pelvic Med</i> 2020; n=308–333 Illumina MiSeq V4 runs) "
        "were synthesized to evaluate literature concordance with our Mendeley qPCR observations:",
        body_style
    ))
    story.append(Paragraph("• <b>Lactobacillus Dominance in Healthy Controls:</b> McKee et al. found that women negative for both cytologic abnormalities and HPV were significantly more likely to feature <i>Lactobacillus</i>-dominated vaginal communities (p < 0.04), concordant with the elevated protective lactobacilli load observed in our normal qPCR cohort.", bullet_style))
    story.append(Paragraph("• <b>Microbial Diversity & <i>Gardnerella vaginalis</i>:</b> Women testing positive for hrHPV or exhibiting cytologic lesions presented significantly higher community alpha diversity and elevated <i>G. vaginalis</i> relative abundance.", bullet_style))
    story.append(Paragraph("• <b><i>Lactobacillus gasseri</i> Depletion:</b> Both 16S amplicon sequencing (McKee et al.) and qPCR (Sekyere et al.) independently identify depletion of <i>L. gasseri</i> in hrHPV-infected women compared with healthy uninfected controls.", bullet_style))
    story.append(Spacer(1, 8))
    
    # Section 8: Home Device Recommendations & Regulatory Guardrails
    story.append(Paragraph("8. Home Device Implications & Regulatory Guardrails", h1_style))
    story.append(Paragraph(
        "For an eventual rapid point-of-care or at-home cervical swab testing platform, biological markers must be partitioned "
        "into clear functional tiers based on analytical strength and clinical role:",
        body_style
    ))
    
    tier_table = Table([
        [Paragraph("<b>Tier</b>", body_style), Paragraph("<b>Target Marker Group</b>", body_style), Paragraph("<b>Measurement Modality</b>", body_style), Paragraph("<b>Intended Role & Scientific Limitations</b>", body_style)],
        [
            Paragraph("<b>Tier 1<br/>Primary Core</b>", body_style),
            Paragraph("<b>Direct High-Risk HPV DNA</b><br/>(Genotypes 16, 18, 31, 52, etc.)", body_style),
            Paragraph("Isothermal amplification (LAMP/RPA) or microfluidic qPCR", body_style),
            Paragraph("<b>Indispensable Diagnostic Engine:</b> Direct molecular detection of viral nucleic acid is the only sufficient indicator of HPV infection.", body_style)
        ],
        [
            Paragraph("<b>Tier 2<br/>Secondary Triage</b>", body_style),
            Paragraph("<b>TMEM45A, SERPINB5 mRNA</b><br/>(CDKN2A / p16INK4a)", body_style),
            Paragraph("Reverse-transcription LAMP or lateral flow immunoassay", body_style),
            Paragraph("<b>Exploratory Dysplasia Triage:</b> Investigational markers to differentiate high-grade progression risk from benign transient carriage. Requires large prospective validation.", body_style)
        ],
        [
            Paragraph("<b>Tier 3<br/>Supportive</b>", body_style),
            Paragraph("<b>Vaginal Dysbiosis Panel</b><br/>(BVAB-2, F. vaginae, Lacto:Pathogen)", body_style),
            Paragraph("Microbial enzymatic/chemical sensor or multiplex bacterial qPCR", body_style),
            Paragraph("<b>Permissiveness / Recurrence Research:</b> Identifies anaerobic dysbiosis and microenvironmental clearance impairment. NOT an HPV diagnostic test.", body_style)
        ]
    ], colWidths=[70, 130, 130, 174])
    tier_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf2f7")),
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#fffdf5")),
        ("BACKGROUND", (0, 2), (-1, 2), colors.HexColor("#f6fffa")),
        ("BACKGROUND", (0, 3), (-1, 3), colors.HexColor("#f8fafc")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(tier_table)
    story.append(Spacer(1, 8))
    
    story.append(Paragraph(
        "<i>Regulatory & Clinical Guardrail:</i> The models, statistical associations, and biomarker rankings in this report are exploratory research findings "
        "derived from retrospective public datasets (Mendeley pn7h6yjt6k, GEO GSE75132, and published PRJNA622998 literature). "
        "They do not constitute evidence of clinical diagnostic efficacy, analytical sensitivity, or medical device readiness. "
        "Translation to a commercial home swab test requires formal analytical limit-of-detection (LoD) verification, cross-reactivity testing, "
        "human sample-adequacy controls (e.g. human beta-globin), and multi-center clinical trial validation under CLIA/FDA regulatory frameworks.",
        body_style
    ))
    
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"biomarker_report.pdf generated successfully at: {PDF_PATH} ({os.path.getsize(PDF_PATH)} bytes)")

if __name__ == "__main__":
    build_pdf()
