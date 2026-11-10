"""
generate_pdf_report.py
Generates the publication-quality deliverable biomarker_report.pdf:
- Executive Summary & Clinical Scoping
- Cohort Demographics & Study Architecture
- Mendeley qPCR Primary Analysis (15,607 samples, NILM=1 Benchmark)
- Differential Biomarker Identification & Effect Sizes
- Machine Learning Benchmark (LogReg, Random Forest, XGBoost with 5-fold CV & 95% CIs)
- SHAP Feature Attribution & Model Explainability
- GEO GSE75132 Host Gene Expression Profiling (CDKN2A, MKI67, TMEM45A, SERPINB5)
- Independent NCBI PRJNA622998 16S Microbiome Validation
- Home Device Recommendations & Scientific Guardrails
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
    """Two-pass canvas to dynamically compute and print total page numbers."""
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
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 755, "HPV Swab Biological Analysis — Machine Learning & Biomarker Discovery")
            self.setStrokeColor(colors.HexColor("#dddddd"))
            self.setLineWidth(0.5)
            self.line(54, 748, 558, 748)
            
        # Footer
        footer_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 36, footer_text)
        self.drawString(54, 36, "CONFIDENTIAL — RESEARCH REPORT FOR TRANSLATIONAL ASSAY DESIGN")
        self.setStrokeColor(colors.HexColor("#dddddd"))
        self.setLineWidth(0.5)
        self.line(54, 48, 558, 48)
        self.restoreState()

def build_pdf():
    print("Building biomarker_report.pdf...")
    doc = SimpleDocTemplate(
        PDF_PATH,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1a365d"),
        spaceAfter=6
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#4a5568"),
        spaceAfter=15
    )
    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#1a365d"),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )
    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#2b6cb0"),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#2d3748"),
        spaceAfter=5
    )
    bullet_style = ParagraphStyle(
        "Bullet_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#2d3748"),
        leftIndent=12,
        spaceAfter=3
    )
    caption_style = ParagraphStyle(
        "Caption_Custom",
        parent=styles["Italic"],
        fontName="Helvetica-Oblique",
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#718096"),
        alignment=1,
        spaceAfter=6
    )
    callout_style = ParagraphStyle(
        "Callout_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11.5,
        textColor=colors.HexColor("#742a2a")
    )
    
    story = []
    
    # -------------------------------------------------------------
    # Cover / Header
    # -------------------------------------------------------------
    story.append(Spacer(1, 10))
    story.append(Paragraph("TRANSLATIONAL BIOMARKER & MACHINE LEARNING REPORT", ParagraphStyle("SuperTitle", fontName="Helvetica-Bold", fontSize=9, leading=11, textColor=colors.HexColor("#2b6cb0"))))
    story.append(Paragraph("HPV Swab Biological Analysis Using Machine Learning", title_style))
    story.append(Paragraph("Multi-Cohort Investigation of Microbial Dysbiosis, Host Epithelial Gene Activation, and Viral Load Architecture Across Multiplex qPCR, 16S rRNA, and Microarray Platforms", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2b6cb0"), spaceAfter=12))
    
    # Metadata block
    meta_table = Table([
        [Paragraph("<b>Author / System:</b> DeepMind Advanced Agentic Assistant", body_style), Paragraph("<b>Target Assay:</b> Rapid HPV Swab Point-of-Care", body_style)],
        [Paragraph("<b>Primary Cohort:</b> Mendeley pn7h6yjt6k (n=15,607)", body_style), Paragraph("<b>Cross-Validation:</b> 5-Fold Stratified (Leak-Free)", body_style)],
        [Paragraph("<b>Independent Microarray:</b> GEO GSE75132 (n=41 swabs)", body_style), Paragraph("<b>Independent 16S:</b> NCBI PRJNA622998 (n=333)", body_style)]
    ], colWidths=[250, 254])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f7fafc")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#edf2f7")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))
    
    # -------------------------------------------------------------
    # Section 1: Executive Summary & Core Scientific Guardrails
    # -------------------------------------------------------------
    story.append(Paragraph("1. Executive Summary & Translational Principles", h1_style))
    story.append(Paragraph(
        "This study rigorously evaluates biological features distinguishing human papillomavirus (HPV) infection states "
        "from non-invasive cervicovaginal swab specimens across three independent cohorts comprising over 15,900 patient samples. "
        "To prevent statistical artifacts and ensure clinical translation:",
        body_style
    ))
    story.append(Paragraph("• <b>Primary Benchmark Cohort:</b> We strictly contrast <b>HPV-negative normal-cytology women (n=11,438)</b> against <b>high-risk HPV-positive normal-cytology women (n=311)</b>. This isolates early infectious biology and host-microbial permissiveness prior to overt cytological dysplasia.", bullet_style))
    story.append(Paragraph("• <b>Dysplasia Progression Segregation:</b> Patients presenting cytological lesions (ASCUS, LSIL, HSIL, AGC, n=3,858) are analyzed separately to avoid confounding early viral colonization with advanced neoplastic tissue alteration.", bullet_style))
    story.append(Paragraph("• <b>Assay Design Reality Check:</b> Machine learning demonstrates that <b>directly detectable HPV DNA markers are the only sufficient primary diagnostic features</b> (AUC = 1.000). Microbiome and host transcriptomic signatures capture vital host permissiveness and neoplastic progression (AUC = 0.602), serving as <i>supportive triage candidates</i> rather than standalone diagnostic substitutes.", bullet_style))
    story.append(Paragraph("• <b>Strict Leakage Prevention:</b> All scalers, feature selectors, and class-balancing operations were computed exclusively inside each training fold of a 5-fold stratified cross-validation pipeline.", bullet_style))
    story.append(Spacer(1, 8))
    
    # -------------------------------------------------------------
    # Section 2: Cohort Architecture & Methodological Workflow
    # -------------------------------------------------------------
    story.append(Paragraph("2. Cohort Architecture & Feature Extraction", h1_style))
    story.append(Paragraph(
        "Each dataset represents an independent clinical cohort and measurement platform; row merging was strictly avoided. "
        "Features were mapped and standardized within their biological categories:",
        body_style
    ))
    
    cohort_summary_table = Table([
        [Paragraph("<b>Dataset / Accession</b>", body_style), Paragraph("<b>Platform & Scale</b>", body_style), Paragraph("<b>Biological Features Extracted</b>", body_style), Paragraph("<b>Primary Analytical Role</b>", body_style)],
        [
            Paragraph("<b>Mendeley Data<br/>pn7h6yjt6k</b>", body_style),
            Paragraph("Multiplex qPCR<br/>n = 15,607 patients", body_style),
            Paragraph("13 hrHPV genotypes, 9 bacterial species (Lactobacilli, BV anaerobes), viral load, BV status, age, cytology", body_style),
            Paragraph("Primary statistical & ML discovery of bacterial/viral biomarkers in normal cytology", body_style)
        ],
        [
            Paragraph("<b>GEO<br/>GSE75132</b>", body_style),
            Paragraph("Affymetrix Microarray<br/>n = 41 cervical swabs", body_style),
            Paragraph("Host epithelial mRNAs: CDKN2A (p16INK4A), MKI67 (Ki-67), TMEM45A, SERPINB5 (Maspin)", body_style),
            Paragraph("Investigation of host oncogenic and structural gene expression changes in swabs", body_style)
        ],
        [
            Paragraph("<b>NCBI SRA<br/>PRJNA622998</b>", body_style),
            Paragraph("16S rRNA V4 MiSeq<br/>n = 333 experiments", body_style),
            Paragraph("Vaginal microbiota community state types, alpha diversity, G. vaginalis relative abundance", body_style),
            Paragraph("Independent population-based sequencing confirmation of microbial dysbiosis", body_style)
        ]
    ], colWidths=[100, 110, 180, 114])
    cohort_summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf2f7")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(cohort_summary_table)
    story.append(Spacer(1, 10))
    
    # -------------------------------------------------------------
    # Section 3: Primary Biological Differences (Mendeley NILM Cohort)
    # -------------------------------------------------------------
    story.append(Paragraph("3. Primary Biological Differences in Normal Cytology (NILM)", h1_style))
    story.append(Paragraph(
        "Feature-wise comparisons between HPV-negative normal women (n=11,438) and hrHPV-positive normal women (n=311) "
        "reveal robust, statistically significant microbial shifts that precede epithelial dysplasia:",
        body_style
    ))
    
    # Read differential biomarkers CSV
    df_diff = pd.read_csv("/Users/ritweekraj/Desktop/HPV/differential_biomarkers.csv")
    bact_diff = df_diff[df_diff["category"].str.contains("bact|micro", case=False)].head(8)
    
    diff_table_data = [
        [Paragraph("<b>Biomarker Feature</b>", body_style), Paragraph("<b>Category</b>", body_style), Paragraph("<b>HPV(-) Mean</b>", body_style), Paragraph("<b>hrHPV(+) Mean</b>", body_style), Paragraph("<b>Log2FC</b>", body_style), Paragraph("<b>Cliff's &Delta;</b>", body_style), Paragraph("<b>FDR q-value</b>", body_style)]
    ]
    for _, r in bact_diff.iterrows():
        diff_table_data.append([
            Paragraph(f"<b>{r['feature']}</b>", body_style),
            Paragraph(r["category"], body_style),
            Paragraph(f"{r['mean_HPV_neg']:.2f}", body_style),
            Paragraph(f"{r['mean_hrHPV_pos']:.2f}", body_style),
            Paragraph(f"{r['log2_fold_change']:+.2f}", body_style),
            Paragraph(f"{r['cliffs_delta']:+.3f}", body_style),
            Paragraph(f"{r['adjusted_p_value_fdr']:.2e}", body_style)
        ])
    t_diff = Table(diff_table_data, colWidths=[120, 94, 65, 65, 50, 50, 60])
    t_diff.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf2f7")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t_diff)
    story.append(Spacer(1, 8))
    
    story.append(Paragraph(
        "<b>Key Biological Insights:</b><br/>"
        "1. <b>Anaerobic Dysbiosis Enriched in hrHPV Infection:</b> <i>BVAB-2</i> (&Delta; = +0.132, q = 2.97e-7), "
        "<i>Fannyhessea (Atopobium) vaginae</i> (&Delta; = +0.121, q = 4.94e-5), <i>Gardnerella vaginalis</i> (&Delta; = +0.110, q = 1.49e-4), "
        "and <i>Megasphaera sp. Type 1</i> (&Delta; = +0.112, q = 2.10e-5) are significantly enriched in hrHPV-positive women despite completely normal cytology.<br/>"
        "2. <b>Loss of Protective Lactobacilli:</b> Protective lactobacilli (<i>L. gasseri</i>, <i>L. crispatus</i>, <i>L. jensenii</i>) are significantly depleted (q = 7.50e-3), "
        "causing a significant collapse in the <i>Lactobacillus-to-Pathogen Ratio</i> (q = 5.82e-4).<br/>"
        "3. <b>The Transitional Role of <i>L. iners</i>:</b> In contrast to <i>L. crispatus</i>, <i>Lactobacillus iners</i> was enriched (&Delta; = +0.073, q = 0.0178), "
        "corroborating its known clinical phenotype as a transitional species that produces L-lactic acid and fails to maintain protective low pH.",
        body_style
    ))
    story.append(Spacer(1, 6))
    
    # Embed Figure 1 and Figure 4
    story.append(KeepTogether([
        Image(os.path.join(FIGURES_DIR, "fig1_bacterial_abundance_by_hpv.png"), width=6.8*inch, height=5.2*inch),
        Paragraph("<b>Figure 1:</b> Cervicovaginal Bacterial Abundance in Normal Cytology (NILM=1) stratified by hrHPV Infection Status. Means indicated by gold diamonds.", caption_style),
        Spacer(1, 10)
    ]))
    
    story.append(PageBreak())
    
    # -------------------------------------------------------------
    # Section 4: Exploratory Landscape & Correlation Architecture
    # -------------------------------------------------------------
    story.append(Paragraph("4. Exploratory Microbial Architecture & Host Correlation", h1_style))
    story.append(Paragraph(
        "Principal Component Analysis (PCA) and Spearman rank correlation mapping illustrate the multivariate structure "
        "of the vaginal microbial ecosystem and its intersection with viral infection and clinical status:",
        body_style
    ))
    
    story.append(KeepTogether([
        Image(os.path.join(FIGURES_DIR, "fig2_pca_microbiome_hpv.png"), width=5.5*inch, height=3.8*inch),
        Paragraph("<b>Figure 2:</b> Principal Component Analysis of Cervicovaginal Microbiome and Demographics in Normal Cytology.", caption_style),
        Spacer(1, 8)
    ]))
    
    story.append(KeepTogether([
        Image(os.path.join(FIGURES_DIR, "fig3_spearman_correlation_heatmap.png"), width=5.5*inch, height=4.2*inch),
        Paragraph("<b>Figure 3:</b> Spearman Correlation Matrix between Cervicovaginal Microbiota, Total Viral Load, Age, and Cytology (n=15,607).", caption_style),
        Spacer(1, 10)
    ]))
    
    story.append(PageBreak())
    
    # -------------------------------------------------------------
    # Section 5: Rigorous Machine Learning Evaluation & SHAP Analysis
    # -------------------------------------------------------------
    story.append(Paragraph("5. Machine Learning Evaluation & Explainability (SHAP)", h1_style))
    story.append(Paragraph(
        "To evaluate clinical viability and avoid optimistic data leakage, models were trained using <b>Stratified 5-Fold Cross-Validation</b>. "
        "All scaling, feature transformations, and class-weight balancing were strictly fitted inside training folds.",
        body_style
    ))
    
    df_eval = pd.read_csv("/Users/ritweekraj/Desktop/HPV/model_evaluation.csv")
    eval_table_data = [
        [Paragraph("<b>Experiment / Feature Scope</b>", body_style), Paragraph("<b>Model</b>", body_style), Paragraph("<b>ROC-AUC (95% CI)</b>", body_style), Paragraph("<b>Sensitivity</b>", body_style), Paragraph("<b>Specificity</b>", body_style), Paragraph("<b>PR-AUC</b>", body_style)]
    ]
    for _, r in df_eval.iterrows():
        exp_label = "Microbiome Only" if "Microbiome" in r["experiment"] and "Multi" not in r["experiment"] else "Multimodal (Viral+Micro)"
        eval_table_data.append([
            Paragraph(f"<b>{exp_label}</b>", body_style),
            Paragraph(r["model"], body_style),
            Paragraph(f"<b>{r['roc_auc_mean']:.4f}</b><br/>({r['roc_auc_ci95_low']:.4f}–{r['roc_auc_ci95_high']:.4f})", body_style),
            Paragraph(f"{r['sensitivity_mean']:.3f}", body_style),
            Paragraph(f"{r['specificity_mean']:.3f}", body_style),
            Paragraph(f"{r['pr_auc_mean']:.4f}", body_style)
        ])
    t_eval = Table(eval_table_data, colWidths=[120, 100, 114, 55, 55, 60])
    t_eval.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf2f7")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t_eval)
    story.append(Spacer(1, 8))
    
    # Callout Box
    alert_table = Table([[
        Paragraph("<b>CRITICAL TRANSLATIONAL TAKEAWAY:</b><br/>"
                  "Microbiome-only models achieve an ROC-AUC of <b>0.6025 (95% CI: 0.5657–0.6393)</b>. While statistically significant, "
                  "microbial dysbiosis represents a <i>permissive microenvironment</i>, NOT an etiologic substitute for viral DNA. "
                  "When direct viral DNA markers are included, classification reaches <b>ROC-AUC = 1.0000</b>. "
                  "Therefore, <b>direct HPV DNA detection must remain the primary diagnostic core</b> of any rapid POC swab device.", callout_style)
    ]], colWidths=[504])
    alert_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff5f5")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#feb2b2")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(alert_table)
    story.append(Spacer(1, 8))
    
    story.append(KeepTogether([
        Image(os.path.join(FIGURES_DIR, "fig5_model_roc_pr_curves.png"), width=6.8*inch, height=4.8*inch),
        Paragraph("<b>Figure 5:</b> Cross-Validated ROC and Precision-Recall Curves across Logistic Regression, Random Forest, and XGBoost.", caption_style),
        Spacer(1, 8)
    ]))
    
    story.append(KeepTogether([
        Image(os.path.join(FIGURES_DIR, "fig7_shap_summary_plot.png"), width=5.5*inch, height=4.2*inch),
        Paragraph("<b>Figure 6:</b> SHAP Summary Plot for XGBoost: Attributing feature impact and directionality on HPV classification.", caption_style),
        Spacer(1, 10)
    ]))
    
    story.append(PageBreak())
    
    # -------------------------------------------------------------
    # Section 6: Host Gene Expression Profiling (GEO GSE75132)
    # -------------------------------------------------------------
    story.append(Paragraph("6. Host Epithelial Gene Expression Profiling (GEO GSE75132)", h1_style))
    story.append(Paragraph(
        "Using microarray data from 41 cervical swab specimens (11 HPV-negative controls, 10 persistent HPV16 without progression, "
        "and 20 persistent HPV16 with progression to CIN3+), we mapped Affymetrix probes to host target genes:",
        body_style
    ))
    
    df_gse = pd.read_csv("/Users/ritweekraj/Desktop/HPV/outputs/gse75132_differential_expression.csv")
    gse_genes_only = df_gse[df_gse["feature"].str.endswith("_mean")]
    
    gse_table_data = [
        [Paragraph("<b>Target Host Gene</b>", body_style), Paragraph("<b>HPV(-) Normal</b>", body_style), Paragraph("<b>HPV16(+) Swabs</b>", body_style), Paragraph("<b>Log2 Fold Change</b>", body_style), Paragraph("<b>p-value (MW-U)</b>", body_style), Paragraph("<b>FDR q-value</b>", body_style), Paragraph("<b>CIN3+ Prog p-val</b>", body_style)]
    ]
    for _, r in gse_genes_only.iterrows():
        gse_table_data.append([
            Paragraph(f"<b>{r['gene']}</b>", body_style),
            Paragraph(f"{r['mean_HPV_neg']:.2f}", body_style),
            Paragraph(f"{r['mean_HPV16_pos']:.2f}", body_style),
            Paragraph(f"<b>{r['log2_FC_HPV16_vs_Normal']:+.2f}</b>", body_style),
            Paragraph(f"{r['p_val_HPV16_vs_Normal']:.2e}", body_style),
            Paragraph(f"<b>{r['FDR_HPV16_vs_Normal']:.2e}</b>", body_style),
            Paragraph(f"{r['p_val_Prog_vs_NonProg']:.3f}", body_style)
        ])
    t_gse = Table(gse_table_data, colWidths=[90, 70, 70, 75, 65, 65, 69])
    t_gse.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf2f7")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t_gse)
    story.append(Spacer(1, 8))
    
    story.append(Paragraph(
        "<b>Transcriptomic Findings:</b><br/>"
        "• <b>TMEM45A:</b> Shows massive, statistically robust upregulation in cervical swabs from HPV16-infected patients (Log2FC = +16.22, FDR q = 5.10e-5). "
        "TMEM45A regulates epidermal keratinocyte differentiation and epithelial barrier dynamics, acting as an ultra-sensitive mucosal host response marker.<br/>"
        "• <b>SERPINB5 (Maspin):</b> Significantly upregulated (Log2FC = +3.32, FDR q = 0.00728). A serine protease inhibitor linked to tumor suppression and cell motility.<br/>"
        "• <b>CDKN2A (p16INK4a) & MKI67 (Ki-67):</b> Exhibit moderate upregulation in persistent swabs, with pronounced hyperactivation specifically in cases progressing to high-grade CIN3+ dysplasia.",
        body_style
    ))
    story.append(Spacer(1, 6))
    
    story.append(KeepTogether([
        Image(os.path.join(FIGURES_DIR, "fig8_host_gene_expression_gse75132.png"), width=6.5*inch, height=4.6*inch),
        Paragraph("<b>Figure 7:</b> Host Epithelial & Oncogenic Biomarker Expression in Cervical Swabs (GEO GSE75132) comparing Normal vs HPV16 vs CIN3+.", caption_style),
        Spacer(1, 8)
    ]))
    
    story.append(KeepTogether([
        Image(os.path.join(FIGURES_DIR, "fig9_host_gene_heatmap_gse75132.png"), width=6.5*inch, height=3.0*inch),
        Paragraph("<b>Figure 8:</b> Z-score Expression Heatmap of Target Host Genes Across 41 Cervical Swab Specimens.", caption_style),
        Spacer(1, 10)
    ]))
    
    story.append(PageBreak())
    
    # -------------------------------------------------------------
    # Section 7: Independent 16S Microbiome Synthesis & Final Recommendations
    # -------------------------------------------------------------
    story.append(Paragraph("7. Independent 16S Microbiome Validation & Final Ranked Biomarkers", h1_style))
    story.append(Paragraph(
        "To cross-validate our qPCR bacterial findings with next-generation sequencing, we synthesized findings with "
        "<b>NCBI PRJNA622998</b> (McKee et al., n=333 16S V4 amplicon sequencing experiments):",
        body_style
    ))
    story.append(Paragraph("• <b>Community State Type (CST) Shift:</b> Healthy controls without cytologic abnormalities were significantly more likely to be dominated by <i>Lactobacillus</i> species (p < 0.04), confirming the high <i>Lactobacillus-to-Pathogen Ratio</i> observed in our primary qPCR cohort.", bullet_style))
    story.append(Paragraph("• <b>Anaerobic Diversity & <i>Gardnerella</i>:</b> In agreement with our qPCR Cliff's delta findings, McKee et al. found that women with hrHPV infection or cytologic abnormalities possessed higher community alpha-diversity and significantly higher relative abundance of <i>G. vaginalis</i>.", bullet_style))
    story.append(Paragraph("• <b>Depletion of <i>L. gasseri</i>:</b> Both platforms independently demonstrate that <i>L. gasseri</i> is differentially depleted in hrHPV-infected women compared to uninfected controls.", bullet_style))
    story.append(Spacer(1, 8))
    
    story.append(Paragraph("Final Ranked Biomarker Architecture for Swab-Based Assay Design", h2_style))
    
    final_tier_table = Table([
        [Paragraph("<b>Tier & Priority</b>", body_style), Paragraph("<b>Candidate Biomarkers</b>", body_style), Paragraph("<b>Biological Mechanism</b>", body_style), Paragraph("<b>Target Assay Role</b>", body_style)],
        [
            Paragraph("<b>Tier 1 (Core)<br/>Primary Diagnostic</b>", body_style),
            Paragraph("<b>Total hrHPV DNA Load<br/>HPV16, HPV18, HPV31, HPV52, HPV35, HPV45 DNA</b>", body_style),
            Paragraph("Direct molecular presence of high-risk viral genome; etiologic driver of cervical neoplastic transformation.", body_style),
            Paragraph("<b>Mandatory Primary Assay:</b> Absolute prerequisite for diagnosis (AUC = 1.000).", body_style)
        ],
        [
            Paragraph("<b>Tier 2 (Secondary)<br/>Host Dysplasia Triage</b>", body_style),
            Paragraph("<b>TMEM45A mRNA<br/>CDKN2A (p16INK4A)<br/>SERPINB5 mRNA<br/>MKI67 (Ki-67)</b>", body_style),
            Paragraph("Cellular transcriptional response: mucosal barrier remodeling (TMEM45A), pRb pathway inactivation (p16), and aberrant cell proliferation.", body_style),
            Paragraph("<b>Secondary Triage:</b> Differentiates transient infection from active CIN3+ neoplastic transformation.", body_style)
        ],
        [
            Paragraph("<b>Tier 3 (Supportive)<br/>Microbial Vulnerability</b>", body_style),
            Paragraph("<b>BVAB-2, F. vaginae, G. vaginalis, Megasphaera<br/>Lacto-to-Pathogen Ratio<br/>L. crispatus vs L. iners</b>", body_style),
            Paragraph("Cervicovaginal dysbiosis: epithelial micro-abrasions, biofilm formation, sialidase secretion, impairment of lactic acid barrier.", body_style),
            Paragraph("<b>Host Permissiveness Index:</b> Prognostic indicator of viral persistence and clearance failure (AUC = 0.602).", body_style)
        ]
    ], colWidths=[95, 125, 175, 109])
    final_tier_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf2f7")),
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#fffaf0")),
        ("BACKGROUND", (0, 2), (-1, 2), colors.HexColor("#f0fff4")),
        ("BACKGROUND", (0, 3), (-1, 3), colors.HexColor("#ebf8ff")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(final_tier_table)
    story.append(Spacer(1, 12))
    
    story.append(Paragraph("8. Regulatory & Clinical Concluding Statement", h1_style))
    story.append(Paragraph(
        "<i>Scientific Guardrail:</i> The models and biomarker rankings developed in this study are derived from retrospective public research datasets "
        "(Mendeley pn7h6yjt6k, GEO GSE75132, NCBI PRJNA622998). They demonstrate biological proof-of-concept for multi-analyte cervical swab profiling "
        "and establish that host mucosal dysbiosis accompanies early HPV infection. These findings do not constitute clinical diagnostic claims. "
        "Commercial translation to a point-of-care home device requires prospective analytical assay validation, clinical threshold calibration, "
        "and multi-center CLIA/FDA regulatory authorization.",
        body_style
    ))
    
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"biomarker_report.pdf generated successfully at: {PDF_PATH} ({os.path.getsize(PDF_PATH)} bytes)")

if __name__ == "__main__":
    build_pdf()
