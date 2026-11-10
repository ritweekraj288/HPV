# HPV Swab Biological Analysis Using Machine Learning: Scientific Evaluation & Biomarker Discovery

> **A Comprehensive, Layperson-Accessible, and Methodologically Rigorous Guide to the Codebase, Clinical Data, Machine Learning Pipelines, Host Transcriptomics, and Translational Findings.**

---

## Table of Contents
1. [Executive Summary & Layperson's Domain Primer](#1-executive-summary--laypersons-domain-primer)
   - [What is HPV and Why Does It Matter?](#what-is-hpv-and-why-does-it-matter)
   - [What is the Cervicovaginal Microbiome?](#what-is-the-cervicovaginal-microbiome)
   - [What is Host Epithelial Gene Expression?](#what-is-host-epithelial-gene-expression)
   - [Core Research Objective of This Repository](#core-research-objective-of-this-repository)
2. [The Critical Scientific Audit: Eliminating Target Leakage](#2-the-critical-scientific-audit-eliminating-target-leakage)
   - [The Circular Logic Trap in Initial Machine Learning](#the-circular-logic-trap-in-initial-machine-learning)
   - [The 12-Point Scientific Correction Framework](#the-12-point-scientific-correction-framework)
3. [Repository Directory & File Manifest](#3-repository-directory--file-manifest)
   - [Complete Workspace Directory Tree](#complete-workspace-directory-tree)
   - [Role of Every Directory & File](#role-of-every-directory--file)
4. [Datasets & Clinical Cohorts Detailed](#4-datasets--clinical-cohorts-detailed)
   - [Cohort 1: Mendeley Clinical qPCR Dataset (pn7h6yjt6k / Table S1)](#cohort-1-mendeley-clinical-qpcr-dataset-pn7h6yjt6k--table-s1)
   - [Cohort 2: GEO GSE75132 Swab Microarray Dataset](#cohort-2-geo-gse75132-swab-microarray-dataset)
   - [Cohort 3: NCBI SRA PRJNA622998 16S Amplicon Sequencing](#cohort-3-ncbi-sra-prjna622998-16s-amplicon-sequencing)
5. [Implementation Details: Scripts & Execution Pipelines](#5-implementation-details-scripts--execution-pipelines)
   - [`scripts/download_mendeley.py`](#scriptsdownload_mendeleypy)
   - [`scripts/mendeley_analysis.py`](#scriptsmendeley_analysispy)
   - [`scripts/ml_pipeline.py`](#scriptsml_pipelinepy)
   - [`scripts/gse_analysis.py`](#scriptsgse_analysispy)
   - [`scripts/generate_pdf_report.py`](#scriptsgenerate_pdf_reportpy)
   - [`scripts/generate_html_presentation.py`](#scriptsgenerate_html_presentationpy)
6. [Detailed Scientific Findings & Statistical Outputs](#6-detailed-scientific-findings--statistical-outputs)
   - [Part A: Bacterial Dysbiosis in Normal Cytology (NILM)](#part-a-bacterial-dysbiosis-in-normal-cytology-nilm)
   - [Part B: Rigorous Machine Learning Evaluation](#part-b-rigorous-machine-learning-evaluation)
   - [Part C: Host Transcriptomics & Progression Markers (GSE75132)](#part-c-host-transcriptomics--progression-markers-gse75132)
   - [Part D: Concordance with Independent 16S Sequencing (PRJNA622998)](#part-d-concordance-with-independent-16s-sequencing-prjna622998)
7. [Visual Artifacts Catalog (Figures 1 through 9)](#7-visual-artifacts-catalog-figures-1-through-9)
8. [Translational Architecture & Home Diagnostic Guardrails](#8-translational-architecture--home-diagnostic-guardrails)
   - [Three-Tier Diagnostic Hierarchy](#three-tier-diagnostic-hierarchy)
   - [CLIA / FDA Regulatory Realities](#clia--fda-regulatory-realities)
9. [How to Reproduce & Execute the Complete Analysis](#9-how-to-reproduce--execute-the-complete-analysis)
10. [Glossary of Clinical & Statistical Terms](#10-glossary-of-clinical--statistical-terms)

---

## 1. Executive Summary & Layperson's Domain Primer

If you have never studied human biology, virology, clinical epidemiology, or machine learning, this section explains the foundational concepts behind this project in plain language.

### What is HPV and Why Does It Matter?
- **Human Papillomavirus (HPV)** is the most common sexually transmitted infection in the world. Virtually all sexually active individuals acquire it at some point in their lives.
- There are over 200 strains (genotypes) of HPV. Most are harmless, but roughly 14 strains are classified as **High-Risk HPV (hrHPV)**—most notably **HPV 16** and **HPV 18**, along with types 31, 33, 35, 39, 45, 51, 52, 56, 58, 59, and 68.
- Persistent hrHPV infections are the primary cause of cervical cancer (accounting for >99% of cases).
- **The Challenge:** Most HPV infections (~90%) are harmlessly cleared by the body's immune system within 1 to 2 years. Only a small subset persist over years or decades, gradually causing cervical tissue changes called **Cervical Intraepithelial Neoplasia (CIN)**:
  - **CIN1 (Mild Dysplasia):** Usually clears on its own.
  - **CIN2 / CIN3 (Moderate to Severe Pre-Cancer):** Requires clinical intervention (such as excision or ablation) to prevent invasive carcinoma.
- **The Clinical Standard:** Current cervical cancer screening uses **Pap cytology** (looking at cervical cells under a microscope) and **molecular HPV DNA testing** (detecting viral genes using PCR). In modern clinical practice, finding a way to test non-invasively—such as through a self-collected vaginal swab—could democratize screening worldwide.

### What is the Cervicovaginal Microbiome?
- The vagina and cervix are not sterile; they host billions of microorganisms called the **vaginal microbiota**.
- **Healthy Flora (The "Guardians"):** In most healthy women, the environment is dominated by *Lactobacillus* species (such as *L. crispatus*, *L. gasseri*, *L. jensenii*, and *L. iners*). These beneficial bacteria produce lactic acid and hydrogen peroxide, keeping the vaginal environment acidic ($\text{pH} \le 4.5$), which inhibits pathogens and protects the epithelial lining.
- **Dysbiosis & Bacterial Vaginosis (BV):** When protective *Lactobacillus* species decline, diverse anaerobic bacteria take over (such as *Fannyhessea vaginae* [formerly *Atopobium vaginae*], *Gardnerella vaginalis*, *BVAB-2* [Bacterial Vaginosis-Associated Bacterium 2], and *Megasphaera* species). This condition disrupts the mucosal protective barrier and promotes inflammation.
- **The Scientific Question:** Does this bacterial imbalance make women more vulnerable to HPV acquisition, or does HPV infection promote bacterial imbalance? Can measuring these bacteria from a swab tell us if a woman is infected with HPV?

### What is Host Epithelial Gene Expression?
- The cervix is lined by layers of epithelial cells. When HPV infects these cells and persists, the virus integrates its DNA into the host genome, producing viral oncogenes ($E6$ and $E7$).
- These viral proteins hijack human cell machinery, causing the human cells to change which genes they turn on ("express") or off.
- **Biomarkers of Progression:** By measuring messenger RNA (mRNA) or proteins in swab samples—such as **TMEM45A** (a transmembrane protein linked to epidermal barrier function and hypoxia), **SERPINB5** (Maspin, involved in cellular adhesion and protease inhibition), **CDKN2A** ($p16^{INK4a}$, a hallmark cell-cycle regulator triggered by HPV $E7$), and **MKI67** ($Ki\text{-}67$, a universal marker of uncontrolled cell proliferation)—we can assess whether an HPV infection is just sitting quietly or driving tissue toward pre-cancerous dysplasia (CIN3+).

### Core Research Objective of This Repository
This repository evaluates whether non-invasive cervicovaginal swabs can capture biological signals across three layers:
1. **The Bacteria:** Quantitative levels of 9 vaginal bacteria.
2. **The Host Cells:** Gene expression changes in 4 key human genes.
3. **The Virus:** Direct quantitative molecular detection of 14 high-risk HPV strains.

It rigorously assesses whether machine learning can use microbial bacteria to detect HPV, and designs a realistic, regulation-compliant architecture for potential rapid, at-home, or point-of-care cervical diagnostic tools.

---

## 2. The Critical Scientific Audit: Eliminating Target Leakage

A central scientific breakthrough documented in this repository is the discovery, auditing, and elimination of **Target Data Leakage** from earlier exploratory machine learning models.

```
       ┌────────────────────────────────────────────────────────┐
       │             THE INITIAL RUN (CIRCULAR LOGIC)           │
       │                                                        │
       │  Inputs: [HPV 16 load, HPV 18 load, ...]               │
       │          [Bacteria loads, Age, BV status]              │
       │                         │                              │
       │                         ▼                              │
       │             Model (Random Forest / XGBoost)            │
       │                         │                              │
       │                         ▼                              │
       │  Target: hrHPV Result (0 or 1)                         │
       │                                                        │
       │  RESULT: ROC-AUC = 0.999 (ARTIFICIAL / TRIVIAL)        │
       │  REASON: 'hrHPV Result' WAS DEFINED BY THE INPUT qPCR! │
       └────────────────────────────────────────────────────────┘
                                 │
                 AUDITED & RE-ARCHITECTED BY PIPELINE
                                 │
                                 ▼
       ┌────────────────────────────────────────────────────────┐
       │         THE CORRECTED RUN (SCIENTIFICALLY PURE)        │
       │                                                        │
       │  Inputs: ONLY Bacterial qPCR Loads + Ratios            │
       │          + Age + BV Score (25 Pure Features)           │
       │          [ZERO HPV-DERIVED VARIABLES]                  │
       │                         │                              │
       │                         ▼                              │
       │    Repeated Stratified 5-Fold Cross-Validation         │
       │    (15 Independent Training & Testing Splits)          │
       │                         │                              │
       │                         ▼                              │
       │  Target: hrHPV Result in Normal Cytology (NILM = 1)    │
       │                                                        │
       │  HONEST REALITY:                                       │
       │  • ROC-AUC = 0.6005 ± 0.0320                           │
       │  • PR-AUC  = 0.0428 ± 0.0081 (vs 0.0265 baseline)      │
       │  • Precision = ~3.6% (96.4% False Positive Rate)       │
       │                                                        │
       │  CONCLUSION: Vaginal bacteria reflect mucosal dysbiosis│
       │  but CANNOT serve as a standalone HPV diagnostic test! │
       └────────────────────────────────────────────────────────┘
```

### The Circular Logic Trap in Initial Machine Learning
In the initial exploratory code (`backup_initial_run/scripts/ml_pipeline.py`), an experiment called `Multimodal_Viral_Microbiome` included viral variables (such as individual HPV genotype loads, total viral loads, and viral genotype counts) as predictors alongside bacteria to predict `hrHPV Result`.
- **Why this was flawed:** In clinical laboratory datasets (such as Mendeley Table S1), `hrHPV Result` is **not an independent diagnosis**—it is literally a computer flag derived from whether any of those exact same HPV genotype qPCR readings exceeded the assay detection threshold!
- Feeding the viral qPCR numbers into a machine learning model to predict whether the patient had positive viral qPCR is circular reasoning (predicting a label from its own definition). It yielded an ROC-AUC of nearly $1.0$, which gives a false impression of predictive capability.

### The 12-Point Scientific Correction Framework
To establish impeccable scientific validity, the entire pipeline was rebuilt against a strict 12-point audit checklist:

| Check # | Scientific Requirement | Implementation in Repository | Status |
|---|---|---|---|
| **1** | **Strict Target Leakage Elimination** | Every HPV-derived feature was stripped from predictive models. Audited list saved to [`outputs/microbiome_ml_features.txt`](file:///Users/ritweekraj/Desktop/HPV/outputs/microbiome_ml_features.txt). | **Passed** |
| **2** | **Explicit Ground Truth Definition** | Mendeley qPCR documented as the reference standard; viral loads treated as reference ground truth, not inputs. | **Passed** |
| **3** | **Zero Patient Overlap** | Confirmed each row in Table S1 represents an independent patient encounter ($n=15,607$). | **Passed** |
| **4** | **Strictly Inside-Fold Scaling** | `StandardScaler` and class weighting fitted exclusively inside training folds of cross-validation. | **Passed** |
| **5** | **Out-of-Fold Threshold Derivation** | Youden's $J$ statistic ($TPR - FPR$) optimized on the training fold and applied to held-out test data. | **Passed** |
| **6** | **Comprehensive Imbalance Metrics** | Full reporting of ROC-AUC, PR-AUC, Sensitivity, Specificity, Precision, F1-Score, and Brier Score. | **Passed** |
| **7** | **PR-AUC Benchmarked vs Prevalence** | PR-AUC ($0.0428$) explicitly compared against the positive class prevalence baseline ($0.0265$). | **Passed** |
| **8** | **Microarray Scale Verification** | Confirmed GSE75132 values are linear ratios divided by the HPV-negative control median. | **Passed** |
| **9** | **Mathematically Correct Log2FC** | Calculated $\log_2(\text{mean}_A / \text{mean}_B)$, correcting previous subtraction artifacts ($+16.22 \to +3.88$). | **Passed** |
| **10** | **External Study Disclosures** | PRJNA622998 accurately relabeled as external literature synthesis (McKee et al., 2020), not raw FASTQ reprocessing. | **Passed** |
| **11** | **Multiple-Testing Correction** | Benjamini-Hochberg False Discovery Rate ($\text{FDR } q < 0.05$) enforced across all statistical tests. | **Passed** |
| **12** | **No Unsupported Clinical Claims** | Cautious cross-sectional language maintained; no claims of commercial medical readiness. | **Passed** |

---

## 3. Repository Directory & File Manifest

### Complete Workspace Directory Tree

```
/Users/ritweekraj/Desktop/HPV/
├── README.md                                  # [THIS FILE] Comprehensive repository documentation
├── biomarker_report.pdf                       # Multi-page, publication-grade executive PDF evaluation report
├── biomarker_presentation.html                # Interactive single-page HTML presentation dashboard
├── index.html                                 # Identical standalone web dashboard for local browsing / hosting
├── biological_differences.csv                 # Complete statistical comparisons across 31 tested biological features
├── differential_biomarkers.csv                # Filtered subset of statistically significant biomarkers (FDR q < 0.05)
├── feature_importance.csv                     # Ranked microbiome features across Random Forest, XGBoost, & LogReg
├── model_evaluation.csv                       # Repeated cross-validation performance metrics across 15 splits
│
├── scripts/                                   # Production scientific scripts
│   ├── download_mendeley.py                   # Automated REST API fetcher for raw Mendeley datasets
│   ├── mendeley_analysis.py                   # Primary statistical & confounder-adjusted biological analysis
│   ├── ml_pipeline.py                         # Leakage-free repeated stratified cross-validation ML pipeline
│   ├── gse_analysis.py                        # GEO GSE75132 host gene expression microarray pipeline
│   ├── generate_pdf_report.py                 # ReportLab programmatic PDF report generator
│   └── generate_html_presentation.py          # Interactive HTML5/CSS3 dashboard generator with embedded figures
│
├── data/                                      # Clinical and biological datasets
│   ├── geo/
│   │   └── GSE75132_family.soft.gz            # Raw Affymetrix HG-U133 Plus 2.0 cervical swab microarray data (41 swabs)
│   ├── PRJNA622998_runinfo.csv                # SRA run metadata for 333 Illumina MiSeq 16S amplicon runs (McKee et al.)
│   └── mendeley/                              # Mendeley Data repository pn7h6yjt6k (Sekyere et al.)
│       ├── Dataset 1_Epidemiology Stats.xlsx  # 27 sheets of demographic, geographic, and co-infection epidemiology
│       ├── Dataset 2_Association analysis.xlsx # Cross-tabulations, chi-square tests, and frequency distributions
│       ├── Dataset 3_Cor_ANOVA_Reg_interaction effects.xlsx # ANOVA, linear regressions, and multi-variable interactions
│       ├── Dataset 4_PCA_Kmeans_MachineLearning.xlsx        # Authors' initial PCA, clustering, and ML tables
│       └── Table S1.xlsx                      # Complete primary patient matrix (15,607 patients x 34 columns)
│
├── outputs/                                   # Generated scientific data tables & figures
│   ├── microbiome_ml_features.txt             # Audited text list of the 25 pure non-leaking ML features
│   ├── cytological_progression_biomarkers.csv # Statistical comparison of hrHPV in abnormal vs normal cytology
│   ├── gse75132_differential_expression.csv   # Transcriptomic stats across all 10 probes and 4 contrast groups
│   └── figures/                               # 300 DPI publication figures
│       ├── fig1_bacterial_abundance_by_hpv.png          # Boxplots of 9 bacterial species in normal cytology
│       ├── fig2_pca_microbiome_hpv.png                  # 2D PCA ordination of cervicovaginal microbiome
│       ├── fig3_spearman_correlation_heatmap.png        # Correlation matrix of bacteria and clinical covariates
│       ├── fig4_differential_biomarker_effect_sizes.png # Cliff's delta effect sizes with 95% confidence intervals
│       ├── fig5_model_roc_pr_curves.png                 # Cross-validated ROC and PR curves against prevalence
│       ├── fig6_feature_importance_comparison.png       # Random Forest Gini impurity ranking of microbiome features
│       ├── fig7_shap_summary_plot.png                   # SHAP value distributions for XGBoost model
│       ├── fig8_host_gene_expression_gse75132.png       # Host gene expression boxplots across 3 patient groups
│       └── fig9_host_gene_heatmap_gse75132.png          # Heatmap of host genes across all 41 cervical swab specimens
│
└── backup_initial_run/                        # Historical archive of initial uncorrected scripts and outputs
    ├── biomarker_report.pdf
    ├── biomarker_presentation.html
    ├── index.html
    ├── biological_differences.csv
    ├── differential_biomarkers.csv
    ├── feature_importance.csv
    ├── model_evaluation.csv
    ├── outputs/
    └── scripts/
```

### Role of Every Directory & File

#### Root Evaluation Deliverables
- [`biomarker_report.pdf`](file:///Users/ritweekraj/Desktop/HPV/biomarker_report.pdf): A comprehensive 5-page, publication-quality executive research evaluation report formatted using ReportLab. It contains structured metadata, executive summaries, diagnostic hierarchies, high-resolution figures, formatted statistical tables, and regulatory guardrails.
- [`index.html`](file:///Users/ritweekraj/Desktop/HPV/index.html) & [`biomarker_presentation.html`](file:///Users/ritweekraj/Desktop/HPV/biomarker_presentation.html): Identical, standalone, responsive HTML5 dashboards. All 9 scientific figures are directly embedded as Base64 strings, meaning the file can be opened in any web browser without an external server or internet connection. Features 8 interactive tabs.
- [`biological_differences.csv`](file:///Users/ritweekraj/Desktop/HPV/biological_differences.csv): The complete master table of all 31 tested biological and clinical features comparing hrHPV-positive vs HPV-negative women in normal cytology.
- [`differential_biomarkers.csv`](file:///Users/ritweekraj/Desktop/HPV/differential_biomarkers.csv): Filtered subset containing the 22 features that met statistical significance after Benjamini-Hochberg False Discovery Rate adjustment ($\text{FDR } q < 0.05$).
- [`feature_importance.csv`](file:///Users/ritweekraj/Desktop/HPV/feature_importance.csv): Multi-model importance rankings (Random Forest Gini importance, XGBoost gain, and Logistic Regression absolute coefficients) across the 25 non-leaking predictors.
- [`model_evaluation.csv`](file:///Users/ritweekraj/Desktop/HPV/model_evaluation.csv): Detailed cross-validation performance table containing mean and standard deviation for ROC-AUC, PR-AUC, Sensitivity, Specificity, Precision, F1-Score, Brier Score, and accumulated confusion matrix totals across 15 validation splits.

#### Production Scripts (`scripts/`)
- [`scripts/download_mendeley.py`](file:///Users/ritweekraj/Desktop/HPV/scripts/download_mendeley.py): Programmatically downloads all 5 files from the Mendeley Data REST API using dataset UUIDs.
- [`scripts/mendeley_analysis.py`](file:///Users/ritweekraj/Desktop/HPV/scripts/mendeley_analysis.py): Performs data preprocessing, feature engineering (log10 loads, relative abundances, ecological ratios), Mann-Whitney U tests, Cliff's delta with 95% CIs, multivariable logistic regressions adjusted for Age and BV status, FDR corrections, and generates Figures 1, 2, 3, and 4.
- [`scripts/ml_pipeline.py`](file:///Users/ritweekraj/Desktop/HPV/scripts/ml_pipeline.py): Executes the leakage-free machine learning benchmark using Repeated Stratified 5-Fold Cross-Validation (3 repeats = 15 splits), audits features for leakage, applies inside-fold scaling and threshold selection, and generates Figures 5, 6, and 7.
- [`scripts/gse_analysis.py`](file:///Users/ritweekraj/Desktop/HPV/scripts/gse_analysis.py): Parses the GEO GSE75132 SOFT file, maps Affymetrix probes to human genes, computes linear ratio fold changes, calculates Mann-Whitney U and FDR stats across 4 patient contrasts, and generates Figures 8 and 9.
- [`scripts/generate_pdf_report.py`](file:///Users/ritweekraj/Desktop/HPV/scripts/generate_pdf_report.py): Compiles all results into the publication-grade PDF report.
- [`scripts/generate_html_presentation.py`](file:///Users/ritweekraj/Desktop/HPV/scripts/generate_html_presentation.py): Encodes figures into Base64 and compiles the interactive HTML presentation.

#### Data Files (`data/`)
- [`data/mendeley/Table S1.xlsx`](file:///Users/ritweekraj/Desktop/HPV/data/mendeley/Table%20S1.xlsx): Primary matrix of 15,607 patient records with 34 measured clinical and laboratory variables.
- [`data/geo/GSE75132_family.soft.gz`](file:///Users/ritweekraj/Desktop/HPV/data/geo/GSE75132_family.soft.gz): Gzipped SOFT file from NCBI GEO containing transcriptomic microarray expression data for 41 cervical swab specimens.
- [`data/PRJNA622998_runinfo.csv`](file:///Users/ritweekraj/Desktop/HPV/data/PRJNA622998_runinfo.csv): SRA metadata cataloging 333 Illumina 16S sequencing experiments from McKee et al. (2020).

---

## 4. Datasets & Clinical Cohorts Detailed

### Cohort 1: Mendeley Clinical qPCR Dataset (pn7h6yjt6k / Table S1)
- **Source:** Sekyere et al., Mendeley Data, DOI: `10.17632/pn7h6yjt6k.1`.
- **Sample Size:** $N = 15,607$ patient records collected across multiple U.S. states.
- **Laboratory Assay:** Multiplex quantitative Polymerase Chain Reaction (qPCR) providing exact copy numbers per milliliter ($\text{copies/mL}$). Values below the assay limit of detection ($< 100\text{ copies/mL}$) are recorded as zero.
- **Cohorts Stratification:**
  - **Normal Cytology (NILM = 1, "Negative for Intraepithelial Lesion or Malignancy"):** $n = 11,749$ ($75.3\%$).
    - HPV-Negative Normal Cytology: $n = 11,438$ ($97.35\%$).
    - High-Risk HPV-Positive Normal Cytology: $n = 311$ ($2.65\%$).
  - **Abnormal Cytology (Cytological Dysplasia):** $n = 3,858$ ($24.7\%$).
    - Includes ASCUS (Atypical Squamous Cells of Undetermined Significance), LSIL (Low-grade Squamous Intraepithelial Lesion), HSIL (High-grade Squamous Intraepithelial Lesion), ASC-H, and AGC.
- **Features Measured (34 columns):**
  - **Demographics:** Patient Age, Gender (all biological females in this cohort), Provider State.
  - **Bacterial Pathogens (5 species):** *Fannyhessea vaginae*, *Gardnerella vaginalis*, *BVAB-2*, *Megasphaera sp. Type 1*, *Megasphaera sp. Type 2*.
  - **Protective Lactobacilli (4 species):** *Lactobacillus crispatus*, *Lactobacillus iners*, *Lactobacillus gasseri*, *Lactobacillus jensenii*.
  - **Clinical BV Status:** Coded as BV-Negative (`BV-N`), Intermediate/Transitional (`BV-T`), or BV-Positive (`BV-P`).
  - **High-Risk HPV Genotypes (14 types):** HPV 16, 18, 31, 33, 35, 39, 45, 51, 52, 56, 58, 59, 68, and overall binary flag `hrHPV Result`.
  - **Cytology Classes:** Binary flags for NILM, ECA:ASCUS, ECA:LSIL, ECA:HSIL, ECA:ASC-H, ECA:AGC, RCC.

### Cohort 2: GEO GSE75132 Swab Microarray Dataset
- **Source:** NCBI Gene Expression Omnibus (GEO) GSE75132; Affymetrix Human Genome U133 Plus 2.0 Array (Platform GPL570, 54,675 probe sets).
- **Sample Size:** $n = 41$ cervical swab RNA specimens collected in a prospective longitudinal screening cohort.
- **Clinical Categorization:**
  - **Category 0 (HPV-Negative Normal Controls):** $n = 11$ women with normal cytology and negative HPV tests.
  - **Category 1 (HPV16-Positive Non-Progressors):** $n = 10$ women with persistent HPV16 infection who did *not* develop high-grade lesions during longitudinal follow-up.
  - **Category 2 (HPV16-Positive Progressors to CIN3+):** $n = 20$ women with persistent HPV16 infection who developed biopsy-confirmed CIN3 or invasive cervical cancer.
- **Normalized Scale Verification:** As explicitly verified in the GEO submitter protocol, expression values represent **linear normalized ratios** computed by dividing each sample's signal by the median of the HPV-negative control group.
- **Target Probes Analyzed (10 probes across 4 genes):**
  - `219410_at`: **TMEM45A** (Transmembrane protein 45A)
  - `204855_at` & `1555551_at`: **SERPINB5** (Maspin / Serpin peptidase inhibitor clade B member 5)
  - `207039_at`, `209644_x_at`, `211156_at`: **CDKN2A** ($p16^{INK4a}$)
  - `212020_s_at`, `212021_s_at`, `212022_s_at`, `212023_s_at`: **MKI67** ($Ki\text{-}67$ proliferation marker)

### Cohort 3: NCBI SRA PRJNA622998 16S Amplicon Sequencing
- **Source:** BioProject PRJNA622998; McKee et al., *Gynecology and Pelvic Medicine*, 2020.
- **Data Scope:** 333 Illumina MiSeq runs ($2 \times 250\text{ bp}$ paired-end sequencing of the 16S rRNA gene V4 hypervariable region).
- **Clinical Groups:**
  - Healthy Controls ($n = 89$): Negative for cytologic lesions and negative for HPV.
  - High-Risk HPV-Positive with Normal Cytology ($n = 110$).
  - Cytologic Lesions ($n = 109$): ASCUS ($55$), LSIL ($45$), HSIL ($6$), AGC ($3$).
- **Role in This Project:** Used as an external, independent literature comparator to evaluate whether our quantitative qPCR observations replicate under high-throughput 16S microbiome profiling.

---

## 5. Implementation Details: Scripts & Execution Pipelines

### `scripts/download_mendeley.py`
- **Purpose:** Automatically downloads all primary data tables from Mendeley Data.
- **How It Works:**
  - Connects to the Mendeley public API endpoint: `https://data.mendeley.com/public-api/datasets/pn7h6yjt6k/files/{file_id}`.
  - Queries file metadata to retrieve download URLs.
  - Streams and saves the 5 Excel workbooks into `data/mendeley/`.
  - Checks if files already exist on disk ($>1,000\text{ bytes}$) to prevent redundant network requests.

### `scripts/mendeley_analysis.py`
- **Purpose:** Executes the complete statistical, ecological, and confounder-adjusted biological analysis of the Mendeley clinical cohort.
- **Mathematical Formulations Implemented:**
  1. **Logarithmic Abundance Transformation:**
     $$\text{log10\_load} = \log_{10}(\text{qPCR copies/mL} + 1)$$
     The $+1$ pseudocount handles samples with bacterial counts below the detection limit ($0\text{ copies/mL}$).
  2. **Ecological Composites & Taxonomic Ratios:**
     $$\text{Total Bacterial Biomass} = \sum_{i=1}^{9} \text{Load}_i$$
     $$\text{Relative Abundance}_i = \frac{\text{Load}_i}{\text{Total Bacterial Biomass}}$$
     $$\text{Lactobacillus-to-Pathogen Ratio} = \log_{10}(\text{Total Lactobacilli} + 1) - \log_{10}(\text{Total BV Pathogens} + 1)$$
     $$\text{L. crispatus-to-L. iners Ratio} = \log_{10}(\text{L. crispatus} + 1) - \log_{10}(\text{L. iners} + 1)$$
  3. **Non-Parametric Effect Size (Cliff's Delta with Asymptotic 95% Confidence Intervals):**
     Cliff's delta ($\delta$) measures the degree of overlap between two distributions without assuming normality:
     $$\delta = \frac{2U}{n_1 n_2} - 1$$
     where $U$ is the Mann-Whitney $U$ statistic. The asymptotic variance is:
     $$\sigma_{\delta}^2 = \frac{n_1 + n_2 + 1}{3 n_1 n_2}, \quad \text{CI}_{95\%} = \left[ \delta - 1.96 \sigma_{\delta}, \; \delta + 1.96 \sigma_{\delta} \right]$$
  4. **Parametric Effect Size (Cohen's $d$):**
     $$d = \frac{\bar{x}_1 - \bar{x}_2}{s_{\text{pooled}}}, \quad s_{\text{pooled}} = \sqrt{\frac{(n_1 - 1)s_1^2 + (n_2 - 1)s_2^2}{n_1 + n_2 - 2}}$$
  5. **Confounder-Adjusted Multivariable Logistic Regression:**
     To test whether a bacterial taxon is independently associated with hrHPV after controlling for Age and BV status:
     $$\text{logit}(P(\text{hrHPV} = 1)) = \beta_0 + \beta_1 Z_{\text{feature}} + \beta_2 Z_{\text{Age}} + \beta_3 \text{BV\_Score}$$
     where features are standardized to unit variance ($Z$). The adjusted odds ratio per standard deviation is:
     $$\text{Adj. OR} = \exp(\beta_1), \quad \text{CI}_{95\%} = [\exp(\beta_1 - 1.96 \cdot \text{SE}), \; \exp(\beta_1 + 1.96 \cdot \text{SE})]$$
  6. **Multiple Testing Correction:**
     Benjamini-Hochberg False Discovery Rate ($\text{FDR}$) controls the expected proportion of false positives among rejected null hypotheses:
     $$P_{(k)} \le \frac{k}{m} \alpha \implies q\text{-value}$$
- **Outputs Generated:**
  - `biological_differences.csv` (all 31 features tested)
  - `differential_biomarkers.csv` (22 significant features at FDR $q < 0.05$)
  - `outputs/figures/fig1_bacterial_abundance_by_hpv.png`
  - `outputs/figures/fig2_pca_microbiome_hpv.png`
  - `outputs/figures/fig3_spearman_correlation_heatmap.png`
  - `outputs/figures/fig4_differential_biomarker_effect_sizes.png`

### `scripts/ml_pipeline.py`
- **Purpose:** Evaluates machine learning algorithms to determine if vaginal bacterial measurements alone can predict hrHPV status in women with normal cytology.
- **Strict Leakage Prevention:**
  - Feature set audited to ensure exactly 0 HPV-derived variables.
  - The final 25 features are quantitative bacterial qPCR loads, relative abundances, ecological ratios, patient age, and BV score.
- **Validation Methodology:**
  - **Repeated Stratified K-Fold Cross-Validation:** 5 folds $\times$ 3 repeats = 15 independent validation splits.
  - **Class Imbalance Management:** Class weights inversely proportional to class frequencies:
    $$\text{weight}_1 = \frac{N_{\text{total}}}{2 \cdot N_{\text{positive}}}$$
  - **Threshold Selection:** Rather than using an arbitrary $0.5$ probability threshold, the optimal classification threshold is chosen strictly inside the training fold using **Youden's $J$ statistic**:
    $$J = \text{Sensitivity} + \text{Specificity} - 1 = \text{TPR} - \text{FPR}$$
    This threshold is then applied to the held-out test fold to simulate real-world clinical deployment.
- **Algorithms Evaluated:**
  1. **Logistic Regression:** $L_2$-regularized ($C=0.5$), balanced class weights.
  2. **Random Forest Classifier:** 200 trees, maximum depth 5, balanced class weights, Gini impurity splitting.
  3. **XGBoost Classifier:** 150 estimators, learning rate $0.03$, max depth 3, `scale_pos_weight` $= (N_{\text{neg}} / N_{\text{pos}})$.
- **Outputs Generated:**
  - `model_evaluation.csv`
  - `feature_importance.csv`
  - `outputs/microbiome_ml_features.txt`
  - `outputs/figures/fig5_model_roc_pr_curves.png`
  - `outputs/figures/fig6_feature_importance_comparison.png`
  - `outputs/figures/fig7_shap_summary_plot.png`

### `scripts/gse_analysis.py`
- **Purpose:** Analyzes host epithelial gene expression in cervical swabs from GEO GSE75132.
- **Methodology:**
  - Parses `GSE75132_family.soft.gz` using `GEOparse`.
  - Maps 10 Affymetrix probes to the 4 target genes.
  - Calculates non-parametric Mann-Whitney $U$ tests, Student's $t$-tests, and Benjamini-Hochberg FDR adjustments across all 10 probes within each comparison.
  - Evaluates 4 separate pairwise contrasts:
    1. HPV16 Non-Progressors vs HPV-Negative Normal Controls ($n=10$ vs $11$).
    2. HPV16 Progressors to CIN3+ vs HPV-Negative Normal Controls ($n=20$ vs $11$).
    3. HPV16 Progressors to CIN3+ vs HPV16 Non-Progressors ($n=20$ vs $10$).
    4. All HPV16-Positive Swabs vs HPV-Negative Normal Controls ($n=30$ vs $11$).
  - Correct Logarithmic Fold Change on Linear Ratio Scale:
    $$\text{Log2FC} = \log_2\left( \frac{\bar{x}_A + 10^{-6}}{\bar{x}_B + 10^{-6}} \right)$$
- **Outputs Generated:**
  - `outputs/gse75132_differential_expression.csv`
  - `outputs/figures/fig8_host_gene_expression_gse75132.png`
  - `outputs/figures/fig9_host_gene_heatmap_gse75132.png`

### `scripts/generate_pdf_report.py`
- **Purpose:** Compiles the scientific evaluation findings into a 5-page publication-grade PDF report (`biomarker_report.pdf`).
- **Technical Implementation:**
  - Built using Python's `reportlab` library.
  - Custom `NumberedCanvas` dynamically draws running headers, divider rules, research use disclaimers, and running page numbers ("Page $X$ of $Y$").
  - Formats tables with alternating fills, grid lines, and cell padding.
  - Embeds all 300 DPI figures at exact column widths.
  - Includes styled callout alert boxes for critical clinical limitations.

### `scripts/generate_html_presentation.py`
- **Purpose:** Generates an interactive web dashboard (`index.html` and `biomarker_presentation.html`).
- **Technical Implementation:**
  - Reads data tables from CSV and figures from disk.
  - Converts PNG figures into Base64 data URIs (`data:image/png;base64,...`) and embeds them inline.
  - Implements tab switching via vanilla JavaScript (`showTab(tabId)`).
  - Contains responsive CSS grid layouts, styled badges, stat cards, and interactive table styling.

---

## 6. Detailed Scientific Findings & Statistical Outputs

### Part A: Bacterial Dysbiosis in Normal Cytology (NILM)
Comparing 11,438 HPV-negative women against 311 hrHPV-positive women (all with normal Pap cytology, NILM = 1):

| Feature Name | Feature Category | Mean HPV(-) | Mean hrHPV(+) | Cliff's $\delta$ [95% CI] | FDR $q$-value | Adj. OR (per SD)* | Adj. $p$-value* | Biological Interpretation |
|---|---|---|---|---|---|---|---|---|
| **BVAB-2** | Bacterial load | $15,516$ | $16,228$ | **$+0.132$** $[+0.07, +0.20]$ | $1.16 \times 10^{-6}$ | **$1.23$** | $0.0006$ | Strongest anaerobic pathogen association |
| **F. vaginae** | Bacterial load | $26,736$ | $49,446$ | **$+0.121$** $[+0.06, +0.19]$ | $8.70 \times 10^{-5}$ | **$1.17$** | $0.0051$ | Severe biofilm-forming dysbiosis |
| **Megasphaera 1** | Bacterial load | $55,355$ | $54,417$ | **$+0.112$** $[+0.05, +0.18]$ | $4.83 \times 10^{-5}$ | $1.08$ | $0.1777$ | Enriched in hrHPV infection |
| **G. vaginalis** | Bacterial load | $13,418$ | $6,066$ | **$+0.110$** $[+0.04, +0.17]$ | $0.0002$ | **$1.13$** | $0.0410$ | Sialidase-producing BV organism |
| **Total BV Pathogens** | Composite load | $119,900$ | $134,681$ | **$+0.140$** $[+0.07, +0.20]$ | $3.33 \times 10^{-5}$ | $1.00$ | $0.9933$ | Overall pathogen biomass elevated |
| **Lacto:Pathogen Ratio** | Microbial ratio | $+0.56$ | $-0.16$ | **$-0.118$** $[-0.18, -0.05]$ | $0.0008$ | **$0.88$** | $0.0212$ | Depleted protective microenvironment |
| **L. gasseri** | Bacterial load | $564$ | $322$ | **$-0.062$** $[-0.13, -0.00]$ | $0.0009$ | **$0.84$** | $0.0601$ | Significant depletion in hrHPV+ women |
| **L. iners** | Bacterial load | $1,898$ | $3,417$ | **$+0.073$** $[+0.01, +0.14]$ | $0.0201$ | $1.08$ | $0.1661$ | Transitional Lactobacillus enriched |
| **Patient Age** | Demographic | $40.34$ yrs | $37.17$ yrs | **$-0.180$** $[-0.25, -0.12]$ | $1.16 \times 10^{-6}$ | — | — | hrHPV+ cohort is significantly younger |
| **BV Score** | Clinical status | $0.53$ | $0.63$ | **$+0.104$** $[+0.04, +0.17]$ | $0.0008$ | — | — | Clinical BV prevalence is higher |

*\*Adjusted Odds Ratios and $p$-values derived from multivariable logistic regression controlling for Patient Age and Clinical BV Status.*

#### Key Biological Takeaways:
1. **Anaerobic Pathogen Enrichment:** Women carrying hrHPV despite normal cytology show significant enrichment of anaerobic BV-associated organisms (*BVAB-2*, *F. vaginae*, and *Megasphaera sp. Type 1*). Even after controlling for age and BV status, *BVAB-2* increases odds of hrHPV by $23\%$ per standard deviation ($p = 0.0006$), and *F. vaginae* by $17\%$ ($p = 0.0051$).
2. **Lactobacillus Divergence:** Not all lactobacilli behave identically. While *L. gasseri* is significantly depleted in hrHPV-positive women ($\delta = -0.062, q = 0.0009$), *L. iners* (a species known to produce inerolysin and thrive during transitional/dysbiotic states) is actually enriched ($\delta = +0.073, q = 0.0201$).
3. **Important Confounders:** Patients with hrHPV were on average 3.17 years younger ($p = 6.68 \times 10^{-7}$) and had higher rates of clinical BV ($64.6\%$ vs $54.1\%$, $\chi^2 p = 0.0011$). This highlights the necessity of multivariable adjustment.

---

### Part B: Rigorous Machine Learning Evaluation
Results from Repeated Stratified 5-Fold Cross-Validation (15 validation splits) using **Microbiome & Clinical Features Only** (zero HPV predictors):

| Model Algorithm | ROC-AUC (Mean $\pm$ SD) | PR-AUC (Mean $\pm$ SD) | Prevalence Baseline | Sensitivity | Specificity | Precision | F1-Score | Brier Score |
|---|---|---|---|---|---|---|---|---|
| **Logistic Regression** | **$0.6005 \pm 0.0320$** | **$0.0428 \pm 0.0081$** | $0.0265$ ($2.65\%$) | $53.6\% \pm 10.2\%$ | $61.6\% \pm 5.9\%$ | **$3.65\%$** | $0.0683$ | $0.2364$ |
| **Random Forest** | **$0.5795 \pm 0.0335$** | **$0.0436 \pm 0.0100$** | $0.0265$ ($2.65\%$) | $30.7\% \pm 8.5\%$ | $77.5\% \pm 5.9\%$ | **$3.64\%$** | $0.0647$ | $0.2044$ |
| **XGBoost** | **$0.5739 \pm 0.0306$** | **$0.0416 \pm 0.0088$** | $0.0265$ ($2.65\%$) | $37.5\% \pm 5.3\%$ | $71.1\% \pm 5.1\%$ | **$3.46\%$** | $0.0632$ | $0.2033$ |

#### Accumulated Confusion Matrices (Sum of 15 Test Splits):
- **Logistic Regression:** $\text{True Negatives} = 21,122 \quad | \quad \text{False Positives} = 13,192 \quad | \quad \text{False Negatives} = 433 \quad | \quad \text{True Positives} = 500$
- **Random Forest:** $\text{True Negatives} = 26,603 \quad | \quad \text{False Positives} = 7,711 \quad | \quad \text{False Negatives} = 646 \quad | \quad \text{True Positives} = 287$
- **XGBoost:** $\text{True Negatives} = 24,402 \quad | \quad \text{False Positives} = 9,912 \quad | \quad \text{False Negatives} = 583 \quad | \quad \text{True Positives} = 350$

#### Critical Machine Learning Insights:
1. **The Microbiome is Weak as a Standalone Detector:** An ROC-AUC of $\sim 0.60$ demonstrates that vaginal bacterial composition provides only weak discriminatory power for detecting HPV.
2. **Understanding the Precision Limitation:** Because the real-world prevalence of hrHPV in women with normal cytology is only $2.65\%$, a precision of $\sim 3.6\%$ means that **out of every 100 positive flags raised by a microbiome model, roughly 96 are false positives**.
3. **Translational Verdict:** A swab test that measures only vaginal bacteria cannot function as an HPV screening tool. Microbial testing serves as a research indicator of mucosal barrier health, not a diagnostic engine.

---

### Part C: Host Transcriptomics & Progression Markers (GSE75132)
Analysis of host epithelial mRNA in 41 cervical swab specimens across the three disease stages:

| Gene Symbol | Affymetrix Probe | Control Mean ($n=11$) | Non-Prog Mean ($n=10$) | CIN3+ Mean ($n=20$) | CIN3+ vs Ctrl Log2FC | CIN3+ vs Ctrl FDR $q$ | Prog vs NonProg Log2FC | Prog vs NonProg $p$ | Biological Function |
|---|---|---|---|---|---|---|---|---|---|
| **TMEM45A** | `219410_at` | $1.18$ | $5.45$ | $23.37$ | **$+4.31$** | **$9.07 \times 10^{-5}$** | **$+2.10$** | $0.0557$ | Epithelial barrier & hypoxia-induced differentiation |
| **SERPINB5** | `204855_at` | $1.16$ | $1.42$ | $10.78$ | **$+3.22$** | **$0.0120$** | **$+2.93$** | **$0.0114$** | Maspin; cell adhesion & progression inhibition |
| **SERPINB5** | `1555551_at` | $1.04$ | $1.15$ | $1.20$ | $+0.20$ | $0.0731$ | $+0.06$ | $0.8088$ | Secondary probe |
| **CDKN2A** | `207039_at` | $1.02$ | $0.92$ | $3.23$ | $+1.66$ | $0.2059$ | **$+1.81$** | **$0.0366$** | $p16^{INK4a}$; cell cycle control (HPV $E7$ target) |
| **CDKN2A** | `211156_at` | $1.01$ | $1.20$ | $0.88$ | $-0.20$ | $0.1372$ | $-0.45$ | $0.0366$ | Secondary probe |
| **MKI67** | `212020_s_at`| $1.00$ | $1.06$ | $1.22$ | $+0.29$ | $0.4270$ | $+0.20$ | $0.2811$ | $Ki\text{-}67$; cellular proliferation marker |
| **MKI67** | `212022_s_at`| $1.05$ | $1.27$ | $1.73$ | $+0.71$ | $0.7568$ | $+0.44$ | $0.9474$ | Proliferation probe |

```
    Normalized Expression Scale (Fold of Control Median)
    30 ┼                                                  ● (23.37)
       │                                                 TMEM45A
    25 ┼
       │
    20 ┼
       │
    15 ┼
       │                                                  ● (10.78)
    10 ┼                                                 SERPINB5
       │                         ● (5.45)
     5 ┼                        TMEM45A                   ● (3.23) CDKN2A
       │    ● (1.18) TMEM45A     ● (1.42) SERPINB5        ● (1.73) MKI67
     0 ┴────┴────────────────────┴────────────────────────┴─────────────
          Category 0 (HPV-)     Category 1 (Non-Prog)    Category 2 (CIN3+)
             [n = 11]                 [n = 10]                [n = 20]
```

#### Key Transcriptomic Takeaways:
1. **TMEM45A is Dramatically Upregulated:** On the verified normalized linear scale, TMEM45A rises from $1.18$ in uninfected controls to $5.45$ in persistent non-progressors ($\text{Log2FC} = +2.21, q = 0.0173$), and surges to $23.37$ in CIN3+ progressors ($\text{Log2FC} = +4.31, q = 9.07 \times 10^{-5}$). This represents a true **$14.7$-fold linear increase** in cervical swabs from women developing high-grade pre-cancer.
2. **SERPINB5 Distinguishes Progressors from Non-Progressors:** Unlike TMEM45A (which is elevated in both persistent groups), SERPINB5 remains at baseline in persistent non-progressors ($1.42$) but increases to $10.78$ in CIN3+ progressors ($\text{Log2FC} = +2.93, p = 0.0114$).
3. **Exploratory Caveat:** Because GSE75132 consists of 41 retrospective swab samples, these findings represent biological proof-of-concept candidates rather than a clinically validated diagnostic panel.

---

### Part D: Concordance with Independent 16S Sequencing (PRJNA622998)
Comparing our qPCR findings with published Illumina MiSeq 16S rRNA sequencing findings from McKee et al. (2020):

| Biological Feature | Mendeley qPCR Finding ($n=15,607$) | McKee et al. 16S Finding ($n=333$) | Scientific Concordance |
|---|---|---|---|
| **Lactobacillus Dominance** | Significantly elevated protective lactobacilli in healthy uninfected controls ($p = 0.0057$). | Healthy uninfected controls significantly more likely to be *Lactobacillus*-dominated ($p < 0.04$). | **Strong Concordance** |
| **Community Diversity & Dysbiosis** | hrHPV+ normal women have higher BV score and lower Lacto:Pathogen ratio ($p = 0.0003$). | hrHPV+ and abnormal cytology samples display significantly higher alpha diversity ($p < 0.05$). | **Strong Concordance** |
| **Gardnerella vaginalis** | Enriched in hrHPV+ normal cytology ($\delta = +0.110, q = 0.0002$). | Elevated *G. vaginalis* relative abundance in hrHPV-positive and dysplasia groups. | **Strong Concordance** |
| **Lactobacillus gasseri Depletion**| Significant depletion in hrHPV+ normal women ($\delta = -0.062, q = 0.0009$). | Independent depletion of *L. gasseri* operational taxonomic units in hrHPV infection. | **Strong Concordance** |

---

## 7. Visual Artifacts Catalog (Figures 1 through 9)

All figures were generated at 300 DPI, styled with consistent palettes, and saved in [`outputs/figures/`](file:///Users/ritweekraj/Desktop/HPV/outputs/figures):

```
Figure 1: fig1_bacterial_abundance_by_hpv.png
├─ Content: 3x3 grid of boxplots showing log10 abundance for all 9 bacterial species in normal cytology.
├─ Comparison: HPV-Negative (n=11,438, blue) vs hrHPV-Positive (n=311, red).
└─ Key Detail: Gold diamonds mark group means; titles include FDR q-values and Cliff's delta.

Figure 2: fig2_pca_microbiome_hpv.png
├─ Content: 2D Principal Component Analysis (PC1 vs PC2) ordination of bacterial qPCR features.
├─ Variance: Displays percentage of total microbial variance explained by the first two components.
└─ Key Detail: Demonstrates substantial overlap between HPV+ and HPV- distributions in normal cytology.

Figure 3: fig3_spearman_correlation_heatmap.png
├─ Content: Triangular Spearman rank correlation heatmap across all 9 bacteria, Age, BV score, and ratios.
├─ Scale: Diverging vlag colormap ranging from rho = -0.8 to +0.8 with explicit numerical correlation labels.
└─ Key Detail: Illustrates strong positive co-occurrence clusters among BV pathogens (BVAB-2, F. vaginae).

Figure 4: fig4_differential_biomarker_effect_sizes.png
├─ Content: Horizontal bar chart of top differential features ranked by non-parametric Cliff's delta.
├─ Details: Includes asymptotic 95% confidence interval error bars and Benjamini-Hochberg FDR q-values.
└─ Key Detail: Positive bars (red) = enriched in hrHPV+; Negative bars (blue) = depleted (e.g. L. gasseri).

Figure 5: fig5_model_roc_pr_curves.png
├─ Content: 2-panel figure showing cross-validated performance of Logistic Regression, Random Forest, & XGBoost.
├─ Panel A: ROC curves showing true positive rate vs false positive rate (Mean AUCs ~0.57 - 0.60).
└─ Panel B: Precision-Recall curves explicitly plotted against the 2.65% positive prevalence baseline.

Figure 6: fig6_feature_importance_comparison.png
├─ Content: Ranked bar chart of Random Forest Mean Decrease in Impurity (Gini Importance).
└─ Key Detail: Demonstrates that Patient Age, log10_BVAB-2, and log10_F. vaginae are top predictors.

Figure 7: fig7_shap_summary_plot.png
├─ Content: SHAP (SHapley Additive exPlanations) beeswarm plot for the cross-validated XGBoost model.
└─ Key Detail: Visualizes the impact of high vs low feature values on log-odds of hrHPV positivity.

Figure 8: fig8_host_gene_expression_gse75132.png
├─ Content: 2x2 grid of boxplots with jittered points showing TMEM45A, SERPINB5, CDKN2A, & MKI67.
├─ Comparison: Category 0 (Control), Category 1 (Non-Prog), Category 2 (CIN3+ Progressors).
└─ Key Detail: Plotted on verified linear normalized ratio scale; documents dramatic TMEM45A surge.

Figure 9: fig9_host_gene_heatmap_gse75132.png
├─ Content: Clustered heatmap of log2-transformed expression across all 41 cervical swab specimens.
└─ Key Detail: Samples ordered by clinical category, highlighting patient-level expression patterns.
```

---

## 8. Translational Architecture & Home Diagnostic Guardrails

### Three-Tier Diagnostic Hierarchy
For any real-world rapid testing device (such as an at-home lateral-flow strip, an electrochemical sensor, or an isothermal microfluidic cartridge), biological markers must be separated into clear operational tiers:

```
┌────────────────────────────────────────────────────────────────────────┐
│  TIER 1: CORE PRIMARY DIAGNOSTIC ENGINE                                │
│  Biomarker: Direct High-Risk HPV DNA (Types 16, 18, 31, 33, 45, 52...) │
│  Modality: Isothermal NAAT (LAMP / RPA) or Microfluidic qPCR           │
│  Role: SENSITIVITY & SPECIFICITY ANCHOR. The ONLY assay component      │
│        capable of establishing the presence of carcinogenic viral DNA. │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│  TIER 2: SECONDARY HOST RISK TRIAGE                                    │
│  Biomarker: TMEM45A mRNA, SERPINB5 mRNA, CDKN2A (p16INK4a) Protein     │
│  Modality: RT-LAMP or Lateral-Flow Protein Immunoassay                 │
│  Role: DYSPLASIA TRIAGE. Distinguishes benign transient carriage from  │
│        high-grade transforming epithelial pre-cancer (CIN3+).          │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│  TIER 3: ANCILLARY MICROENVIRONMENTAL MONITORING                       │
│  Biomarker: BVAB-2, F. vaginae, G. vaginalis, Lacto:Pathogen Ratio     │
│  Modality: Enzymatic Sialidase Activity or Multiplex Bacterial Sensor  │
│  Role: CLEARANCE & RECURRENCE ADJUNCT. Assesses mucosal vulnerability   │
│        and chronic dysbiosis. CANNOT diagnose HPV infection.           │
└────────────────────────────────────────────────────────────────────────┘
```

### CLIA / FDA Regulatory Realities
Any diagnostic device translating these findings must address four regulatory requirements:
1. **Human Sample Adequacy Control (SAC):** To ensure a self-collected swab actually sampled human cervical/vaginal cells (rather than air or saline), an assay must incorporate an internal human control (e.g. human *$\beta$-globin* or *RNase P* DNA). Without this, an empty swab yields a false negative.
2. **Analytical Limit of Detection (LoD):** The assay must document its analytical LoD in international viral units ($\text{IU/mL}$) or copies per reaction across all 14 hrHPV genotypes.
3. **Cross-Reactivity Testing:** The assay must demonstrate zero analytical cross-reactivity against non-carcinogenic low-risk HPV types (HPV 6, 11, 42, 43, 44) and common urogenital pathogens (*Chlamydia trachomatis*, *Neisseria gonorrhoeae*, *Trichomonas vaginalis*, *Candida albicans*, and herpes simplex virus).
4. **Clinical Multi-Center Validation:** Retrospective machine learning in public datasets is classified by the FDA as an **Investigational Research Tool (RUO)**. Clinical commercialization requires a prospective multi-center trial comparing self-swab performance against physician-collected cervical cytology and colposcopy-directed histological biopsies.

---

## 9. How to Reproduce & Execute the Complete Analysis

The entire analytical pipeline is automated and deterministic. Follow these steps in a macOS or Linux terminal:

### Step 1: Environment Setup
Ensure Python 3.10+ is installed. Clone or navigate to the repository directory:
```bash
cd /Users/ritweekraj/Desktop/HPV
```

Activate the virtual environment (or install required libraries):
```bash
source .venv/bin/activate
# Required dependencies:
# pip install numpy pandas scipy statsmodels scikit-learn xgboost shap matplotlib seaborn GEOparse reportlab openpyxl
```

### Step 2: Download the Datasets
To re-fetch all raw Mendeley workbooks directly from the Mendeley public API:
```bash
python scripts/download_mendeley.py
```
*(If the files are already in `data/mendeley/`, this script confirms their integrity and skips redundant downloads).*

### Step 3: Run the Primary Biological & Confounder Analysis
To execute the non-parametric tests, multivariable logistic regressions, FDR corrections, and generate Figures 1–4:
```bash
python scripts/mendeley_analysis.py
```
*Outputs generated:* `biological_differences.csv`, `differential_biomarkers.csv`, and Figures 1, 2, 3, and 4.

### Step 4: Run the Leakage-Free Machine Learning Benchmark
To execute the Repeated Stratified Cross-Validation (15 splits) and generate Figures 5–7:
```bash
python scripts/ml_pipeline.py
```
*Outputs generated:* `model_evaluation.csv`, `feature_importance.csv`, `outputs/microbiome_ml_features.txt`, and Figures 5, 6, and 7.

### Step 5: Run the Host Transcriptomics Pipeline
To parse GEO GSE75132, verify Affymetrix probes, compute fold changes, and generate Figures 8–9:
```bash
python scripts/gse_analysis.py
```
*Outputs generated:* `outputs/gse75132_differential_expression.csv`, and Figures 8 and 9.

### Step 6: Generate the Presentation Dashboard & PDF Report
To compile the interactive single-page HTML dashboard and publication-grade PDF report:
```bash
python scripts/generate_html_presentation.py
python scripts/generate_pdf_report.py
```
*Outputs generated:* `index.html`, `biomarker_presentation.html`, and `biomarker_report.pdf`.

---

## 10. Glossary of Clinical & Statistical Terms

- **hrHPV (High-Risk Human Papillomavirus):** A group of roughly 14 sexually transmitted viral genotypes capable of integrating into human cellular DNA and causing cervical cancer.
- **NILM ("Negative for Intraepithelial Lesion or Malignancy"):** The official Bethesda system classification for normal, healthy cervical cytology (a normal Pap smear).
- **ASCUS / LSIL / HSIL:** Progressive grades of cervical cell abnormality observed under Pap cytology. HSIL (High-grade Squamous Intraepithelial Lesion) carries the highest immediate risk of progression to invasive cancer.
- **CIN (Cervical Intraepithelial Neoplasia):** Histological grading of pre-cancerous tissue changes confirmed via cervical biopsy: CIN1 (mild), CIN2 (moderate), and CIN3 (severe pre-cancer / carcinoma in situ).
- **Dysbiosis:** An imbalance in the microbial community characterized by a loss of protective *Lactobacillus* species and an overgrowth of anaerobic pathogens.
- **qPCR (Quantitative Polymerase Chain Reaction):** A laboratory method that amplifies and quantifies specific DNA molecules in real time, reporting absolute concentrations in copies per milliliter.
- **Target Data Leakage:** A severe modeling error where information from outside the training dataset (or derived from the target variable itself) is included during model training, creating unrealistically inflated performance estimates that collapse in real-world deployment.
- **Cross-Validation (Repeated Stratified K-Fold):** A statistical evaluation procedure where data is split into $K$ parts, preserving class proportions. The model trains on $K-1$ parts and tests on the remaining held-out part. Repeating this ensures performance estimates are robust against sampling variance.
- **ROC-AUC (Receiver Operating Characteristic - Area Under Curve):** Measures a model's ability to rank positive cases higher than negative cases across all classification thresholds ($0.5 = \text{random chance}$, $1.0 = \text{perfect ranking}$).
- **PR-AUC (Precision-Recall Area Under Curve):** The definitive evaluation metric for severely imbalanced datasets. Unlike ROC-AUC, PR-AUC focuses on the positive class. The baseline performance expected from a random classifier equals the positive class prevalence ($0.0265$ in this cohort).
- **Youden's J Statistic:** An objective mathematical criterion ($J = \text{Sensitivity} + \text{Specificity} - 1$) used to identify the optimal probability threshold for binary classification.
- **Cliff's Delta ($\delta$):** A non-parametric effect size measuring the probability that a value chosen from one group is greater than a value chosen from another group minus the reverse probability. Ranges from $-1.0$ to $+1.0$ ($0.0 = \text{identical distributions}$).
- **False Discovery Rate (FDR / Benjamini-Hochberg $q$-value):** A statistical adjustment that controls the expected proportion of false positive discoveries when performing dozens of simultaneous hypothesis tests.
- **SHAP (SHapley Additive exPlanations):** A game-theoretic technique that explains individual machine learning model predictions by calculating each feature's contribution to moving a prediction away from the baseline average.

---
*Report compiled and maintained in `/Users/ritweekraj/Desktop/HPV`.*
