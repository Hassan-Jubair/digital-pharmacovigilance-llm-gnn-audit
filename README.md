# Digital Pharmacovigilance Methodological Audit (LLM Extraction and Temporal GNN Forecasting)

[![Target Journal](https://img.shields.io/badge/Journal-PLOS%20Digital%20Health-blue)](https://journals.plos.org/digitalhealth/)
[![Code License: MIT](https://img.shields.io/badge/Code%20License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Data License: CC BY 4.0](https://img.shields.io/badge/Data%20License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

Privacy-preserving replication archive for **"Auditing Zero-Shot Large Language Model and Graph Neural Network Pipelines for Digital Pharmacovigilance of Grey-Market Peptides and Metabolic Therapeutics"** (Tajul Islam Rafi, Hossain Ahmed, Md Abu Kawsar Prodhan Hemal, and Hassan Jubair).

## Core Empirical Findings

Across 81,223 Reddit records (March 2014 to August 2026; 26,778 named-account dense records across 15,768 accounts and seven active communities), this package documents and reproduces four methodological audits:

1. **Human-LLM Annotation Anchoring Bias:** Contrasting our unblinded Round 1 review sheet (`N = 200`) against a fresh, strictly blinded dual-annotator benchmark (`N = 120`: 60 Metabolic [30 Retatrutide, 30 Approved GLP-1/GIP] and 60 BPC-157; blinded human binary kappa = 0.9400, symptom kappa = 0.8530) reveals +20.1 percentage points of binary F1 inflation (93.1% vs. 73.0%) and +42.0 percentage points of raw multi-label micro-F1 inflation (70.9% vs. 28.9%).
2. **Zero-Shot 4-Bit LLM Extraction Distortion:** Strictly blinded evaluation of `unsloth/mistral-7b-instruct-v0.2-bnb-4bit` (`max_new_tokens=20`, `temperature=0.1`) yields moderate binary screening (F1 = 73.0%, kappa = 0.6098) but poor granular extraction (raw micro-F1 = 28.9%; synonym-normalized = 44.4%; metabolic raw = 19.4%, BPC-157 = 34.8%). Prompt-example echoing inflated nausea from 0.0% to 19.2% in BPC-157 and from 15.4% to 38.5% in metabolic posts.
3. **Subreddit Ascertainment Bias:** Excluding outcome-named subreddits (`r/Anhedonia`, `r/PSSD`) reduced BPC-157 raw anhedonia share by 66.5% (7.65% to 2.56%), whereas Retatrutide remained invariant (3.04% to 2.98%).
4. **Leak-Free Temporal GNN Forecasting Null Result:** After removing 894 `ANONYMOUS_USER` deleted-author hub records and enforcing 4-index zero-overlap embargo splits across 164 contiguous 4-week reply graphs (June 2023 to August 2026), Structural GCN (AUPRC = 0.684) and Temporal GAT (0.775) did not outperform zero-parameter prevalence mean reversion (0.798; GCN Delta AUROC = -0.101, paired block-bootstrap p = 0.003), and within-graph label permutation raised GCN AUPRC to 0.774.

## Repository Structure

```text
digital-pharmacovigilance-llm-gnn-audit/
|-- README.md
|-- LICENSE
|-- CITATION.cff
|-- requirements.txt
|-- config/
|   `-- pipeline_prompt_and_hyperparams.json
|-- data/
|   |-- anonymized_temporal_graphs.pt
|   |-- window_level_aggregates_and_predictions.csv
|   |-- deidentified_Metabolic_N60_annotation_labels.csv
|   |-- deidentified_BPC157_N60_annotation_labels.csv
|   `-- deidentified_N200_annotation_labels.csv
|-- scripts/
|   `-- reproduce_audit_metrics.py
|-- supporting_information_plos/
|   |-- S1_Table.csv
|   |-- S2_Table.csv
|   |-- S3_Table.csv
|   |-- S4_Table.xlsx
|   |-- S5_Table.csv
|   `-- S6_Table.xlsx
`-- tables/
    `-- Rev_Table*.csv (all 27 audit tables)
```

## Quick Reproduction

```bash
pip install -r requirements.txt
python scripts/reproduce_audit_metrics.py
```

## Privacy and Ethics Compliance

In compliance with Reddit API Terms of Service, CIOMS Guideline 12, and AoIR ethical standards, no raw verbatim Reddit text, Reddit record IDs, or username hashes are included in this repository. All downstream validation and forecasting benchmarks are reproducible directly from the anonymized tensors and aggregate tables provided.

## Citation

If you use this audit benchmark or anonymized graph dataset, please cite:

> Rafi TI, Ahmed H, Hemal MAKP, Jubair H. Auditing Zero-Shot Large Language Model and Graph Neural Network Pipelines for Digital Pharmacovigilance of Grey-Market Peptides and Metabolic Therapeutics. *PLOS Digital Health* (in revision). Zenodo archive: https://doi.org/[INSERT ZENODO DOI].
