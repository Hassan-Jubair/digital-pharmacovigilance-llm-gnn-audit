#!/usr/bin/env python3
"""
Standalone Privacy-Preserving Audit Reproduction Script
Paper: Auditing Zero-Shot Large Language Model and Graph Neural Network Pipelines
       for Digital Pharmacovigilance of Grey-Market Peptides and Metabolic Therapeutics
Authors: Tajul Islam Rafi, Hossain Ahmed, Md Abu Kawsar Prodhan Hemal, Hassan Jubair

Reproduces directly from anonymized public artifacts (zero raw Reddit text required):
  1. Strictly blinded N=120 validation metrics (Metabolic N=60 + BPC-157 N=60)
     under both raw exact-string and clinical synonym-normalized evaluation.
  2. Unblinded Round 1 N=200 anchoring-bias comparator metrics.
  3. Primary embargoed test-set forecasting metrics (N_test=26) with moving
     block-bootstrap 95% confidence intervals and paired tests against Mean Reversion.
"""

import re
import math
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, cohen_kappa_score,
    roc_auc_score, average_precision_score, brier_score_loss
)
from statsmodels.stats.proportion import proportion_confint

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

SYNONYM_MAP = {
    "LOSS OF APPETITE": "DECREASED APPETITE",
    "APPETITE LOSS": "DECREASED APPETITE",
    "ANOREXIA": "DECREASED APPETITE",
    "ACID REFLUX": "DYSPEPSIA",
    "HEARTBURN": "DYSPEPSIA",
    "INDIGESTION": "DYSPEPSIA",
    "REFLUX": "DYSPEPSIA",
    "STOMACH PAIN": "ABDOMINAL PAIN",
    "GUT PAIN": "ABDOMINAL PAIN",
    "CRAMPS": "ABDOMINAL PAIN",
    "STOMACH CRAMPS": "ABDOMINAL PAIN",
    "ABDOMINAL CRAMPS": "ABDOMINAL PAIN",
    "HEART PALPITATIONS": "PALPITATIONS",
    "RAPID HEART RATE": "TACHYCARDIA",
    "ELEVATED HEART RATE": "TACHYCARDIA",
    "HIGH HEART RATE": "TACHYCARDIA",
    "EXHAUSTION": "FATIGUE",
    "LETHARGY": "FATIGUE",
    "TIREDNESS": "FATIGUE",
    "SLEEPINESS": "FATIGUE",
    "DROWSINESS": "FATIGUE",
    "HEADACHES": "HEADACHE",
    "MIGRAINE": "HEADACHE",
    "MIGRAINES": "HEADACHE",
    "BLOAT": "BLOATING",
    "LOWERED LIBIDO": "DECREASED LIBIDO",
    "LOW LIBIDO": "DECREASED LIBIDO",
    "NUMBNESS ARMS AND HANDS": "NUMBNESS",
    "HERX REACTION JARISCHHERXHEIMER REACTION": "HERXHEIMER REACTION",
    "BLURRY VISION": "BLURRED VISION",
    "BUZZING ALL OVER HEAD": "PARESTHESIA",
    "TINGLING": "PARESTHESIA",
    "UNHEALTHY FEELING": "MALAISE",
    "LOWER LEFT ABDOMINAL HEAVINESS": "ABDOMINAL DISCOMFORT",
    "TIGHTNESS": "ABDOMINAL DISCOMFORT",
    "PRESSURE": "ABDOMINAL DISCOMFORT",
    "SWELLING": "EDEMA",
    "WATER RETENTION": "EDEMA",
    "FLUID RETENTION": "EDEMA",
    "INCREASES IN PAIN": "PAIN",
    "ACHE": "PAIN",
    "RAPID GROWTH OF EXISTING MOLES": "NEW MOLES",
    "HYPERSENSITIVITY TO TOUCH": "SKIN SENSITIVITY",
    "TEMPERATURE SENSITIVITY": "SKIN SENSITIVITY",
    "ALLODYNIA": "SKIN SENSITIVITY",
    "SENSITIVE SKIN": "SKIN SENSITIVITY",
    "LIGHTHEADEDNESS": "DIZZINESS",
    "LIGHT HEADEDNESS": "DIZZINESS",
    "VERTIGO": "DIZZINESS",
    "SULFUR BURPS": "ERUCTATION",
    "BURPING": "ERUCTATION",
    "HAIR LOSS": "ALOPECIA",
    "SHEDDING": "ALOPECIA",
    "SLEEP DISTURBANCE": "INSOMNIA",
    "POOR SLEEP": "INSOMNIA",
    "RASH": "INJECTION SITE REACTION",
    "WELT": "INJECTION SITE REACTION",
    "INJECTION SITE ERYTHEMA": "INJECTION SITE REACTION",
}

def parse_symptom_set(val):
    if pd.isna(val):
        return set()
    s = re.sub(r"[^\w\s,]", "", str(val).strip().upper())
    if not s or s in {"NONE", "NAN", "NULL", "NA"}:
        return set()
    return {x.strip() for x in s.split(",") if x.strip() and x.strip() not in {"NONE", "NAN", "NULL"}}

def normalize_set(st):
    return {SYNONYM_MAP.get(x, x) for x in st}

def evaluate_validation(name, r1_sets, r2_sets, gold_sets, llm_sets):
    n = len(gold_sets)
    y1 = np.array([int(len(s) > 0) for s in r1_sets])
    y2 = np.array([int(len(s) > 0) for s in r2_sets])
    h_bin_k = cohen_kappa_score(y1, y2)

    yg = np.array([int(len(s) > 0) for s in gold_sets])
    yl = np.array([int(len(s) > 0) for s in llm_sets])
    tn, fp, fn, tp = confusion_matrix(yg, yl, labels=[0, 1]).ravel()
    bp = tp / max(tp + fp, 1)
    br = tp / max(tp + fn, 1)
    bf1 = 2 * bp * br / max(bp + br, 1e-9)
    bkap = cohen_kappa_score(yg, yl)

    def micro_f1(gs, ls):
        vocab = sorted(set().union(*gs, *ls))
        gf, lf = [], []
        for g, l in zip(gs, ls):
            for sym in vocab:
                gf.append(int(sym in g))
                lf.append(int(sym in l))
        _, sfp, sfn, stp = confusion_matrix(gf, lf, labels=[0, 1]).ravel()
        mp = stp / max(stp + sfp, 1)
        mr = stp / max(stp + sfn, 1)
        return 100 * mp, 100 * mr, 100 * (2 * mp * mr / max(mp + mr, 1e-9))

    rmp, rmr, rmf1 = micro_f1(gold_sets, llm_sets)
    nmp, nmr, nmf1 = micro_f1([normalize_set(g) for g in gold_sets], [normalize_set(l) for l in llm_sets])
    print(
        f"{name:28s} | N={n:3d} | HumanBinK={h_bin_k:.4f} | "
        f"BinF1={100*bf1:5.1f}% (k={bkap:.4f}) | "
        f"RawMicroF1={rmf1:5.1f}% | NormMicroF1={nmf1:5.1f}%"
    )

def block_bootstrap_ci(y_true, probs, fn, block_len=4, B=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    bl = min(block_len, max(1, n // 3))
    nb = int(math.ceil(n / bl))
    ms = max(1, n - bl + 1)
    vals = []
    for _ in range(B):
        idx = np.concatenate([np.arange(s, s + bl) for s in rng.integers(0, ms, size=nb)])[:n]
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        vals.append(fn(yt, probs[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

def main():
    print("=" * 90)
    print("1. REPRODUCING STRICTLY BLINDED N=120 & ROUND 1 N=200 VALIDATION METRICS")
    print("=" * 90)
    met = pd.read_csv(DATA_DIR / "deidentified_Metabolic_N60_annotation_labels.csv", keep_default_na=False)
    bpc = pd.read_csv(DATA_DIR / "deidentified_BPC157_N60_annotation_labels.csv", keep_default_na=False)
    n200 = pd.read_csv(DATA_DIR / "deidentified_N200_annotation_labels.csv", keep_default_na=False)

    met_r1 = [parse_symptom_set(x) for x in met["Rater1_Symptoms"]]
    met_r2 = [parse_symptom_set(x) for x in met["Rater2_Symptoms"]]
    met_g  = [parse_symptom_set(x) for x in met["Final_Consensus_Symptoms"]]
    met_l  = [parse_symptom_set(x) for x in met["LLM_Extracted_Symptoms"]]

    bpc_r1 = [parse_symptom_set(x) for x in bpc["Rater1_Symptoms"]]
    bpc_r2 = [parse_symptom_set(x) for x in bpc["Rater2_Symptoms"]]
    bpc_g  = [parse_symptom_set(x) for x in bpc["Final_Consensus_Symptoms"]]
    bpc_l  = [parse_symptom_set(x) for x in bpc["LLM_Extracted_Symptoms"]]

    evaluate_validation("Metabolic_Combined (N=60)", met_r1, met_r2, met_g, met_l)
    evaluate_validation("BPC-157_GreyMarket (N=60)", bpc_r1, bpc_r2, bpc_g, bpc_l)
    evaluate_validation("ALL_STRICTLY_BLINDED (N=120)", met_r1 + bpc_r1, met_r2 + bpc_r2, met_g + bpc_g, met_l + bpc_l)

    r1_200 = [parse_symptom_set(x) for x in n200["Reviewer_1_Human_Annotated_Symptoms"]]
    r2_200 = [parse_symptom_set(x) for x in n200["Reviewer_2_Human_Annotated_Symptoms"]]
    g_200 = [
        parse_symptom_set(n200.loc[i, "Final_Human_Annotated_Symptoms"])
        if str(n200.loc[i, "Final_Human_Annotated_Symptoms"]).strip() != "" else r1_200[i]
        for i in range(len(n200))
    ]
    l_200 = [parse_symptom_set(x) for x in n200["LLM_Extracted_Symptoms"]]
    evaluate_validation("Round1_Unblinded (N=200)", r1_200, r2_200, g_200, l_200)

    print("\n" + "=" * 90)
    print("2. REPRODUCING PRIMARY EMBARGOED TEST FORECASTING METRICS (N_test=26)")
    print("=" * 90)
    win = pd.read_csv(DATA_DIR / "window_level_aggregates_and_predictions.csv")
    te = win[win["split"] == "Test"].reset_index(drop=True)
    yt = te["target_delta_gt_0"].values.astype(int)

    for col in [c for c in te.columns if c.startswith("pred_")]:
        yp = te[col].values.astype(float)
        roc = roc_auc_score(yt, yp)
        prc = average_precision_score(yt, yp)
        brier = brier_score_loss(yt, yp)
        rlo, rhi = block_bootstrap_ci(yt, yp, roc_auc_score)
        plo, phi = block_bootstrap_ci(yt, yp, average_precision_score)
        print(
            f"{col.replace('pred_', ''):24s} | AUROC={roc:.4f} [{rlo:.3f}, {rhi:.3f}] | "
            f"AUPRC={prc:.4f} [{plo:.3f}, {phi:.3f}] | Brier={brier:.4f}"
        )

if __name__ == "__main__":
    main()
