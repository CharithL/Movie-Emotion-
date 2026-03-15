# Generalization Testing (5 Levels)

## Overview
Tests whether trained LSTM surrogates generalize beyond their training distribution — the key requirement for neural prosthetic viability. Follows the DESCARTES Generalization Guide.

## Final Results (2026-03-15)

| Level | Test | Ratio | Verdict |
|-------|------|-------|---------|
| **1** | Cross-Condition (Temporal Split) | C5: **1.109**, C4: **0.822** | STRONG |
| **2** | Cross-Session (ses-1 → ses-2) | **-0.153** | FAILURE |
| **3** | Cross-Patient (Leave-One-Out) | **-0.003** | FAILURE |
| **4** | Cross-Task | N/A | NOT TESTABLE |
| **5** | Perturbation Robustness | See below | PARTIAL |

**Overall: NOT prosthetic-viable.** Models generalize temporally but fail across sessions and patients.

### Level 1: Cross-Condition (Temporal Split) — STRONG
- **Method**: Train on early 60%, test on late 40% of recording
- **Circuit 5** (sub-CS48 movie): ratio = 1.109 ± 0.047 across 5 seeds
- **Circuit 4** (3 WM patients): ratio = 0.822 overall
  - sub-10: 1.445 (STRONG), sub-8: 0.721 (PARTIAL), sub-4: 0.300 (FAILURE)
- **Interpretation**: Models learn temporal dynamics, not memorized time points

### Level 2: Cross-Session — FAILURE
- **Method**: Train on session 1, test on session 2 (different day)
- **3 patients tested**: sub-1, sub-10, sub-12
- All produce negative CC on session 2 (worse than chance)
- **Interpretation**: Neural representations drift too much between sessions for same-architecture transfer. Neuron alignment by truncation is insufficient.

### Level 3: Cross-Patient — FAILURE
- **Method**: Leave-one-patient-out with PatientAgnosticSurrogate
- **Architecture**: per-neuron embedding → mean pool → LSTM(128) → neuron decoder
- **6 patients** (top CC: sub-11, sub-8, sub-3, sub-7, sub-1, sub-4)
- All held-out patients: CC ≈ 0 (range: -0.011 to +0.001)
- **Interpretation**: Population embedding with 50 epochs is insufficient. Each patient's neural code is too idiosyncratic for simple mean-pooling to abstract.

### Level 4: Cross-Task — NOT TESTABLE
No shared patients between Circuit 4 (DANDI 000469, human MTL/frontal) and Circuit 5 (DANDI 000623, human limbic/prefrontal).

### Level 5: Perturbation Robustness — PARTIAL

**Circuit 4 (sub-10, baseline CC=0.162):**
| Perturbation | Retention | Robust? |
|-------------|-----------|---------|
| 10% neuron dropout | 95.0% | YES |
| 20% neuron dropout | 91.5% | YES |
| 30% neuron dropout | 86.3% | YES |
| 50% neuron dropout | 68.8% | NO |
| SNR=5 noise | 89.7% | YES |
| SNR=1 noise | 63.9% | NO |
| Gain drift 0.5x | 93.4% | YES |

**Circuit 5 (sub-CS48, baseline CC=0.086):**
| Perturbation | Retention | Robust? |
|-------------|-----------|---------|
| 10% neuron dropout | 90.0% | YES |
| 20% neuron dropout | 79.9% | NO |
| Noise (all SNR) | ~100% | YES |
| Gain drift (all) | ~100% | YES |

Circuit 4 is more robust to dropout; Circuit 5 is noise-immune but dropout-sensitive.

## Method Details

### Generalization Ratio
```
ratio = CC_novel / CC_trained
> 0.8  = STRONG (prosthetic-grade)
0.5-0.8 = PARTIAL (needs improvement)
< 0.5  = FAILURE (memorized training distribution)
```

### Adaptations from Guide
- **Level 1**: No condition metadata in NWB trial tables → temporal split instead of leave-one-condition-out
- **Level 3**: Pre-loads all patient data once to avoid 250 NWB file re-reads (original design crashed machines)
- **Level 3**: Caps trials to 64 per patient to limit memory (sub-11 has 378×1247 bins)

## Running
```bash
# Local (CPU, memory-safe, ~14 min)
cd "movie emotion"
python -X utf8 generalization/run_generalization.py

# Cloud GPU
ALLOW_GPU=1 python -X utf8 generalization/run_generalization.py
```

## Output
`results/generalization/generalization_report.json` — full per-seed, per-patient results.

## Datasets Used
| Dataset | DANDI ID | Species | Brain Region | Patients |
|---------|----------|---------|-------------|----------|
| Working Memory | 000469 | Human | MTL + Frontal | 13 eligible |
| Movie Emotion | 000623 | Human | Limbic → Prefrontal | sub-CS48 |
