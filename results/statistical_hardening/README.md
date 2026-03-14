# Statistical Hardening Results (Phase 1-6)

## Overview
Six-phase statistical hardening pipeline testing whether LSTM surrogate models learn genuine neural computation vs. exploiting temporal autocorrelation artifacts.

## Method
- **Phase 1**: Matched baselines + iAAFT surrogate significance testing (1000 surrogates for C2, 200 for others)
- **Phase 2**: Gate probing + temporal generalization matrices
- **Phase 3**: Distributed Alignment Search (DAS) + conditional mutual information
- **Phase 4**: Family-wise error correction (Simes + Holm)
- **Phase 5**: CC-normalized effect sizes + transfer probing
- **Phase 6**: Topological analysis (PCA-based dimensionality)

## Key Results

### Circuit 2 (CA3 → CA1 Hippocampal)
- **File**: `phase1_circuit2.json`
- **N surrogates**: 1000
- **gamma_amp**: delta_R2 = 0.234, p = 0.163 — **NOT SIGNIFICANT**
- **All 8 tested targets FAIL** iAAFT significance
- **Conclusion**: The cornerstone CA3→CA1 finding does not survive phase-randomized null testing

### Circuit 3 (Thalamus → Cortex, Mouse)
- **File**: `phase1_circuit3.json`
- **sub-440959** (real bio targets from DANDI 000363): All 7 targets fail. theta_modulation closest at p=0.055
- **sub-440956** (proxy targets only): PC2 survives (p=0.000) but is circular (PCA on hidden states)
- **Conclusion**: No biological target survives when using real DANDI-streamed data

### Circuit 4 (Human Working Memory, DANDI 000469)
- **File**: `phase1_circuit4.json`
- **13 patients tested** across ses-1 and ses-2
- **trial_variance** and **temporal_stability** survive in 8/13 patients (62%)
- **firing_rate** survives in 7/13 patients (54%)
- **sub-10_ses-1**: 5/5 targets significant (strongest patient)
- **population_synchrony**: Only significant in sub-10 (N=1 outlier)

### Circuit 5 (Movie Emotion, Limbic → Prefrontal)
- **File**: `phase1_sub-CS48.json`
- **temporal_stability**: p=0.000, z=5.6 — **SIGNIFICANT** (only survivor)
- All other targets (arousal, threat_pe, event_boundaries, etc.) fail
- **Phases 2-6** in `phases2to6_sub-CS48.json`: population_synchrony shows partial DAS (interchange accuracy ~0.49) but transient temporal pattern

## Datasets Used
| Dataset | DANDI ID | Species | Brain Region |
|---------|----------|---------|-------------|
| Hippocampal model | Simulated | N/A | CA3 → CA1 |
| Thalamocortical | 000363 | Mouse | Thalamus → Visual Cortex |
| Working Memory | 000469 | Human | MTL + Frontal (41 patients) |
| Movie Emotion | 000623 | Human | Limbic → Prefrontal |

## Interpretation
The statistical hardening pipeline reveals that most surrogate "findings" are artifacts of temporal autocorrelation. Only **temporal_stability** (Circuit 5) and **trial_variance/temporal_stability** (Circuit 4, 62% of patients) survive rigorous testing. The gamma_amp finding in Circuit 2 — previously the project's cornerstone — is not statistically significant.
