# Generalization Testing (5 Levels)

## Overview
Tests whether trained LSTM surrogates generalize beyond their training distribution — the key requirement for neural prosthetic viability. Follows the DESCARTES Generalization Guide.

## Five Levels

### Level 1: Cross-Condition (Temporal Split)
- **Method**: Train on early 60% of data, test on late 40%
- **Circuits**: Circuit 4 (3 best WM patients) + Circuit 5 (sub-CS48 movie)
- **Metric**: Generalization Ratio = CC_novel / CC_trained (>0.8 = STRONG)
- **Adaptation**: No condition metadata exists in NWB trial tables, so temporal split substitutes for leave-one-condition-out

### Level 2: Cross-Session
- **Method**: Train on session 1, test on session 2 (same patient, different day)
- **Data**: Patients with both ses-1 and ses-2 NWBs in DANDI 000469
- **Neuron alignment**: Truncate to minimum shared neuron count

### Level 3: Cross-Patient (Population Embedding)
- **Method**: Leave-one-patient-out with PatientAgnosticSurrogate
- **Architecture**: neuron_encoder → mean pooling → LSTM → neuron_decoder
- **Handles**: Variable neuron counts across patients via per-neuron embedding

### Level 4: Cross-Task
- **Status**: NOT TESTABLE
- **Reason**: No shared patients between Circuit 4 (DANDI 000469, human MTL/frontal) and Circuit 5 (DANDI 000623, human limbic/prefrontal). Different brain regions.

### Level 5: Perturbation Robustness
- **Neuron Dropout**: Zero out 10-50% of input neurons, measure CC retention
- **Noise Injection**: Add Gaussian noise at SNR 1-20, measure degradation
- **Gain Drift**: Simulate electrode drift (gradual gain change over time)
- **Pass criterion**: >80% CC retention at mild perturbation levels

## Running

**Locally (CPU, memory-safe):**
```bash
cd "movie emotion"
python -X utf8 generalization/run_generalization.py
```

**On Vast.ai (GPU):**
```bash
ALLOW_GPU=1 python -X utf8 generalization/run_generalization.py
```

## Output
Results saved to `results/generalization/generalization_report.json`

## Memory Safety
- Forces CPU by default (set `ALLOW_GPU=1` for cloud)
- Loads only 1-2 patients at a time for Level 3
- Explicit `gc.collect()` between all levels
- Level 2 skips if ses-2 NWBs not downloaded (no mid-run downloads)
