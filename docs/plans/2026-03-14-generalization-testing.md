# DESCARTES Generalization Testing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Test whether trained LSTM surrogates generalize beyond training data across 5 levels, from temporal splits to perturbation robustness.

**Architecture:** Single script `generalization/run_generalization.py` implementing all 5 levels as independent functions. Each level returns a results dict. A main orchestrator runs levels sequentially with checkpoints (stop if Level 1 ratio < 0.5). Uses existing `HumanLSTMSurrogate` (Circuit 4) and `LimbicPrefrontalLSTM` (Circuit 5) models.

**Tech Stack:** PyTorch (CUDA RTX 5070), NumPy, pynwb, scipy. Models: HumanLSTMSurrogate (C4), LimbicPrefrontalLSTM (C5). Data: NWB files (C4), preprocessed NPZ (C5).

---

## Data Reality Check

| Item | Expected by Guide | Actual |
|------|-------------------|--------|
| C4 memory load column | `loads`, `set_size`, etc. | **MISSING** — only `id`, `start_time`, `stop_time` |
| C4 Level 1 approach | Leave-one-load-out | **Temporal split** (early vs late trials) |
| C5 data format | `(n_windows, T, n_input)` | `preprocessed_data/sub-CS48.npz` — (572+191+191, 100, 46/54) |
| C4 ses-2 NWBs | Available for cross-session | **Only 3 ses-1** + 10 newly downloaded ses-1. Need ses-2 downloads for Level 2 |
| Shared patients C4/C5 | For cross-task | **None** — different DANDI datasets (000469 vs movie), different species regions |
| Device | GPU | RTX 5070, 12.8 GB VRAM |

## Adaptation Decisions

1. **Level 1 C4**: Use temporal split (first 60% vs last 40% of 378 trials) since no load metadata exists. This still tests whether the surrogate learned the transformation vs memorized specific trial patterns.
2. **Level 2**: Download ses-2 NWBs for sub-1, sub-10, sub-12 (3 patients with highest CC). Match neurons by electrode index (same NWB structure between sessions).
3. **Level 4**: Skip — no shared patients between Circuit 4 (human MTL/frontal, Rutishauser) and Circuit 5 (movie emotion, different dataset). Cannot meaningfully test cross-task without same neurons. Document as "NOT TESTABLE" in report.

---

### Task 1: Create generalization directory and compute_cc utility

**Files:**
- Create: `generalization/__init__.py`
- Create: `generalization/utils.py`

**Step 1:** Create utility module with `compute_cc()` and shared training loop

```python
# generalization/utils.py
def compute_cc(y_pred, y_true):
    """Mean per-neuron Pearson correlation."""

def train_model_early_stop(model, X_tr, Y_tr, X_val, Y_val,
                            n_epochs=200, patience=20, lr=1e-3, device='cuda'):
    """Standard training loop with early stopping. Returns trained model."""
```

**Step 2:** Verify utility imports work

Run: `python -c "from generalization.utils import compute_cc; print('OK')"`

---

### Task 2: Level 1 — Cross-Condition (Temporal Split)

**Files:**
- Create: `generalization/level1_cross_condition.py`

**Step 1:** Implement Circuit 5 temporal split

```python
def level1_circuit5(preprocessed_path, model_checkpoint_path,
                     train_fraction=0.6, n_seeds=5, device='cuda'):
    """Train on early movie (60%), test on late movie (40%).
    Returns dict with cc_same, cc_novel, generalization_ratio."""
```

Uses full timeline data (input_rates/output_rates), re-windows into train/test splits by time.

**Step 2:** Implement Circuit 4 temporal split

```python
def level1_circuit4(nwb_path, schema, model_checkpoint_path,
                     train_fraction=0.6, n_seeds=5, device='cuda'):
    """Train on early trials (60%), test on late trials (40%).
    Reports CC on held-out early trials vs CC on late trials."""
```

Loads all 378 trials, splits temporally, trains fresh HumanLSTMSurrogate.

**Step 3:** Run both and report ratios

Run: `python -c "from generalization.level1_cross_condition import ...; ..."`
Expected: Generalization ratio per circuit. CHECKPOINT: ratio > 0.5 to proceed.

---

### Task 3: Level 2 — Cross-Session

**Files:**
- Create: `generalization/level2_cross_session.py`

**Step 1:** Download ses-2 NWBs for sub-1, sub-10, sub-12

**Step 2:** Implement electrode-matched cross-session test

```python
def level2_cross_session(ses1_nwb, ses2_nwb, schema,
                          hidden_size=128, n_seeds=5, device='cuda'):
    """Train on session 1, test on session 2. Match neurons by electrode."""
```

Extract electrode IDs from both sessions, find shared neurons, subset X/Y, train on ses-1 shared neurons, test on ses-2 shared neurons.

**Step 3:** Run for each patient with 2 sessions

---

### Task 4: Level 3 — Cross-Patient (PatientAgnosticSurrogate)

**Files:**
- Create: `generalization/level3_cross_patient.py`

**Step 1:** Implement PatientAgnosticSurrogate

```python
class PatientAgnosticSurrogate(nn.Module):
    """Population embedding: neuron_encoder → mean → LSTM → neuron_decoder.
    Handles variable n_input/n_output across patients."""
```

**Step 2:** Implement leave-one-patient-out

```python
def level3_cross_patient(patient_data_dict, n_epochs=100, device='cuda'):
    """For each patient: train on all others, test on held-out.
    patient_data_dict: {pid: {'X': (n_trials, T, n_in), 'Y': (n_trials, T, n_out), 'cc': float}}
    """
```

Uses 13 downloaded patients. Each iteration trains PatientAgnosticSurrogate on 12, tests on 1.

**Step 3:** Run and report per-patient CC_novel and ratio vs within-patient CC

---

### Task 5: Level 5 — Perturbation Robustness

**Files:**
- Create: `generalization/level5_perturbation.py`

**Step 1:** Implement three perturbation tests

```python
def neuron_dropout_test(model, X_test, Y_test, fracs=[0.1,0.2,0.3,0.5], ...)
def noise_injection_test(model, X_test, Y_test, snrs=[20,10,5,2,1], ...)
def gain_drift_test(model, X_test, Y_test, magnitudes=[0.1,0.2,0.5,1.0], ...)
```

**Step 2:** Run on Circuit 4 sub-10 (best patient, CC=0.858) and Circuit 5 CS48

---

### Task 6: Orchestrator and Report

**Files:**
- Create: `generalization/run_generalization.py`

**Step 1:** Implement main orchestrator

```python
def run_all_levels(device='cuda'):
    """Run all generalization levels sequentially.
    Stops after Level 1 if ratio < 0.5.
    Saves results/generalization_report.json."""
```

**Step 2:** Run full pipeline

Run: `cd "movie emotion" && python -X utf8 generalization/run_generalization.py`

**Step 3:** Print summary table and save report

---

## Execution Order

1. Task 1 (utils) → Task 2 (Level 1) → CHECKPOINT
2. If Level 1 passes: Task 3 (Level 2) + Task 4 (Level 3) in sequence
3. Task 5 (Level 5) — independent of Levels 2-3
4. Task 6 (orchestrator + report)

## Time Estimates

- Task 1: 5 min (boilerplate)
- Task 2: 20 min (training 5 seeds × 2 circuits)
- Task 3: 15 min (3 patients × 5 seeds, requires NWB download)
- Task 4: 30 min (13 leave-one-out iterations)
- Task 5: 10 min (perturbation sweeps, inference only)
- Task 6: 5 min (orchestrator)

**Total: ~85 minutes**
