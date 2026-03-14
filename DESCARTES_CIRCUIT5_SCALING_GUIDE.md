# DESCARTES Circuit 5 — Claude Code Guide: Probe Pipeline Fix + 14-Patient Scaling

## Context

Read the main guide at DESCARTES_MOVIE_EMOTION_GUIDE.md for full project context.

This is a focused execution guide for two specific tasks:
1. Fix the Ridge ΔR² probe pipeline to eliminate impossible values
2. Scale corrected probing across all 14 included patients

The pilot run on sub-CS48 revealed:
- CC = 0.63 on focused 10-neuron output (reliable for probing)
- ΔR² values frequently outside [-1, +1], which is mathematically impossible
  for a valid delta (e.g., face_presence at +1.234, arousal at +5.913)
- Arousal/valence signals still producing ΔR² of -9 to -18
- Ablation methodology does not work for this LSTM architecture 
  (random ablation z-scores are 2-6× worse than target-aligned — model 
  is holographic, not modular). DO NOT run ablation in this scaling pass.
- Emotion category is a 4-class categorical label requiring logistic 
  regression, not Ridge regression

The goal: clean probing results across 14 patients that can be reported
as "which biological variables does the limbic→prefrontal transformation
consistently learn?" — even if none are causally mandatory.

---

## Task A: Fix the Probe Pipeline

### A.1 The Root Cause of Impossible ΔR²

R² is defined as 1 - (SS_res / SS_tot). When the model predicts worse
than the mean baseline, R² goes negative — potentially very negative 
if the predictions are anti-correlated with the target. This happens 
when:

1. The target signal has pathological variance (extreme outliers, 
   bimodal distribution, near-constant with rare spikes)
2. The Ridge regression overfits on training folds and anti-predicts 
   on test folds
3. The target is categorical but treated as continuous
4. The target was not properly normalized

When R²_untrained is -5.0 and R²_trained is +0.3, ΔR² = 5.3 — which
looks like a huge positive finding but actually means both models are
terrible, one just slightly less terrible.

### A.2 Fix: Sanitized Ridge ΔR² Function

Replace the existing ridge_delta_r2 function with this version:

```python
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score
from typing import Dict, Optional

def sanitized_ridge_delta_r2(hidden_states: np.ndarray,
                              target: np.ndarray,
                              hidden_untrained: np.ndarray,
                              n_folds: int = 5,
                              alphas: np.ndarray = np.logspace(-3, 3, 20),
                              max_abs_r2: float = 1.0) -> Dict:
    """
    Sanitized Ridge ΔR² that guards against impossible values.
    
    THREE GUARDS:
    1. Target normalization: z-score target, clip outliers to ±3σ
    2. R² clamping: clamp individual fold R² to [-1, +1]
       (R² < -1 means the probe is catastrophically wrong — 
       any value below -1 carries no additional information)
    3. Validity flag: mark result as INVALID if R²_trained < 0
       (if trained model can't beat the mean, the target is 
       not meaningfully decodable regardless of ΔR²)
    
    Args:
        hidden_states: (T, hidden_size) trained model hidden states
        target: (T,) probe target (continuous, NOT categorical)
        hidden_untrained: (T, hidden_size) untrained model hidden states
        n_folds: cross-validation folds
        alphas: Ridge alpha search grid
        max_abs_r2: clamp R² to [-max_abs_r2, +max_abs_r2]
    
    Returns:
        dict with:
            'r2_trained': float, clamped R²
            'r2_untrained': float, clamped R²
            'delta_r2': float, guaranteed in [-2, +2] range
            'valid': bool, True only if r2_trained > 0
            'target_stats': dict with target distribution info
    """
    # ── Guard 1: Normalize and clip target ──
    target = target.copy().astype(np.float64)
    t_mean = np.mean(target)
    t_std = np.std(target)
    
    target_stats = {
        'mean': float(t_mean),
        'std': float(t_std),
        'min': float(np.min(target)),
        'max': float(np.max(target)),
        'pct_1': float(np.percentile(target, 1)),
        'pct_99': float(np.percentile(target, 99)),
    }
    
    # Skip if target is near-constant (std < 1e-8)
    if t_std < 1e-8:
        return {
            'r2_trained': 0.0,
            'r2_untrained': 0.0,
            'delta_r2': 0.0,
            'valid': False,
            'reason': 'target near-constant (std < 1e-8)',
            'target_stats': target_stats,
        }
    
    # Z-score and clip to ±3σ
    target = (target - t_mean) / t_std
    target = np.clip(target, -3.0, 3.0)
    
    # ── Probe trained model ──
    kf = KFold(n_splits=n_folds, shuffle=False)
    r2_trained_folds = []
    
    for train_idx, test_idx in kf.split(hidden_states):
        ridge = RidgeCV(alphas=alphas)
        ridge.fit(hidden_states[train_idx], target[train_idx])
        r2 = ridge.score(hidden_states[test_idx], target[test_idx])
        # Guard 2: Clamp per-fold R²
        r2 = np.clip(r2, -max_abs_r2, max_abs_r2)
        r2_trained_folds.append(r2)
    
    r2_trained = float(np.mean(r2_trained_folds))
    
    # ── Probe untrained model ──
    r2_untrained_folds = []
    
    for train_idx, test_idx in kf.split(hidden_untrained):
        ridge = RidgeCV(alphas=alphas)
        ridge.fit(hidden_untrained[train_idx], target[train_idx])
        r2 = ridge.score(hidden_untrained[test_idx], target[test_idx])
        r2 = np.clip(r2, -max_abs_r2, max_abs_r2)
        r2_untrained_folds.append(r2)
    
    r2_untrained = float(np.mean(r2_untrained_folds))
    
    # ── Compute delta ──
    delta_r2 = r2_trained - r2_untrained
    
    # ── Guard 3: Validity check ──
    # If trained model can't beat the mean (R² < 0), the target
    # is not meaningfully decodable. ΔR² might still be positive
    # (trained is less bad than untrained) but this is not a
    # genuine "learned representation" finding.
    valid = r2_trained > 0.0
    
    return {
        'r2_trained': r2_trained,
        'r2_untrained': r2_untrained,
        'delta_r2': delta_r2,
        'r2_trained_folds': r2_trained_folds,
        'r2_untrained_folds': r2_untrained_folds,
        'valid': valid,
        'reason': 'OK' if valid else 'r2_trained <= 0 (target not decodable)',
        'target_stats': target_stats,
    }


def classification_delta_accuracy(hidden_states: np.ndarray,
                                   target_labels: np.ndarray,
                                   hidden_untrained: np.ndarray,
                                   n_folds: int = 5) -> Dict:
    """
    For CATEGORICAL probe targets (like emotion_category).
    Uses logistic regression with balanced accuracy.
    
    Balanced accuracy corrects for class imbalance — if 83% of
    time bins are neutral, chance balanced accuracy = 1/n_classes,
    not 83%.
    
    Args:
        hidden_states: (T, hidden_size) trained hidden states
        target_labels: (T,) integer class labels
        hidden_untrained: (T, hidden_size) untrained hidden states
        n_folds: CV folds
    
    Returns:
        dict with accuracies, delta, class distribution
    """
    # ── Class distribution check ──
    unique_classes, class_counts = np.unique(target_labels, return_counts=True)
    n_classes = len(unique_classes)
    class_dist = {int(c): int(n) for c, n in zip(unique_classes, class_counts)}
    majority_fraction = float(np.max(class_counts) / len(target_labels))
    chance_balanced = 1.0 / n_classes
    
    if n_classes < 2:
        return {
            'acc_trained': 0.0,
            'acc_untrained': 0.0,
            'delta_accuracy': 0.0,
            'chance_balanced': 0.0,
            'valid': False,
            'reason': f'only {n_classes} class(es) in data',
            'class_distribution': class_dist,
        }
    
    # ── Standardize features (important for logistic regression) ──
    scaler_t = StandardScaler()
    hidden_scaled = scaler_t.fit_transform(hidden_states)
    
    scaler_u = StandardScaler()
    untrained_scaled = scaler_u.fit_transform(hidden_untrained)
    
    # ── Trained model ──
    kf = KFold(n_splits=n_folds, shuffle=False)
    acc_trained_folds = []
    
    for train_idx, test_idx in kf.split(hidden_scaled):
        # Check that test fold has ≥2 classes
        test_classes = np.unique(target_labels[test_idx])
        if len(test_classes) < 2:
            continue
        
        try:
            clf = LogisticRegressionCV(
                max_iter=1000,
                class_weight='balanced',
                scoring='balanced_accuracy',
                cv=3  # inner CV for regularization
            )
            clf.fit(hidden_scaled[train_idx], target_labels[train_idx])
            preds = clf.predict(hidden_scaled[test_idx])
            acc = balanced_accuracy_score(target_labels[test_idx], preds)
            acc_trained_folds.append(acc)
        except Exception as e:
            print(f"  LogReg fold failed: {e}")
            continue
    
    acc_trained = float(np.mean(acc_trained_folds)) if acc_trained_folds else 0.0
    
    # ── Untrained model ──
    acc_untrained_folds = []
    
    for train_idx, test_idx in kf.split(untrained_scaled):
        test_classes = np.unique(target_labels[test_idx])
        if len(test_classes) < 2:
            continue
        
        try:
            clf = LogisticRegressionCV(
                max_iter=1000,
                class_weight='balanced',
                scoring='balanced_accuracy',
                cv=3
            )
            clf.fit(untrained_scaled[train_idx], target_labels[train_idx])
            preds = clf.predict(untrained_scaled[test_idx])
            acc = balanced_accuracy_score(target_labels[test_idx], preds)
            acc_untrained_folds.append(acc)
        except Exception as e:
            continue
    
    acc_untrained = float(np.mean(acc_untrained_folds)) if acc_untrained_folds else 0.0
    
    delta = acc_trained - acc_untrained
    
    return {
        'acc_trained': acc_trained,
        'acc_untrained': acc_untrained,
        'delta_accuracy': delta,
        'chance_balanced': chance_balanced,
        'majority_fraction': majority_fraction,
        'n_classes': n_classes,
        'n_valid_folds_trained': len(acc_trained_folds),
        'n_valid_folds_untrained': len(acc_untrained_folds),
        'valid': acc_trained > chance_balanced,
        'reason': 'OK' if acc_trained > chance_balanced else 'trained below chance',
        'class_distribution': class_dist,
    }
```

### A.3 Probe Target Hygiene

Before probing, validate every target signal. Create this function and
call it on every target before passing to the probe:

```python
def validate_probe_target(name: str, target: np.ndarray) -> Dict:
    """
    Check a probe target for pathological properties.
    Returns validation report and whether to skip this target.
    
    SKIP conditions:
    - Near-constant (std < 1e-6 after z-scoring)
    - >95% identical values (trivially dominated by one value)
    - Contains NaN or Inf
    """
    report = {
        'name': name,
        'length': len(target),
        'dtype': str(target.dtype),
        'has_nan': bool(np.any(np.isnan(target))),
        'has_inf': bool(np.any(np.isinf(target))),
        'mean': float(np.nanmean(target)),
        'std': float(np.nanstd(target)),
        'min': float(np.nanmin(target)),
        'max': float(np.nanmax(target)),
        'pct_01': float(np.nanpercentile(target, 1)),
        'pct_25': float(np.nanpercentile(target, 25)),
        'pct_50': float(np.nanpercentile(target, 50)),
        'pct_75': float(np.nanpercentile(target, 75)),
        'pct_99': float(np.nanpercentile(target, 99)),
    }
    
    # Check for near-constant
    if report['std'] < 1e-6:
        report['skip'] = True
        report['skip_reason'] = 'near-constant'
        return report
    
    # Check for dominance by one value
    unique_vals, counts = np.unique(target, return_counts=True)
    max_frac = counts.max() / len(target)
    report['max_value_fraction'] = float(max_frac)
    report['n_unique_values'] = len(unique_vals)
    
    if max_frac > 0.95:
        report['skip'] = True
        report['skip_reason'] = f'{max_frac:.1%} identical values'
        return report
    
    # Check for NaN/Inf
    if report['has_nan'] or report['has_inf']:
        report['skip'] = True
        report['skip_reason'] = 'contains NaN or Inf'
        return report
    
    report['skip'] = False
    report['skip_reason'] = 'OK'
    return report
```

### A.4 Valence Target

Drop valence as a separate probe target. In the current implementation,
valence = -1 × arousal, providing zero independent information.
Until a genuine valence annotation is sourced (behavioral ratings or
published annotations for this specific film clip), valence must not
appear in the results.

### A.5 Probe Target List (Final, Clean)

After applying all fixes, the probe targets for each patient are:

```
CONTINUOUS TARGETS (use sanitized_ridge_delta_r2):
  1. theta_power        — from hippocampal+amygdala LFP, 4-8 Hz band
  2. gamma_power        — from hippocampal+amygdala LFP, 30-80 Hz band
  3. arousal            — from pupil diameter (primary) blended with 
                          scene-cut density (secondary)
  4. threat_pe          — derivative of arousal (predicted vs observed)
  5. population_synchrony — pairwise spike rate correlations
  6. temporal_stability — population vector autocorrelation
  7. event_boundaries   — smoothed impulse at scene cuts
  8. encoding_success   — subsequent memory signal from recognition test

CATEGORICAL TARGET (use classification_delta_accuracy):
  9. emotion_category   — 4-class: neutral/fear/surprise/relief
                          Report class distribution per patient.
                          If any class <5% of time bins in test set,
                          note this in the report.

DROPPED:
  - valence            — identical to -1 × arousal, no independent info
  - face_presence      — ONLY if annotation quality is verified.
                          The pickle file (short_faceannots.pkl) has
                          per-frame coordinates — check whether these
                          align reliably with neural timestamps.
                          If alignment is uncertain, drop it.
```

---

## Task B: Scale to All 14 Patients

### B.1 Per-Patient Pipeline

For each of the 14 included patients, execute the following steps
in order. The entire pipeline for one patient should be a single
script execution.

```
STEP 1: PREPROCESS
  - Load NWB file
  - Extract spike times for input regions (Left/Right amygdala, 
    Left/Right hippocampus) and output regions (Left/Right ACC, 
    Left/Right preSMA, Left/Right vmPFC)
  - Region column: electrodes → group_name → .location
  - Get movie epoch from trials table: trial 0 has 
    stim_phase='encoding', start_time and stop_time define the 
    movie window
  - Bin at 20 ms, Gaussian smooth (σ=2 bins = 40 ms)
  - Z-score each neuron independently
  - Create sliding windows: 2000 ms window, 500 ms stride
  - Temporal split: first 60% = train, next 20% = val, last 20% = test
  - Save: preprocessed_data/{patient_id}.npz

STEP 2: IDENTIFY FOCUSED OUTPUT NEURONS
  - Train a quick pilot LSTM (hidden=128, 1 seed, 50 epochs max)
    on ALL output neurons
  - Compute per-neuron CC on validation set
  - Select neurons with CC > 0.3
  - If fewer than 5 neurons pass: lower threshold to 0.2
  - If fewer than 3 neurons pass: EXCLUDE this patient 
    (insufficient limbic→prefrontal drive for reliable probing)
  - Record which neurons were selected and their pilot CC values
  - Save: results/{patient_id}/focused_neurons.json

STEP 3: COMPUTE PROBE TARGETS
  - Extract LFP from limbic channels specifically:
    Use electrode table to find channels where electrode group 
    location contains 'amygdala' or 'hippocampus'
    Map electrode indices to LFP data matrix columns
    For LFP_macro: column_index = electrode_index - min(all_electrode_indices)
    Verify: selected channels should number 8-32 depending on patient
  - Bandpass filter: theta 4-8 Hz, gamma 30-80 Hz (4th order Butterworth)
  - Hilbert transform for instantaneous amplitude
  - Downsample power envelopes to match 20 ms binning
  - Extract pupil diameter from processing['behavior']['PupilTracking']
    Access: pt.time_series['TimeSeries'] (NOT iteration)
    Timestamps: reconstruct from starting_time + np.arange(N) / rate
    Blinks: values ≤ 0 or below 5th percentile → interpolate
    Detrend: high-pass filter at 0.01 Hz to remove slow drift
    Z-score, downsample to 20 ms bins
    Blend with scene-cut density: 0.7 × pupil + 0.3 × cuts
  - Compute all 9 targets (8 continuous + 1 categorical)
  - Run validate_probe_target() on each, print report
  - Skip any target that fails validation
  - Save: probe_targets/{patient_id}.npz

STEP 4: TRAIN SURROGATES (10 seeds)
  - For each seed 0-9:
    - Create LSTM: input_proj(n_input, 128) → LSTM(128, 2 layers) 
      → output_proj(128 → 64 → n_focused_output)
    - Train with Adam lr=1e-3, MSE loss, patience=20, max 200 epochs
    - Gradient clipping at 1.0
    - Record: val_loss, per-neuron CC, mean CC
    - Save checkpoint: results/{patient_id}/seed_{s}/model.pt
  - Also create 10 UNTRAINED models (same architecture, different 
    random init, no training) for the baseline comparison.
    Use seeds 100-109 for untrained (separate from trained seeds).
  - Report: mean CC ± std across 10 seeds
  - QUALITY GATE: if mean CC < 0.3, flag patient as LOW QUALITY
    (still run probing but mark all results as unreliable)

STEP 5: RIDGE ΔR² PROBING (per seed)
  - For each of the 10 trained seeds:
    - Extract hidden states on TEST data: 
      model.eval(), forward with return_hidden=True
    - Extract hidden states from corresponding untrained model
    - Flatten: (n_windows, T_per_window, hidden) → (N, hidden)
    - For each CONTINUOUS probe target:
      - Slice target to match test windows
      - Call sanitized_ridge_delta_r2()
      - Record: r2_trained, r2_untrained, delta_r2, valid flag
    - For emotion_category:
      - Slice to match test windows
      - Call classification_delta_accuracy()
      - Record: acc_trained, acc_untrained, delta_accuracy, 
        class_distribution, valid flag
  - Aggregate across seeds:
    - Mean ΔR² ± std across 10 seeds
    - Seed consistency: in how many seeds is ΔR² > 0.05?
    - Only report targets where ≥7/10 seeds agree on sign
  - Save: results/{patient_id}/probing_results.json

STEP 6: PATIENT REPORT
  - Save a JSON report with all results:
    {
      "patient_id": str,
      "n_input_neurons": int,
      "n_output_neurons_total": int,
      "n_focused_neurons": int,
      "focused_neuron_ids": list,
      "mean_cc": float,
      "cc_std": float,
      "quality_flag": "OK" or "LOW_QUALITY",
      "probe_results": {
        "target_name": {
          "method": "ridge" or "classification",
          "metric_trained": float,
          "metric_untrained": float,
          "delta": float,
          "delta_std_across_seeds": float,
          "seed_consistency": "8/10",
          "valid": bool,
          "status": "STRONG" or "CANDIDATE" or "zombie" or "INVALID"
        }
      },
      "target_validation": {
        "target_name": {validation report from validate_probe_target}
      },
      "excluded_targets": list of skipped targets with reasons
    }
  - Save: results/{patient_id}/patient_report.json
```

### B.2 Status Classification

For each probe target in the final report, classify as:

```
STRONG:    ΔR² > 0.15 AND valid=True AND seed_consistency ≥ 7/10
CANDIDATE: ΔR² > 0.05 AND valid=True AND seed_consistency ≥ 5/10
zombie:    ΔR² ≤ 0.05 OR valid=False OR seed_consistency < 5/10
INVALID:   target failed validation (skipped)

For classification targets (emotion_category):
STRONG:    ΔAccuracy > 0.10 AND acc_trained > chance AND consistency ≥ 7/10
CANDIDATE: ΔAccuracy > 0.03 AND acc_trained > chance AND consistency ≥ 5/10
zombie:    otherwise
```

### B.3 Cross-Patient Summary

After all 14 patients complete, build the summary table:

```python
def build_cross_patient_summary(results_dir: str) -> Dict:
    """
    Aggregate per-patient probing results into cross-patient table.
    
    For each probe target, report:
    1. Fraction of patients where STRONG
    2. Fraction of patients where CANDIDATE or better
    3. Fraction of patients where zombie
    4. Fraction of patients where INVALID (target skipped)
    5. Mean ΔR² across patients (only valid patients)
    6. Whether the target is UNIVERSALLY learned (>50% patients CANDIDATE+)
    """
    import json
    from pathlib import Path
    
    results_path = Path(results_dir)
    patient_reports = []
    
    for report_file in sorted(results_path.glob("*/patient_report.json")):
        with open(report_file) as f:
            patient_reports.append(json.load(f))
    
    n_patients = len(patient_reports)
    print(f"Found {n_patients} patient reports")
    
    # Collect all target names
    all_targets = set()
    for r in patient_reports:
        all_targets.update(r.get('probe_results', {}).keys())
    
    summary = {}
    for target in sorted(all_targets):
        strong = 0
        candidate = 0
        zombie = 0
        invalid = 0
        deltas = []
        low_quality_patients = 0
        
        for r in patient_reports:
            if r.get('quality_flag') == 'LOW_QUALITY':
                low_quality_patients += 1
            
            if target in r.get('excluded_targets', []):
                invalid += 1
                continue
            
            probe = r.get('probe_results', {}).get(target, {})
            status = probe.get('status', 'zombie')
            
            if status == 'STRONG':
                strong += 1
            elif status == 'CANDIDATE':
                candidate += 1
            elif status == 'INVALID':
                invalid += 1
            else:
                zombie += 1
            
            if probe.get('valid', False):
                deltas.append(probe.get('delta', 0.0))
        
        n_valid = len(deltas)
        summary[target] = {
            'n_patients': n_patients,
            'n_valid': n_valid,
            'n_strong': strong,
            'n_candidate_or_better': strong + candidate,
            'n_zombie': zombie,
            'n_invalid': invalid,
            'fraction_learned': (strong + candidate) / max(n_valid, 1),
            'mean_delta': float(np.mean(deltas)) if deltas else None,
            'std_delta': float(np.std(deltas)) if deltas else None,
            'universally_learned': (strong + candidate) / max(n_valid, 1) > 0.5,
        }
    
    # Print summary table
    print(f"\n{'='*90}")
    print(f"CROSS-PATIENT PROBING SUMMARY — CIRCUIT 5 (n={n_patients})")
    print(f"{'='*90}")
    print(f"{'Target':<25} {'Strong':>6} {'Cand+':>6} {'Zombie':>6} "
          f"{'Invalid':>7} {'Mean ΔR²':>10} {'Learned?':>10}")
    print(f"{'-'*90}")
    
    for target in sorted(summary.keys(), 
                          key=lambda t: summary[t].get('mean_delta') or -999, 
                          reverse=True):
        s = summary[target]
        mean_d = f"{s['mean_delta']:+.3f}" if s['mean_delta'] is not None else "N/A"
        learned = "YES" if s['universally_learned'] else "no"
        print(f"{target:<25} {s['n_strong']:>6} {s['n_candidate_or_better']:>6} "
              f"{s['n_zombie']:>6} {s['n_invalid']:>7} {mean_d:>10} {learned:>10}")
    
    print(f"\nLow quality patients (CC < 0.3): {low_quality_patients}")
    
    return summary
```

### B.4 The Output Table for the Paper

The final cross-patient table should look like this:

```
┌────────────────────────┬─────────┬───────────┬──────────┬──────────────┐
│ Probe Target           │ Learned │ Mean ΔR²  │ Status   │ Cross-Domain │
│                        │ (n/14)  │ (±std)    │          │ Match?       │
├────────────────────────┼─────────┼───────────┼──────────┼──────────────┤
│ encoding_success       │ ??/14   │ +0.???    │ ???      │ → Circuit 4  │
│ theta_power            │ ??/14   │ +0.???    │ ???      │ → Circuits 2-4│
│ gamma_power            │ ??/14   │ +0.???    │ ???      │ → Circuit 2  │
│ threat_pe              │ ??/14   │ +0.???    │ ???      │ NEW          │
│ arousal                │ ??/14   │ +0.???    │ ???      │ NEW          │
│ temporal_stability     │ ??/14   │ +0.???    │ ???      │ → Circuits 3-4│
│ population_synchrony   │ ??/14   │ +0.???    │ ???      │ → Circuit 4  │
│ event_boundaries       │ ??/14   │ +0.???    │ ???      │ NEW          │
│ emotion_category       │ ??/14   │ +0.??? Δa │ ???      │ NEW          │
└────────────────────────┴─────────┴───────────┴──────────┴──────────────┘

"Learned" = STRONG or CANDIDATE in that patient
Variables learned in >50% of patients = "universally learned"
Variables learned in <20% = "not reliably learned"
```

### B.5 Interpretation Guide

After the cross-patient summary is built, the narrative depends on what
pattern emerges:

```
IF theta_power is learned in >50% of patients:
  → Theta is universally represented in the limbic→prefrontal
    transformation, consistent with Circuits 2-4.
  → However, since ablation showed it is NOT mandatory (causal),
    this is a "universal learned zombie" — the transformation
    consistently develops theta representations that are
    computationally epiphenomenal.
  → This is DIFFERENT from Circuits 2-3 where theta was mandatory.
  → Interpretation: passive processing learns the same representations
    as active processing but doesn't depend on them.

IF encoding_success is learned in >50% of patients:
  → The limbic→prefrontal transformation inherently carries
    memory-predictive information across patients.
  → This bridges directly to Circuit 4 (human WM) where
    encoding-related signals were also present.

IF emotion-specific targets (arousal, threat_pe, emotion_category)
are learned in >50% of patients:
  → The transformation learns domain-specific representations
    not seen in Circuits 2-4.
  → Even as zombies, this shows the transformation LEARNS emotion
    — it develops emotion-correlated internal states.

IF nothing is consistently learned across patients:
  → The limbic→prefrontal transformation during passive movie
    watching is too variable across individuals for any universal
    computational intermediate — extending the Circuit 4 finding
    of individual variability to an even more variable regime.
```

---

## Execution Instructions

### Step 1: Create the pipeline script

Build a single Python script `run_all_patients.py` that:
1. Reads the patient inventory from patient_inventory.json
2. For each included patient (14 patients), runs Steps 1-6 above
3. Saves per-patient results
4. After all patients, runs build_cross_patient_summary()

The script should:
- Handle errors gracefully (if one patient fails, continue to next)
- Print progress: "Processing patient X/14: sub-CSNN"
- Print per-patient summary after each patient completes
- Print cross-patient summary table at the end
- Estimate total runtime after first patient completes

### Step 2: Execute

Run the script. Expected runtime: ~1 minute per patient on GPU 
(based on pilot: 53 seconds), so ~15 minutes total for 14 patients.

### Step 3: Review and Save

Save final outputs:
- results/cross_patient_summary.json
- results/cross_patient_table.txt (formatted text table)
- Per-patient: results/{patient_id}/patient_report.json

---

## Critical Reminders

1. NEVER report ΔR² values outside [-2, +2]. If they appear, the 
   pipeline has a bug. Stop and debug.

2. NEVER use Ridge regression on categorical targets. Emotion_category 
   is the only categorical target — it gets logistic regression.

3. NEVER interpret results from patients with mean CC < 0.3 as 
   reliable. Flag them, run them, but mark all their results as 
   LOW_QUALITY in the summary.

4. DO NOT run ablation. The bottleneck sweep proved the ablation 
   methodology does not work for this LSTM architecture. The 
   publishable finding from Circuit 5 is the PROBING pattern 
   (which variables are learned), not the ABLATION pattern 
   (which are mandatory).

5. Valence is dropped. Do not compute or report valence until a 
   genuine valence annotation (independent of arousal) is available.

6. LFP channel selection: ALWAYS verify that selected channels are
   from limbic regions (amygdala, hippocampus) by checking electrode
   group location. NEVER grab channels by index position in the 
   data matrix — electrode ordering varies across patients.

7. Pupil extraction: access via 
   nwb.processing['behavior']['PupilTracking'].time_series['TimeSeries']
   Timestamps are reconstructed from starting_time + arange(N)/rate.
   Do NOT call .timestamps[:] on rate-based TimeSeries objects.

8. The focused output approach (predicting only CC > 0.3 neurons) 
   is REQUIRED for every patient. The number of focused neurons will 
   vary across patients — record this in each patient report.
