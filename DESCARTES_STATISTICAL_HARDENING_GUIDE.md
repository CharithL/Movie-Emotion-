# DESCARTES Statistical Hardening — Claude Code Implementation Guide

## Overview

This guide implements rigorous statistical validation methods across all 
DESCARTES circuits. The methods are ordered by priority: methods that 
could INVALIDATE existing results come first (autocorrelation fixes, 
significance testing), followed by methods that add new analytical 
power (causal testing, geometric comparison).

All methods operate on EXISTING saved hidden states and probe targets.
No model retraining is required unless explicitly stated.

## Prerequisites

```bash
pip install pynwb numpy scipy scikit-learn torch matplotlib seaborn
pip install arch          # Stationary Block Bootstrap
pip install nolitsa        # iAAFT surrogates (or custom implementation)
pip install pyvene         # DAS interchange interventions
pip install cebra          # Contrastive embeddings
pip install ripser persim giotto-tda  # Persistent homology
pip install pymc bambi arviz          # Bayesian inference
pip install statsmodels    # Granger causality, time series tools
```

## Data Requirements

For each circuit that has been run, locate:
- Trained model checkpoints: results/{circuit}/{patient}/seed_{s}/model.pt
- Preprocessed data: preprocessed_data/{patient}.npz (X_train, X_val, X_test, Y_test)
- Probe targets: probe_targets/{patient}.npz
- Existing probe results: results/{circuit}/{patient}/probing_results.json

If hidden states were not saved separately, they must be re-extracted:
load model checkpoint → forward pass on test data with return_hidden=True.

---

## PHASE 1: Fix the Foundation (Invalidation Tests)

These methods test whether existing results survive proper statistical 
controls. Run these FIRST. If results don't survive, everything 
downstream changes.

### 1.1 Matched Baselines (10 Untrained per 10 Trained)

**Problem:** Current pipeline compares 10 trained seeds against fewer 
untrained seeds. Untrained networks have structured random projections 
(Glorot/Kaiming), and R²_untrained varies across initializations.

```python
"""
phase1/matched_baselines.py

For each trained model seed, create and cache the EXACT pre-training
initialization as the matched untrained baseline.

CRITICAL: The untrained model for seed_i must use seed_i's initialization
weights BEFORE any training occurred. If you didn't save these, 
reinitialize with the same seed.
"""
import torch
import numpy as np
import json
from pathlib import Path


def create_matched_baselines(model_class, model_kwargs, seeds, 
                              X_test, device='cpu'):
    """
    For each seed, create the untrained model with that seed's 
    initialization and extract hidden states.
    
    Args:
        model_class: e.g., LimbicPrefrontalLSTM
        model_kwargs: dict of constructor args (n_input, n_output, hidden_size, etc.)
        seeds: list of 10 training seeds
        X_test: (n_windows, T, n_input) test data tensor
    
    Returns:
        dict: {seed: untrained_hidden_states} where each is (N, hidden_size)
    """
    untrained_states = {}
    
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = model_class(**model_kwargs).to(device)
        # Do NOT train — this IS the untrained baseline for this seed
        
        model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X_test).to(device)
            _, hidden = model(X_t, return_hidden=True)
        
        # Flatten: (n_windows, T, hidden) → (N, hidden)
        h = hidden.cpu().numpy().reshape(-1, hidden.shape[-1])
        untrained_states[seed] = h
        
        print(f"  Seed {seed}: untrained hidden shape = {h.shape}")
    
    return untrained_states


def extract_trained_hidden_states(model_class, model_kwargs, 
                                   checkpoint_dir, seeds, 
                                   X_test, device='cpu'):
    """Load each trained checkpoint and extract hidden states."""
    trained_states = {}
    
    for seed in seeds:
        ckpt_path = Path(checkpoint_dir) / f"seed_{seed}" / "model.pt"
        if not ckpt_path.exists():
            # Try alternative naming conventions
            ckpt_path = Path(checkpoint_dir) / f"seed_{seed}" / "best_model.pt"
        
        model = model_class(**model_kwargs).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        
        model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X_test).to(device)
            _, hidden = model(X_t, return_hidden=True)
        
        h = hidden.cpu().numpy().reshape(-1, hidden.shape[-1])
        trained_states[seed] = h
    
    return trained_states


def compute_paired_delta_r2(trained_states, untrained_states, 
                             target, seeds, probe_func):
    """
    Compute ΔR² with MATCHED baselines: each trained seed compared
    only to its own pre-training initialization.
    
    Returns per-seed ΔR² values and aggregated statistics.
    """
    results = []
    
    for seed in seeds:
        h_trained = trained_states[seed]
        h_untrained = untrained_states[seed]
        
        # Align lengths
        min_len = min(len(h_trained), len(h_untrained), len(target))
        
        result = probe_func(
            h_trained[:min_len], 
            target[:min_len], 
            h_untrained[:min_len]
        )
        result['seed'] = seed
        results.append(result)
    
    deltas = [r['delta_r2'] for r in results]
    
    return {
        'per_seed': results,
        'mean_delta': float(np.mean(deltas)),
        'std_delta': float(np.std(deltas)),
        'median_delta': float(np.median(deltas)),
        'n_positive': sum(1 for d in deltas if d > 0),
        'n_seeds': len(seeds),
    }
```

### 1.2 Gap Cross-Validation

**Problem:** Standard temporal KFold leaks information through 
autocorrelation. Adjacent time bins in train and test share signal.

```python
"""
phase1/gap_cross_validation.py

Implement purged/gap cross-validation for time series probing.
The gap between train and test folds must exceed the decorrelation 
time of the slowest probe target.
"""
import numpy as np
from sklearn.linear_model import RidgeCV
from typing import List, Tuple


def estimate_decorrelation_time(signal: np.ndarray, 
                                 threshold: float = 0.05) -> int:
    """
    Estimate the decorrelation time of a signal.
    Returns the lag (in samples) where autocorrelation drops below threshold.
    """
    from statsmodels.tsa.stattools import acf
    
    # Compute autocorrelation up to lag = len/4
    max_lag = min(len(signal) // 4, 500)
    autocorr = acf(signal, nlags=max_lag, fft=True)
    
    # Find first lag where |acf| < threshold
    for lag in range(1, len(autocorr)):
        if abs(autocorr[lag]) < threshold:
            return lag
    
    # If never drops below threshold, return max_lag
    return max_lag


def gap_cross_validation(hidden_states: np.ndarray,
                          target: np.ndarray,
                          n_folds: int = 5,
                          gap_samples: int = None,
                          alphas: np.ndarray = np.logspace(-3, 3, 20),
                          max_abs_r2: float = 1.0) -> dict:
    """
    Gap (purged) cross-validation for Ridge regression probing.
    
    Temporal folds with buffer zones between train and test sets.
    The gap prevents autocorrelation from leaking information.
    
    Args:
        hidden_states: (T, hidden_size)
        target: (T,) — must be z-scored and clipped to ±3σ before calling
        n_folds: number of temporal folds
        gap_samples: size of gap on each side of test fold.
                     If None, auto-estimated from target autocorrelation.
        alphas: Ridge regularization search
        max_abs_r2: clamp R² to this range
    
    Returns:
        dict with r2, fold details, gap size used
    """
    T = len(target)
    assert len(hidden_states) == T
    
    # Auto-estimate gap from target autocorrelation
    if gap_samples is None:
        gap_samples = estimate_decorrelation_time(target)
        print(f"  Auto-estimated gap: {gap_samples} samples "
              f"({gap_samples * 0.02:.2f}s at 20ms bins)")
    
    # Create temporal folds with gaps
    fold_size = T // n_folds
    r2_folds = []
    fold_details = []
    
    for fold in range(n_folds):
        test_start = fold * fold_size
        test_end = test_start + fold_size
        
        # Training data: everything OUTSIDE test + gap
        train_mask = np.ones(T, dtype=bool)
        # Remove test fold
        train_mask[test_start:test_end] = False
        # Remove gap before test
        gap_before_start = max(0, test_start - gap_samples)
        train_mask[gap_before_start:test_start] = False
        # Remove gap after test
        gap_after_end = min(T, test_end + gap_samples)
        train_mask[test_end:gap_after_end] = False
        
        train_idx = np.where(train_mask)[0]
        test_idx = np.arange(test_start, test_end)
        
        if len(train_idx) < 50:
            # Not enough training data after gap removal
            continue
        
        ridge = RidgeCV(alphas=alphas)
        ridge.fit(hidden_states[train_idx], target[train_idx])
        r2 = ridge.score(hidden_states[test_idx], target[test_idx])
        r2 = np.clip(r2, -max_abs_r2, max_abs_r2)
        r2_folds.append(r2)
        
        fold_details.append({
            'fold': fold,
            'n_train': len(train_idx),
            'n_test': len(test_idx),
            'gap_before': test_start - gap_before_start,
            'gap_after': gap_after_end - test_end,
            'r2': float(r2),
        })
    
    return {
        'r2_mean': float(np.mean(r2_folds)) if r2_folds else 0.0,
        'r2_std': float(np.std(r2_folds)) if r2_folds else 0.0,
        'r2_folds': [float(r) for r in r2_folds],
        'n_valid_folds': len(r2_folds),
        'gap_samples': gap_samples,
        'fold_details': fold_details,
    }
```

### 1.3 iAAFT Phase-Randomized Significance Testing

**Problem:** No p-values for any ΔR². Current thresholds (0.05, 0.15) 
are arbitrary. Need formal null distribution.

```python
"""
phase1/iaaft_significance.py

Generate phase-randomized surrogates that preserve the autocorrelation
spectrum of the target, then compute null distribution of ΔR².
"""
import numpy as np
from typing import Tuple


def iaaft_surrogate(signal: np.ndarray, 
                     n_iterations: int = 100) -> np.ndarray:
    """
    Iterated Amplitude Adjusted Fourier Transform surrogate.
    
    Preserves:
    - Exact amplitude distribution (histogram) of original signal
    - Power spectrum (and thus autocorrelation function)
    
    Destroys:
    - Phase relationships (i.e., the specific temporal ordering 
      that relates this signal to the LSTM hidden states)
    
    This is the gold standard for testing whether a correlation 
    is driven by shared spectral properties vs genuine relationship.
    """
    n = len(signal)
    
    # Store original amplitude spectrum and sorted values
    original_spectrum = np.fft.rfft(signal)
    original_amplitudes = np.abs(original_spectrum)
    sorted_values = np.sort(signal)
    
    # Initialize with random shuffle (preserves amplitude distribution)
    surrogate = signal.copy()
    np.random.shuffle(surrogate)
    
    for iteration in range(n_iterations):
        # Step 1: Replace spectrum amplitudes with original
        surr_spectrum = np.fft.rfft(surrogate)
        surr_phases = np.angle(surr_spectrum)
        adjusted_spectrum = original_amplitudes * np.exp(1j * surr_phases)
        surrogate = np.fft.irfft(adjusted_spectrum, n=n)
        
        # Step 2: Rank-order to match original amplitude distribution
        rank_order = np.argsort(np.argsort(surrogate))
        surrogate = sorted_values[rank_order]
    
    return surrogate


def iaaft_significance_test(hidden_trained: np.ndarray,
                             hidden_untrained: np.ndarray,
                             target: np.ndarray,
                             probe_func,
                             n_surrogates: int = 1000,
                             gap_samples: int = None) -> dict:
    """
    Compute p-value for ΔR² using iAAFT phase-randomized null distribution.
    
    For each surrogate:
    1. Generate iAAFT surrogate of the target
    2. Probe trained hidden states against surrogate target
    3. Probe untrained hidden states against surrogate target
    4. Compute surrogate ΔR²
    
    The p-value is the fraction of surrogate ΔR² values that exceed 
    the observed ΔR².
    
    Args:
        hidden_trained: (T, hidden_size)
        hidden_untrained: (T, hidden_size)
        target: (T,) original target (z-scored, clipped)
        probe_func: function(hidden, target, hidden_untrained) → dict with 'delta_r2'
        n_surrogates: number of phase-randomized surrogates
        gap_samples: for gap CV (passed to probe_func if it uses gap CV)
    
    Returns:
        dict with observed_delta, null_distribution, p_value, significant
    """
    # Observed ΔR²
    observed = probe_func(hidden_trained, target, hidden_untrained)
    observed_delta = observed['delta_r2']
    
    # Null distribution
    null_deltas = []
    
    for i in range(n_surrogates):
        # Generate surrogate target
        surrogate_target = iaaft_surrogate(target)
        
        # Probe against surrogate
        surr_result = probe_func(hidden_trained, surrogate_target, 
                                  hidden_untrained)
        null_deltas.append(surr_result['delta_r2'])
        
        if (i + 1) % 100 == 0:
            print(f"    Surrogate {i+1}/{n_surrogates}")
    
    null_deltas = np.array(null_deltas)
    
    # One-sided p-value: probability of observing ΔR² ≥ observed under null
    p_value = float(np.mean(null_deltas >= observed_delta))
    
    # Also compute the z-score against the null
    null_mean = np.mean(null_deltas)
    null_std = np.std(null_deltas)
    z_score = (observed_delta - null_mean) / max(null_std, 1e-10)
    
    return {
        'observed_delta_r2': float(observed_delta),
        'null_mean': float(null_mean),
        'null_std': float(null_std),
        'null_percentiles': {
            '5': float(np.percentile(null_deltas, 5)),
            '50': float(np.percentile(null_deltas, 50)),
            '95': float(np.percentile(null_deltas, 95)),
            '99': float(np.percentile(null_deltas, 99)),
        },
        'p_value': p_value,
        'z_score': float(z_score),
        'significant_005': p_value < 0.05,
        'significant_001': p_value < 0.01,
        'n_surrogates': n_surrogates,
    }
```

### 1.4 Input Decodability Control

**Problem:** If a biological variable is equally decodable from the raw 
input as from the hidden states, the LSTM added no representational value.

```python
"""
phase1/input_decodability.py

Test whether biological variables are already decodable from the raw 
input (amygdala + hippocampus firing rates) without any LSTM processing.
"""

def input_vs_hidden_comparison(X_test: np.ndarray,
                                hidden_trained: np.ndarray,
                                target: np.ndarray,
                                gap_samples: int = None) -> dict:
    """
    Compare probe accuracy on raw input vs trained hidden states.
    
    If R²_input ≈ R²_hidden, the LSTM didn't add value for this target.
    The target was already linearly available in the input.
    
    Args:
        X_test: (n_windows, T, n_input) raw input data
        hidden_trained: (N, hidden_size) flattened hidden states
        target: (N,) probe target
    
    Returns:
        dict comparing input and hidden probing
    """
    # Flatten input to match hidden states
    X_flat = X_test.reshape(-1, X_test.shape[-1])
    
    min_len = min(len(X_flat), len(hidden_trained), len(target))
    X_flat = X_flat[:min_len]
    h = hidden_trained[:min_len]
    t = target[:min_len]
    
    # Probe from raw input
    input_result = gap_cross_validation(X_flat, t, gap_samples=gap_samples)
    
    # Probe from hidden states
    hidden_result = gap_cross_validation(h, t, gap_samples=gap_samples)
    
    # Compute added value
    added_value = hidden_result['r2_mean'] - input_result['r2_mean']
    
    return {
        'r2_input': input_result['r2_mean'],
        'r2_hidden': hidden_result['r2_mean'],
        'added_value': float(added_value),
        'input_sufficient': input_result['r2_mean'] > 0 and added_value < 0.02,
        'reason': ('LSTM adds no value — target already in input' 
                   if input_result['r2_mean'] > 0 and added_value < 0.02 
                   else 'LSTM adds representational value'),
    }
```

---

## PHASE 2: LSTM-Specific Probing

Methods that exploit the internal structure of LSTMs specifically.

### 2.1 Gate-Specific Probing

**Problem:** All probing targets h_t (hidden state). The LSTM has 4 gates 
and a cell state. Biological variables may live in the forget gate or 
cell state, invisible to h_t probing.

```python
"""
phase2/gate_probing.py

Extract and probe all LSTM internal components separately.
"""
import torch
import torch.nn as nn
import numpy as np


class LSTMWithGateAccess(nn.Module):
    """
    Wrapper that exposes all LSTM gate activations.
    
    Standard LSTM forward pass computes:
        i_t = σ(W_ii x_t + W_hi h_{t-1} + b_i)   # input gate
        f_t = σ(W_if x_t + W_hf h_{t-1} + b_f)   # forget gate
        g_t = tanh(W_ig x_t + W_hg h_{t-1} + b_g) # candidate cell
        o_t = σ(W_io x_t + W_ho h_{t-1} + b_o)    # output gate
        c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t           # cell state
        h_t = o_t ⊙ tanh(c_t)                       # hidden state
    
    h_t is what gets probed normally. But:
    - c_t carries long-term memory (biological variables that persist)
    - f_t controls what persists (if f_t ≈ 1 for theta-related dims, 
      the model actively maintains theta)
    - o_t controls what gets exposed to output (if o_t ≈ 0 for 
      theta dims, theta is stored but not used — a zombie mechanism)
    """
    
    def __init__(self, original_model):
        """Wrap an existing trained LSTM model."""
        super().__init__()
        self.model = original_model
    
    def forward_with_gates(self, x: torch.Tensor):
        """
        Custom forward pass that records all gate activations.
        
        Returns:
            y_pred: (batch, time, n_output)
            components: dict of (batch, time, hidden_size) tensors
        """
        batch, time, n_input = x.shape
        hidden_size = self.model.hidden_size
        n_layers = self.model.lstm.num_layers
        device = x.device
        
        # Project input
        projected = self.model.input_proj(x)
        
        # Manual LSTM forward pass to capture gates
        # Initialize states
        h = torch.zeros(n_layers, batch, hidden_size, device=device)
        c = torch.zeros(n_layers, batch, hidden_size, device=device)
        
        # Storage for gate activations (last layer only)
        all_h = []
        all_c = []
        all_i = []
        all_f = []
        all_g = []
        all_o = []
        
        for t in range(time):
            inp = projected[:, t:t+1, :]  # (batch, 1, hidden)
            
            # Use the built-in LSTM for one step
            # PyTorch doesn't expose gates directly, so we compute manually
            # for the last layer using the weight matrices
            
            # Run through all layers
            output, (h, c) = self.model.lstm(inp, (h, c))
            
            all_h.append(h[-1].clone())  # last layer hidden
            all_c.append(c[-1].clone())  # last layer cell
            
            # To get gates, we need to manually compute them for last layer
            # Get weights for last layer
            layer = n_layers - 1
            w_ih = getattr(self.model.lstm, f'weight_ih_l{layer}')
            w_hh = getattr(self.model.lstm, f'weight_hh_l{layer}')
            b_ih = getattr(self.model.lstm, f'bias_ih_l{layer}')
            b_hh = getattr(self.model.lstm, f'bias_hh_l{layer}')
            
            # Input to last layer is the output of previous layer
            if layer > 0:
                layer_input = all_h[-1] if t == 0 else output.squeeze(1)
            else:
                layer_input = inp.squeeze(1)
            
            # Compute gates manually
            h_prev = all_h[-2] if len(all_h) > 1 else torch.zeros_like(all_h[-1])
            gates = layer_input @ w_ih.t() + b_ih + h_prev @ w_hh.t() + b_hh
            
            i_gate = torch.sigmoid(gates[:, :hidden_size])
            f_gate = torch.sigmoid(gates[:, hidden_size:2*hidden_size])
            g_gate = torch.tanh(gates[:, 2*hidden_size:3*hidden_size])
            o_gate = torch.sigmoid(gates[:, 3*hidden_size:])
            
            all_i.append(i_gate)
            all_f.append(f_gate)
            all_g.append(g_gate)
            all_o.append(o_gate)
        
        # Stack time dimension
        components = {
            'hidden': torch.stack(all_h, dim=1),   # (batch, time, hidden)
            'cell': torch.stack(all_c, dim=1),
            'input_gate': torch.stack(all_i, dim=1),
            'forget_gate': torch.stack(all_f, dim=1),
            'candidate': torch.stack(all_g, dim=1),
            'output_gate': torch.stack(all_o, dim=1),
        }
        
        # Get output predictions
        h_stack = components['hidden']
        y_pred = self.model.output_proj(h_stack)
        
        return y_pred, components
    
    def extract_all_components(self, X_test, device='cpu'):
        """
        Extract all gate activations on test data, flattened for probing.
        
        Returns:
            dict of {component_name: (N, hidden_size) numpy array}
        """
        self.model.eval()
        self.eval()
        
        with torch.no_grad():
            X_t = torch.FloatTensor(X_test).to(device)
            _, components = self.forward_with_gates(X_t)
        
        result = {}
        for name, tensor in components.items():
            # Flatten: (batch, time, hidden) → (N, hidden)
            result[name] = tensor.cpu().numpy().reshape(-1, tensor.shape[-1])
        
        return result


def probe_all_components(components: dict, 
                          target: np.ndarray,
                          untrained_components: dict,
                          probe_func) -> dict:
    """
    Run probing on each LSTM component separately.
    
    Key interpretation:
    - High ΔR² in cell but low in hidden → stored but not exposed (zombie mechanism)
    - High ΔR² in forget_gate → model actively maintains this variable
    - High ΔR² in output_gate → model selectively gates this variable's exposure
    - High ΔR² in hidden → standard finding (variable is available for output)
    """
    results = {}
    
    for name in components:
        h_trained = components[name]
        h_untrained = untrained_components.get(name)
        
        if h_untrained is None:
            print(f"  Skipping {name}: no untrained baseline")
            continue
        
        min_len = min(len(h_trained), len(h_untrained), len(target))
        
        result = probe_func(
            h_trained[:min_len],
            target[:min_len],
            h_untrained[:min_len]
        )
        result['component'] = name
        results[name] = result
        
        print(f"  {name:15s}: ΔR² = {result['delta_r2']:+.4f}")
    
    return results
```

### 2.2 Temporal Generalization Matrix

**Problem:** Current probing collapses the time dimension. We don't know 
whether representations are transient or sustained within each window.

```python
"""
phase2/temporal_generalization.py

Train a probe at time t, test at time t'. The resulting T×T matrix 
reveals the temporal dynamics of the representation.

King & Dehaene (2014) — standard in cognitive neuroscience.
"""
import numpy as np
from sklearn.linear_model import Ridge


def temporal_generalization_matrix(hidden_states: np.ndarray,
                                    target: np.ndarray,
                                    n_windows: int,
                                    T_per_window: int,
                                    alpha: float = 1.0) -> np.ndarray:
    """
    Compute the temporal generalization matrix for a probe target.
    
    hidden_states are reshaped back to (n_windows, T_per_window, hidden)
    to preserve the within-window time structure.
    
    The matrix entry [t_train, t_test] = R² when the probe is trained 
    at time t_train within the window and tested at time t_test.
    
    Interpretation:
    - Strong diagonal: representation is present but re-encoded at each step
    - Off-diagonal square: representation is sustained (attractor-like)
    - Below-diagonal: representation builds up over time
    - Above-diagonal: representation precedes the target (predictive coding)
    
    Args:
        hidden_states: (N, hidden_size) where N = n_windows * T_per_window
        target: (N,) or (n_windows,) — if per-window, broadcast to all timesteps
        n_windows: number of windows
        T_per_window: number of time bins per window
        alpha: Ridge regularization
    
    Returns:
        gen_matrix: (T_per_window, T_per_window) R² values
    """
    hidden_size = hidden_states.shape[1]
    
    # Reshape to (n_windows, T, hidden)
    H = hidden_states.reshape(n_windows, T_per_window, hidden_size)
    
    # Handle target shape
    if len(target) == n_windows:
        # Per-window target — broadcast to all timesteps
        T = np.repeat(target, T_per_window)
    else:
        T = target[:n_windows * T_per_window]
    
    T = T.reshape(n_windows, T_per_window)
    
    # Split windows into train/test (temporal split of WINDOWS, not time bins)
    n_train = int(0.7 * n_windows)
    H_train = H[:n_train]  # (n_train, T_per_window, hidden)
    H_test = H[n_train:]
    T_train = T[:n_train]
    T_test = T[n_train:]
    
    gen_matrix = np.zeros((T_per_window, T_per_window))
    
    for t_train in range(T_per_window):
        # Train probe at time t_train
        X_tr = H_train[:, t_train, :]  # (n_train, hidden)
        y_tr = T_train[:, t_train]     # (n_train,)
        
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_tr, y_tr)
        
        for t_test in range(T_per_window):
            # Test at time t_test
            X_te = H_test[:, t_test, :]
            y_te = T_test[:, t_test]
            
            r2 = ridge.score(X_te, y_te)
            gen_matrix[t_train, t_test] = np.clip(r2, -1, 1)
    
    return gen_matrix
```

---

## PHASE 3: Causal Inference Without Ablation

### 3.1 Distributed Alignment Search (DAS)

**Problem:** Resample ablation failed because LSTM representations are 
holographic. DAS finds the right SUBSPACE to intervene on.

```python
"""
phase3/das_causal.py

Distributed Alignment Search using pyvene library.
Learns a rotation matrix that isolates a biological variable in the 
hidden state space, then tests causal necessity via interchange 
interventions.

This is the replacement for failed resample ablation.
"""

def run_das_for_target(model, X_test, target, 
                        target_name: str,
                        subspace_dim: int = 8,
                        n_epochs: int = 100,
                        device: str = 'cpu'):
    """
    Run Distributed Alignment Search to test causal necessity.
    
    DAS procedure:
    1. Select pairs of inputs where the target variable differs maximally
       (e.g., high-theta vs low-theta windows)
    2. Learn a rotation matrix R that isolates a subspace of the hidden 
       state correlated with the target
    3. Perform interchange interventions: swap the isolated subspace 
       between high and low pairs
    4. Measure whether the output changes predictably
    
    If swapping the target-aligned subspace changes the output to match 
    what the other condition would produce, the target is CAUSALLY USED.
    
    Args:
        model: trained LSTM model
        X_test: (n_windows, T, n_input) test data
        target: (n_windows,) per-window target values
        target_name: name for reporting
        subspace_dim: dimensionality of the isolated subspace (try 4, 8, 16)
        n_epochs: training epochs for the rotation matrix
        device: torch device
    
    Returns:
        dict with DAS results including interchange intervention accuracy
    """
    try:
        import pyvene as pv
    except ImportError:
        print("pyvene not installed. Install with: pip install pyvene")
        return {'error': 'pyvene not installed'}
    
    import torch
    
    # Sort windows by target value
    sorted_idx = np.argsort(target)
    n = len(sorted_idx)
    
    # Create intervention pairs: high-target vs low-target
    low_idx = sorted_idx[:n // 3]
    high_idx = sorted_idx[-n // 3:]
    
    # For each pair, the hypothesis is:
    # If we take a high-target input, replace the theta-aligned subspace 
    # with the value from a low-target input, the output should change 
    # to look more like what the low-target input would produce.
    
    # Extract hidden states for both groups
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X_test).to(device)
        y_pred, hidden = model(X_t, return_hidden=True)
    
    # Window-averaged hidden states
    h_mean = hidden.mean(dim=1).cpu().numpy()  # (n_windows, hidden_size)
    y_mean = y_pred.mean(dim=1).cpu().numpy()  # (n_windows, n_output)
    
    h_low = h_mean[low_idx]
    h_high = h_mean[high_idx]
    y_low = y_mean[low_idx]
    y_high = y_mean[high_idx]
    
    # Learn rotation matrix via gradient descent
    # The rotation R should satisfy:
    #   R @ h_high ≈ R @ h_low in the first subspace_dim dimensions
    #   implies the output changes from y_high toward y_low
    
    hidden_size = h_mean.shape[1]
    
    # Initialize rotation as identity
    R = torch.eye(hidden_size, requires_grad=True, device=device, 
                   dtype=torch.float32)
    
    optimizer = torch.optim.Adam([R], lr=0.01)
    
    h_low_t = torch.FloatTensor(h_low).to(device)
    h_high_t = torch.FloatTensor(h_high).to(device)
    
    n_pairs = min(len(h_low), len(h_high))
    
    best_loss = float('inf')
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # Orthogonalize R via QR decomposition (soft constraint)
        Q, _ = torch.linalg.qr(R)
        
        # Rotate hidden states
        h_high_rotated = h_high_t[:n_pairs] @ Q.T
        h_low_rotated = h_low_t[:n_pairs] @ Q.T
        
        # Interchange: replace first subspace_dim dimensions 
        # of high with those of low
        h_intervened = h_high_rotated.clone()
        h_intervened[:, :subspace_dim] = h_low_rotated[:, :subspace_dim]
        
        # Rotate back
        h_intervened_original = h_intervened @ Q
        
        # Pass through output layer
        with torch.no_grad():
            y_intervened = model.output_proj(h_intervened_original.unsqueeze(1))
            y_intervened = y_intervened.squeeze(1)
        
        # Loss: intervened output should be closer to y_low than y_high
        y_low_t = torch.FloatTensor(y_low[:n_pairs]).to(device)
        y_high_t = torch.FloatTensor(y_high[:n_pairs]).to(device)
        
        dist_to_low = ((y_intervened - y_low_t) ** 2).mean()
        dist_to_high = ((y_intervened - y_high_t) ** 2).mean()
        
        # We want dist_to_low < dist_to_high
        loss = dist_to_low - dist_to_high + 0.01 * (R - Q).pow(2).sum()
        
        loss.backward()
        optimizer.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
    
    # Evaluate: what fraction of interchange interventions move output 
    # closer to the source condition?
    with torch.no_grad():
        Q, _ = torch.linalg.qr(R)
        
        correct = 0
        total = n_pairs
        
        for i in range(n_pairs):
            h_high_r = (h_high_t[i] @ Q.T).unsqueeze(0)
            h_low_r = (h_low_t[i] @ Q.T).unsqueeze(0)
            
            # Interchange
            h_int = h_high_r.clone()
            h_int[0, :subspace_dim] = h_low_r[0, :subspace_dim]
            h_int_orig = h_int @ Q
            
            y_int = model.output_proj(h_int_orig.unsqueeze(1)).squeeze()
            y_h = torch.FloatTensor(y_high[i]).to(device)
            y_l = torch.FloatTensor(y_low[i]).to(device)
            
            # Did output move toward low condition?
            dist_to_low_after = ((y_int - y_l) ** 2).sum()
            dist_to_high_after = ((y_int - y_h) ** 2).sum()
            
            if dist_to_low_after < dist_to_high_after:
                correct += 1
    
    interchange_accuracy = correct / max(total, 1)
    
    # Interpretation:
    # >0.7 = CAUSALLY USED (swapping subspace predictably changes output)
    # 0.4-0.7 = PARTIALLY CAUSAL (some influence)
    # <0.4 = NOT CAUSAL (subspace is irrelevant to output)
    
    status = 'CAUSAL' if interchange_accuracy > 0.7 else \
             'PARTIAL' if interchange_accuracy > 0.4 else 'NOT_CAUSAL'
    
    return {
        'target': target_name,
        'subspace_dim': subspace_dim,
        'interchange_accuracy': float(interchange_accuracy),
        'n_pairs': n_pairs,
        'status': status,
        'best_loss': float(best_loss),
    }
```

### 3.2 Conditional Mutual Information (Observational Mandatory Test)

**Problem:** Need a formal test for mandatory variables that doesn't 
require any intervention.

```python
"""
phase3/conditional_mi.py

Conditional Mutual Information: I(Hidden; Target | Output)

If I(H;T) > 0 but I(H;T|O) > 0:
    → Hidden contains target info NOT used by output → ZOMBIE

If I(H;T) > 0 but I(H;T|O) ≈ 0:
    → All target info in hidden IS used by output → MANDATORY
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors


def ksg_mi_estimate(X: np.ndarray, Y: np.ndarray, k: int = 5) -> float:
    """
    Kraskov-Stögbauer-Grassberger mutual information estimator.
    Non-parametric, works for continuous variables, handles 
    moderate dimensionality (reduce X if dim > 20 via PCA first).
    
    Args:
        X: (N, d_x) — e.g., hidden states (PCA-reduced to ~10 dims)
        Y: (N, 1) or (N,) — target variable
    
    Returns:
        mi_estimate: float, estimated mutual information in nats
    """
    from scipy.special import digamma
    
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    
    N = len(X)
    d_x = X.shape[1]
    d_y = Y.shape[1]
    
    # Joint space
    XY = np.hstack([X, Y])
    
    # Find k-th neighbor distances in joint space
    nn_joint = NearestNeighbors(n_neighbors=k+1, metric='chebyshev')
    nn_joint.fit(XY)
    distances, _ = nn_joint.kneighbors(XY)
    epsilon = distances[:, -1]  # k-th neighbor distance
    
    # Count neighbors within epsilon in marginal spaces
    nn_x = NearestNeighbors(metric='chebyshev')
    nn_x.fit(X)
    
    nn_y = NearestNeighbors(metric='chebyshev')
    nn_y.fit(Y)
    
    n_x = np.zeros(N)
    n_y = np.zeros(N)
    
    for i in range(N):
        # Count points within epsilon[i] in X-space
        n_x[i] = len(nn_x.radius_neighbors(X[i:i+1], radius=epsilon[i], 
                                              return_distance=False)[0]) - 1
        n_y[i] = len(nn_y.radius_neighbors(Y[i:i+1], radius=epsilon[i],
                                              return_distance=False)[0]) - 1
    
    mi = digamma(k) + digamma(N) - np.mean(digamma(n_x + 1) + digamma(n_y + 1))
    
    return max(mi, 0.0)


def conditional_mi_test(hidden: np.ndarray,
                         target: np.ndarray,
                         output: np.ndarray,
                         n_pca_dims: int = 10,
                         k: int = 5) -> dict:
    """
    Compute I(H;T) and I(H;T|O) to classify variable as mandatory or zombie.
    
    I(H;T|O) = I(H;T,O) - I(H;O)
             ≈ I(H_reduced; [T,O]) - I(H_reduced; O)
    
    where H_reduced is PCA-reduced hidden states.
    
    Args:
        hidden: (N, hidden_size) — trained hidden states
        target: (N,) — biological target
        output: (N, n_output) — model predictions
    
    Returns:
        dict with I(H;T), I(H;T|O), and mandatory/zombie classification
    """
    from sklearn.decomposition import PCA
    
    N = len(hidden)
    
    # PCA reduce hidden states for MI estimation
    pca = PCA(n_components=min(n_pca_dims, hidden.shape[1], N - 1))
    H_reduced = pca.fit_transform(hidden)
    
    # PCA reduce output
    if output.ndim > 1 and output.shape[1] > 5:
        pca_out = PCA(n_components=5)
        O_reduced = pca_out.fit_transform(output)
    else:
        O_reduced = output if output.ndim == 2 else output.reshape(-1, 1)
    
    T = target.reshape(-1, 1)
    
    # I(H; T) — does hidden encode target?
    mi_h_t = ksg_mi_estimate(H_reduced, T, k=k)
    
    # I(H; O) — does hidden encode output? (should be high)
    mi_h_o = ksg_mi_estimate(H_reduced, O_reduced, k=k)
    
    # I(H; [T,O]) — does hidden encode target AND output jointly?
    TO = np.hstack([T, O_reduced])
    mi_h_to = ksg_mi_estimate(H_reduced, TO, k=k)
    
    # I(H; T | O) = I(H; T,O) - I(H; O)
    cmi = mi_h_to - mi_h_o
    
    # Classification
    # If I(H;T) > threshold and I(H;T|O) ≈ 0: MANDATORY
    # If I(H;T) > threshold and I(H;T|O) > threshold: ZOMBIE
    # If I(H;T) ≈ 0: NOT ENCODED
    
    mi_threshold = 0.05  # nats
    
    if mi_h_t < mi_threshold:
        status = 'NOT_ENCODED'
    elif cmi < mi_threshold:
        status = 'MANDATORY (all target info used by output)'
    else:
        status = 'ZOMBIE (target info not transmitted to output)'
    
    return {
        'I_H_T': float(mi_h_t),
        'I_H_O': float(mi_h_o),
        'I_H_TO': float(mi_h_to),
        'I_H_T_given_O': float(cmi),
        'status': status,
        'n_pca_dims': n_pca_dims,
        'k_neighbors': k,
    }
```

---

## PHASE 4: Multiple Comparison Correction

### 4.1 Hierarchical FDR

```python
"""
phase4/hierarchical_fdr.py

Hierarchical False Discovery Rate controlling procedure.
Arranges hypotheses in a tree and only tests children if parent is rejected.

Tree structure for DESCARTES:
Level 0: "Did this patient's LSTM learn ANY biological variable?"
Level 1: "Did it learn OSCILLATORY variables?" / "Did it learn COGNITIVE variables?"
Level 2: Specific targets (theta, gamma, arousal, etc.)
"""
import numpy as np
from statsmodels.stats.multitest import multipletests


def hierarchical_fdr(p_values: dict, 
                      alpha: float = 0.05) -> dict:
    """
    Apply hierarchical FDR to structured hypothesis tree.
    
    Args:
        p_values: nested dict matching the tree structure:
            {
                'oscillatory': {
                    'theta_power': 0.023,
                    'gamma_power': 0.450,
                    'population_synchrony': 0.089,
                },
                'cognitive': {
                    'arousal': 0.340,
                    'threat_pe': 0.012,
                    'encoding_success': 0.067,
                    'event_boundaries': 0.091,
                    'temporal_stability': 0.031,
                },
                'categorical': {
                    'emotion_category': 0.560,
                }
            }
        alpha: global FDR level
    
    Returns:
        dict with same structure, each leaf annotated with 
        'rejected' (True/False) and 'adjusted_p'
    """
    results = {}
    
    # Level 0: aggregate p-value per family (Fisher's method)
    from scipy.stats import combine_pvalues
    
    family_pvals = {}
    for family_name, targets in p_values.items():
        pvals = list(targets.values())
        if len(pvals) > 0:
            _, combined_p = combine_pvalues(pvals, method='fisher')
            family_pvals[family_name] = combined_p
    
    # Level 1: test families
    family_names = list(family_pvals.keys())
    family_p_array = [family_pvals[f] for f in family_names]
    
    if len(family_p_array) > 0:
        rejected_families, adjusted_p_families, _, _ = multipletests(
            family_p_array, alpha=alpha, method='fdr_bh')
    else:
        rejected_families = []
        adjusted_p_families = []
    
    # Level 2: for each rejected family, test individual targets
    for i, family_name in enumerate(family_names):
        family_result = {
            'family_p': float(family_pvals[family_name]),
            'family_adjusted_p': float(adjusted_p_families[i]),
            'family_rejected': bool(rejected_families[i]),
            'targets': {}
        }
        
        if rejected_families[i]:
            # Test individual targets within this family
            target_names = list(p_values[family_name].keys())
            target_pvals = [p_values[family_name][t] for t in target_names]
            
            rejected_targets, adjusted_p_targets, _, _ = multipletests(
                target_pvals, alpha=alpha, method='fdr_bh')
            
            for j, target_name in enumerate(target_names):
                family_result['targets'][target_name] = {
                    'raw_p': float(target_pvals[j]),
                    'adjusted_p': float(adjusted_p_targets[j]),
                    'rejected': bool(rejected_targets[j]),
                    'significant': bool(rejected_targets[j]),
                }
        else:
            # Family not rejected — all children automatically non-significant
            for target_name in p_values[family_name]:
                family_result['targets'][target_name] = {
                    'raw_p': float(p_values[family_name][target_name]),
                    'adjusted_p': 1.0,
                    'rejected': False,
                    'significant': False,
                    'note': 'parent family not rejected',
                }
        
        results[family_name] = family_result
    
    return results
```

---

## PHASE 5: Cross-Domain Comparison

### 5.1 CC-Normalized ΔR²

```python
"""
phase5/normalization.py

Normalize ΔR² by model quality (CC²) for cross-circuit comparison.
"""

def normalize_by_cc(delta_r2: float, cc: float) -> float:
    """
    ΔR²_conditional = ΔR²_raw / CC²
    
    Answers: "Of the biological variance the LSTM captured, 
    what fraction is explained by this specific target?"
    """
    cc_squared = cc ** 2
    if cc_squared < 0.01:
        return float('nan')  # CC too low for meaningful normalization
    return delta_r2 / cc_squared
```

### 5.2 Transfer Probing Across Circuits

```python
"""
phase5/transfer_probing.py

Train a probe on Circuit A's hidden states → target.
Apply (without retraining) to Circuit B's hidden states.
If transfer works, the circuits share representational structure.
"""

def transfer_probe(source_hidden: np.ndarray,
                    source_target: np.ndarray,
                    dest_hidden: np.ndarray,
                    dest_target: np.ndarray,
                    alpha: float = 1.0) -> dict:
    """
    Train Ridge on source circuit, test on destination circuit.
    
    Source and destination hidden states must have the same dimensionality.
    If not, PCA-align both to the same number of components first.
    """
    from sklearn.linear_model import Ridge
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    # Align dimensionality if different
    if source_hidden.shape[1] != dest_hidden.shape[1]:
        n_components = min(source_hidden.shape[1], dest_hidden.shape[1], 32)
        pca_source = PCA(n_components=n_components)
        pca_dest = PCA(n_components=n_components)
        source_hidden = pca_source.fit_transform(source_hidden)
        dest_hidden = pca_dest.fit_transform(dest_hidden)
    
    # Standardize
    scaler = StandardScaler()
    source_hidden = scaler.fit_transform(source_hidden)
    dest_hidden_scaled = scaler.transform(dest_hidden)  # Use source statistics
    
    # Train on source
    ridge = Ridge(alpha=alpha)
    ridge.fit(source_hidden, source_target)
    
    # Test on destination
    r2_transfer = ridge.score(dest_hidden_scaled, dest_target)
    
    # Also compute within-source R² for reference
    r2_within = ridge.score(source_hidden, source_target)
    
    return {
        'r2_within_source': float(r2_within),
        'r2_transfer': float(np.clip(r2_transfer, -1, 1)),
        'transfer_ratio': float(r2_transfer / max(r2_within, 0.01)),
    }
```

---

## PHASE 6: Geometric Comparison (CEBRA + Persistent Homology)

### 6.1 CEBRA Embedding

```python
"""
phase6/cebra_comparison.py

CEBRA embeds hidden states and biological variables into a shared
latent space using contrastive learning. Compare topology.
"""

def cebra_embed_and_compare(hidden_states: np.ndarray,
                              target: np.ndarray,
                              n_components: int = 3,
                              max_iterations: int = 5000) -> dict:
    """
    Embed hidden states using target as auxiliary variable.
    Compare embeddings from trained vs untrained models.
    """
    try:
        import cebra
    except ImportError:
        return {'error': 'cebra not installed'}
    
    model = cebra.CEBRA(
        model_architecture='offset10-model',
        batch_size=512,
        output_dimension=n_components,
        max_iterations=max_iterations,
        verbose=False,
    )
    
    model.fit(hidden_states, target)
    embedding = model.transform(hidden_states)
    
    return {
        'embedding': embedding,
        'loss': float(model.state_dict_['loss'][-1]) if hasattr(model, 'state_dict_') else None,
        'n_components': n_components,
    }
```

### 6.2 Persistent Homology

```python
"""
phase6/persistent_homology.py

Compute topological features of hidden state trajectories.
Compare between high-arousal and low-arousal segments.
"""
import numpy as np


def compute_persistence(hidden_states: np.ndarray,
                         max_dim: int = 1,
                         n_points: int = 500) -> dict:
    """
    Compute persistent homology of hidden state point cloud.
    
    Args:
        hidden_states: (N, hidden_size) — subsample if N is large
        max_dim: maximum homological dimension (1 = loops, 2 = voids)
        n_points: subsample size (ripser is O(n³))
    
    Returns:
        dict with Betti numbers and persistence diagrams
    """
    try:
        from ripser import ripser
        from persim import wasserstein as wasserstein_distance
    except ImportError:
        return {'error': 'ripser/persim not installed'}
    
    # Subsample if too large
    if len(hidden_states) > n_points:
        idx = np.random.choice(len(hidden_states), n_points, replace=False)
        idx.sort()
        points = hidden_states[idx]
    else:
        points = hidden_states
    
    # PCA reduce for computational tractability
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(10, points.shape[1]))
    points_reduced = pca.fit_transform(points)
    
    # Compute persistence
    result = ripser(points_reduced, maxdim=max_dim)
    
    # Extract Betti numbers (count features with persistence > threshold)
    persistence_threshold = 0.1
    betti = {}
    for dim in range(max_dim + 1):
        dgm = result['dgms'][dim]
        # Count features with persistence > threshold
        persistence = dgm[:, 1] - dgm[:, 0]
        betti[f'betti_{dim}'] = int(np.sum(persistence > persistence_threshold))
    
    return {
        'diagrams': result['dgms'],
        'betti_numbers': betti,
        'n_points_used': len(points),
    }


def compare_topology_by_condition(hidden_states: np.ndarray,
                                    condition_labels: np.ndarray,
                                    max_dim: int = 1) -> dict:
    """
    Compare topological structure between conditions.
    
    E.g., compare hidden state topology during high-arousal vs 
    low-arousal movie segments. If topology differs, the LSTM 
    develops distinct dynamical regimes for different states.
    """
    try:
        from persim import wasserstein as wasserstein_distance
    except ImportError:
        return {'error': 'persim not installed'}
    
    # Split by condition
    conditions = np.unique(condition_labels)
    topologies = {}
    
    for cond in conditions:
        mask = condition_labels == cond
        h_cond = hidden_states[mask]
        topologies[str(cond)] = compute_persistence(h_cond, max_dim=max_dim)
    
    # Compute pairwise Wasserstein distances between persistence diagrams
    if len(conditions) >= 2:
        cond_names = [str(c) for c in conditions]
        distances = {}
        for i in range(len(cond_names)):
            for j in range(i+1, len(cond_names)):
                for dim in range(max_dim + 1):
                    dgm_i = topologies[cond_names[i]]['diagrams'][dim]
                    dgm_j = topologies[cond_names[j]]['diagrams'][dim]
                    dist = wasserstein_distance(dgm_i, dgm_j)
                    distances[f'{cond_names[i]}_vs_{cond_names[j]}_dim{dim}'] = float(dist)
    else:
        distances = {}
    
    return {
        'per_condition': {k: v['betti_numbers'] for k, v in topologies.items()},
        'wasserstein_distances': distances,
    }
```

---

## EXECUTION PLAN

### Run Order (total ~2-4 hours on GPU)

```
PHASE 1 — Foundation (~60 min)
  1.1 Create matched baselines for all circuits with saved checkpoints
  1.2 Re-run probing with gap CV on Circuit 5 CS48 (pilot)
  1.3 Run iAAFT significance on CS48 (1000 surrogates × 9 targets)
  1.4 Run input decodability control on CS48

  → CHECKPOINT: Do any existing findings survive?
  → If CS48 temporal_stability has p > 0.05 after iAAFT, it was spurious.
  → If input decodability equals hidden decodability, LSTM adds nothing.

PHASE 2 — LSTM Internals (~30 min)
  2.1 Extract gate activations for CS48 best seed
  2.2 Probe all 6 components (h, c, i, f, g, o) for all targets
  2.3 Compute temporal generalization matrix for top targets

  → CHECKPOINT: Are variables in cell state but not hidden? (zombie mechanism)
  → Does the forget gate track any biological variable? (active maintenance)

PHASE 3 — Causality (~45 min)
  3.1 Run DAS on CS48 for targets that survived Phase 1
  3.2 Run CMI test on CS48 for all targets

  → CHECKPOINT: DAS interchange accuracy > 0.7 for any target?
  → CMI I(H;T|O) ≈ 0 for any target? (mandatory by information theory)

PHASE 4 — Multiple comparisons (~5 min)
  4.1 Collect all p-values from Phase 1.3
  4.2 Apply hierarchical FDR
  4.3 Report which targets survive correction

PHASE 5 — Cross-domain (~30 min)
  5.1 CC-normalize all ΔR² values across circuits
  5.2 Run transfer probing: Circuit 2 → Circuit 5 (does gamma transfer?)
  5.3 Run transfer probing: Circuit 3 → Circuit 5 (does theta transfer?)

PHASE 6 — Geometry (~30 min)
  6.1 CEBRA embedding of CS48 hidden states
  6.2 Persistent homology: compare high vs low arousal topology
  6.3 Compare CS48 topology to Circuit 2 and Circuit 3

  → FINAL: Are the circuits geometrically similar?
```

### Master Script

```python
"""
run_statistical_hardening.py

Master script that executes all phases in order.
Operates on existing saved checkpoints and hidden states.
"""
import json
import numpy as np
from pathlib import Path

# Configuration — adjust paths to your setup
CONFIG = {
    'circuit5_dir': 'results/sub-CS48/',
    'circuit2_dir': 'results/circuit2/',  
    'circuit3_dir': 'results/circuit3/',
    'preprocessed_dir': 'preprocessed_data/',
    'probe_targets_dir': 'probe_targets/',
    'output_dir': 'results/statistical_hardening/',
    'seeds': list(range(10)),
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'n_surrogates': 1000,
}

def main():
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # === PHASE 1 ===
    print("=" * 60)
    print("PHASE 1: Foundation Tests")
    print("=" * 60)
    
    # 1.1 Matched baselines
    # ... (load models, create matched baselines)
    
    # 1.2 Gap CV
    # ... (re-probe with gap CV)
    
    # 1.3 iAAFT significance
    # ... (phase-randomized null distributions)
    
    # 1.4 Input decodability
    # ... (compare input vs hidden probing)
    
    # Save Phase 1 results
    # ...
    
    # === PHASE 2 ===
    print("=" * 60)
    print("PHASE 2: Gate-Specific Probing")
    print("=" * 60)
    # ... 
    
    # === PHASE 3 ===
    print("=" * 60)
    print("PHASE 3: Causal Inference")
    print("=" * 60)
    # ...
    
    # === PHASE 4 ===
    print("=" * 60)
    print("PHASE 4: Multiple Comparison Correction")
    print("=" * 60)
    # ...
    
    # === PHASE 5 ===
    print("=" * 60)
    print("PHASE 5: Cross-Domain Comparison")
    print("=" * 60)
    # ...
    
    # === PHASE 6 ===
    print("=" * 60)
    print("PHASE 6: Geometric Comparison")  
    print("=" * 60)
    # ...
    
    # === FINAL REPORT ===
    print("=" * 60)
    print("GENERATING FINAL REPORT")
    print("=" * 60)
    # Compile all results into a unified report
    

if __name__ == '__main__':
    main()
```

---

## CRITICAL REMINDERS

1. Phase 1 results can INVALIDATE everything. If iAAFT shows p > 0.05 
   for all targets, all current "CANDIDATE" findings are spurious. 
   Do not proceed to later phases until Phase 1 is interpreted.

2. Gate probing (Phase 2) may reveal variables hidden in the cell 
   state that never appeared in h_t probing. If cell state encodes 
   theta but hidden state doesn't, theta is stored but gated out — 
   a specific mechanistic explanation for zombie behavior.

3. DAS (Phase 3.1) is the REPLACEMENT for failed resample ablation. 
   It works on distributed representations because it learns the 
   right rotation, not brute-force dimension ablation.

4. CMI (Phase 3.2) provides the observational mandatory test: 
   I(H;T|O) = 0 means mandatory. This is the formal version of 
   what ablation tried to test.

5. All ΔR² values in the final report must have:
   - A p-value (from iAAFT)
   - A confidence interval (from stationary block bootstrap)
   - An hFDR-corrected significance flag
   - A CC-normalized value for cross-circuit comparison
   
6. Transfer probing (Phase 5.2) directly tests cross-domain 
   universality without any normalization assumptions. If a probe 
   trained on Circuit 2's gamma representation also decodes something 
   meaningful from Circuit 5, the circuits share structure.

7. Persistent homology (Phase 6.2) is the only method that can 
   detect qualitative topological differences (limit cycles, 
   attractors) between emotional states — invisible to any 
   regression-based probe.
