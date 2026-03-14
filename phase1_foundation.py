"""
phase1_foundation.py — DESCARTES Statistical Hardening Phase 1

Fix the Foundation (Invalidation Tests) on Circuit 5 CS48 pilot.

  1.1 Matched Baselines: 10 untrained per 10 trained (same seed init)
  1.2 Gap Cross-Validation: purged temporal folds
  1.3 iAAFT Significance: phase-randomized null distribution for p-values
  1.4 Input Decodability: is the target already in the raw input?

If existing findings don't survive these tests, everything downstream changes.
"""
import numpy as np
import torch
import torch.nn as nn
import json
import time
import sys
from pathlib import Path
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from statsmodels.tsa.stattools import acf
from numba import njit, prange
from joblib import Parallel, delayed

# ================================================================
# CONFIG
# ================================================================
BASE_DIR = Path(__file__).parent
PREPROCESSED_DIR = BASE_DIR / 'preprocessed_data'
PROBE_TARGETS_DIR = BASE_DIR / 'probe_targets'
RESULTS_DIR = BASE_DIR / 'results'
OUTPUT_DIR = RESULTS_DIR / 'statistical_hardening'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model params (must match run_all_patients.py)
HIDDEN_SIZE = 128
N_LAYERS = 2
DROPOUT = 0.1
N_SEEDS = 10
MAX_EPOCHS = 200
BATCH_SIZE = 32
LR = 1e-3
PATIENCE = 20

# Preprocessing params
BIN_MS = 20
WINDOW_MS = 2000
STRIDE_MS = 500

# Phase 1 params
N_SURROGATES = 200  # iAAFT surrogates (guide says 1000; 200 for pilot speed)
N_PROBE_FOLDS = 5
SKIP_TARGETS = {'valence', 'face_presence'}
CATEGORICAL_TARGETS = {'emotion_category'}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ================================================================
# MODEL (identical to run_all_patients.py)
# ================================================================
class LimbicPrefrontalLSTM(nn.Module):
    def __init__(self, n_input, n_output, hidden_size=128,
                 n_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_proj = nn.Linear(n_input, hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size, hidden_size=hidden_size,
            num_layers=n_layers, batch_first=True,
            dropout=dropout if n_layers > 1 else 0)
        mid = max(hidden_size // 2, 4)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, mid), nn.ReLU(),
            nn.Linear(mid, n_output))

    def forward(self, x, return_hidden=False):
        projected = self.input_proj(x)
        lstm_out, _ = self.lstm(projected)
        y_pred = self.output_proj(lstm_out)
        if return_hidden:
            return y_pred, lstm_out
        return y_pred


def set_inference_mode(model):
    model.train(False)


def extract_hidden_flat(model, X):
    set_inference_mode(model)
    with torch.no_grad():
        _, h = model(torch.FloatTensor(X).to(DEVICE), return_hidden=True)
    return h.cpu().numpy().reshape(-1, h.shape[-1])


def train_model(model, X_train, Y_train, X_val, Y_val, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    X_tr = torch.FloatTensor(X_train).to(DEVICE)
    Y_tr = torch.FloatTensor(Y_train).to(DEVICE)
    X_v = torch.FloatTensor(X_val).to(DEVICE)
    Y_v = torch.FloatTensor(Y_val).to(DEVICE)
    best_val = float('inf')
    best_state = None
    pat = 0
    for ep in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(len(X_tr))
        nb = max(1, len(X_tr) // BATCH_SIZE)
        for b in range(nb):
            idx = perm[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            optimizer.zero_grad()
            loss = criterion(model(X_tr[idx]), Y_tr[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        set_inference_mode(model)
        with torch.no_grad():
            vl = criterion(model(X_v), Y_v).item()
        if vl < best_val:
            best_val = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= PATIENCE:
                break
    model.load_state_dict(best_state)
    set_inference_mode(model)
    return model


def align_target_to_windows(target, n_test_windows, test_start_window,
                             window_bins, stride_bins):
    T = len(target)
    aligned = []
    for w in range(n_test_windows):
        s = (test_start_window + w) * stride_bins
        e = s + window_bins
        if e <= T:
            aligned.append(target[s:e])
        else:
            chunk = target[s:min(e, T)]
            if len(chunk) < window_bins:
                chunk = np.pad(chunk, (0, window_bins - len(chunk)), mode='edge')
            aligned.append(chunk)
    return np.concatenate(aligned).astype(np.float32)


# ================================================================
# 1.1 MATCHED BASELINES
# ================================================================
def create_matched_baselines(n_input, n_focused, X_test, seeds):
    """Create untrained models using SAME seed as each trained model."""
    untrained_states = {}
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = LimbicPrefrontalLSTM(n_input, n_focused, HIDDEN_SIZE,
                                      N_LAYERS, DROPOUT).to(DEVICE)
        # Do NOT train — this IS the pre-training initialization for this seed
        h = extract_hidden_flat(model, X_test)
        untrained_states[seed] = h
        print(f"    Seed {seed}: matched untrained hidden shape = {h.shape}")
    return untrained_states


def train_and_extract(n_input, n_focused, X_train, Y_train, X_val, Y_val,
                       X_test, seeds):
    """Train each seed and extract trained hidden states."""
    trained_states = {}
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = LimbicPrefrontalLSTM(n_input, n_focused, HIDDEN_SIZE,
                                      N_LAYERS, DROPOUT)
        model = train_model(model, X_train, Y_train, X_val, Y_val, seed)
        h = extract_hidden_flat(model, X_test)
        trained_states[seed] = h
        print(f"    Seed {seed}: trained hidden shape = {h.shape}")
    return trained_states


# ================================================================
# 1.2 GAP CROSS-VALIDATION
# ================================================================
def estimate_decorrelation_time(signal, threshold=0.05):
    """Lag where autocorrelation drops below threshold."""
    max_lag = min(len(signal) // 4, 500)
    if max_lag < 2:
        return 1
    try:
        autocorr = acf(signal, nlags=max_lag, fft=True)
        for lag in range(1, len(autocorr)):
            if abs(autocorr[lag]) < threshold:
                return lag
        return max_lag
    except Exception:
        return 10  # safe default


def to_window_level(h_flat, target_flat, n_windows, bins_per_window):
    """Convert bin-level data to window-level by averaging within each window.
    h_flat: (n_windows * bins_per_window, hidden_size)
    target_flat: (n_windows * bins_per_window,)
    Returns: h_win (n_windows, hidden_size), t_win (n_windows,)
    """
    N = n_windows * bins_per_window
    h = h_flat[:N].reshape(n_windows, bins_per_window, -1)
    t = target_flat[:N].reshape(n_windows, bins_per_window)
    return h.mean(axis=1), t.mean(axis=1)


def shuffled_cv_r2(hidden_states, target, n_folds=5,
                     alphas=np.logspace(-3, 3, 20), seed=42):
    """Standard shuffled KFold CV R² (the original probing method)."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    r2_folds = []
    for tr, te in kf.split(hidden_states):
        ridge = RidgeCV(alphas=alphas)
        ridge.fit(hidden_states[tr], target[tr])
        r2 = ridge.score(hidden_states[te], target[te])
        r2_folds.append(np.clip(r2, -1.0, 1.0))
    return float(np.mean(r2_folds)), r2_folds


def temporal_cv_r2(hidden_states, target, n_folds=5, gap_windows=2,
                     alphas=np.logspace(-3, 3, 20)):
    """Temporal (gap) CV R² — diagnostic for autocorrelation inflation."""
    N = len(target)
    fold_size = N // n_folds
    r2_folds = []
    for fold in range(n_folds):
        ts = fold * fold_size
        te = ts + fold_size
        mask = np.ones(N, dtype=bool)
        mask[ts:te] = False
        mask[max(0, ts - gap_windows):ts] = False
        mask[te:min(N, te + gap_windows)] = False
        tr_idx = np.where(mask)[0]
        te_idx = np.arange(ts, te)
        if len(tr_idx) < 20:
            continue
        r2 = RidgeCV(alphas=alphas).fit(
            hidden_states[tr_idx], target[tr_idx]).score(
            hidden_states[te_idx], target[te_idx])
        r2_folds.append(np.clip(r2, -1.0, 1.0))
    return float(np.mean(r2_folds)) if r2_folds else 0.0, r2_folds


def window_probe(h_trained, target, h_untrained, n_windows=None,
                   bins_per_window=None):
    """Paired shuffled-CV probing at window level: returns delta_r2.
    Shuffled CV is the standard method; significance comes from iAAFT (1.3).
    """
    if n_windows is not None and bins_per_window is not None:
        h_tr_w, t_w = to_window_level(h_trained, target, n_windows,
                                        bins_per_window)
        h_un_w, _ = to_window_level(h_untrained, target, n_windows,
                                      bins_per_window)
    else:
        h_tr_w, t_w, h_un_w = h_trained, target, h_untrained

    t = t_w.copy().astype(np.float64)
    t_std = np.std(t)
    if t_std < 1e-8:
        return {'delta_r2': 0.0, 'r2_trained': 0.0, 'r2_untrained': 0.0,
                'valid': False, 'reason': 'near-constant'}
    t = (t - np.mean(t)) / t_std
    t = np.clip(t, -3.0, 3.0)

    r2_tr, folds_tr = shuffled_cv_r2(h_tr_w, t)
    r2_un, folds_un = shuffled_cv_r2(h_un_w, t)

    # Also compute temporal CV as diagnostic
    r2_temporal, _ = temporal_cv_r2(h_tr_w, t, gap_windows=2)

    delta = r2_tr - r2_un
    return {
        'delta_r2': float(delta),
        'r2_trained': float(r2_tr),
        'r2_untrained': float(r2_un),
        'r2_temporal_cv': float(r2_temporal),
        'inflation': float(r2_tr - r2_temporal),
        'n_windows': len(t),
        'trained_folds': [float(f) for f in folds_tr],
        'untrained_folds': [float(f) for f in folds_un],
        'valid': r2_tr > 0.0,
    }


# ================================================================
# 1.3 iAAFT SIGNIFICANCE
# ================================================================
@njit(cache=True)
def _rank_reorder(surrogate, sorted_values):
    """Numba-accelerated rank reordering (the inner-loop bottleneck)."""
    n = len(surrogate)
    # argsort of argsort = rank
    order = np.argsort(surrogate)
    rank = np.empty(n, dtype=np.int64)
    for i in range(n):
        rank[order[i]] = i
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = sorted_values[rank[i]]
    return out


def iaaft_surrogate(signal, n_iterations=50):
    """Iterated Amplitude Adjusted Fourier Transform surrogate.
    Preserves amplitude distribution AND power spectrum.
    Destroys phase relationships (the part that matters for real correlations).
    Numba-accelerated rank reordering for ~5x speedup on inner loop.
    """
    n = len(signal)
    signal_f64 = signal.astype(np.float64)
    original_spectrum = np.fft.rfft(signal_f64)
    original_amplitudes = np.abs(original_spectrum)
    sorted_values = np.sort(signal_f64)

    surrogate = signal_f64.copy()
    np.random.shuffle(surrogate)

    for _ in range(n_iterations):
        surr_spectrum = np.fft.rfft(surrogate)
        surr_phases = np.angle(surr_spectrum)
        adjusted_spectrum = original_amplitudes * np.exp(1j * surr_phases)
        surrogate = np.fft.irfft(adjusted_spectrum, n=n)
        surrogate = _rank_reorder(surrogate, sorted_values)

    return surrogate


def _one_surrogate(h_trained, h_untrained, target, rng_seed):
    """Single surrogate: iAAFT target → shuffled CV → ΔR²."""
    np.random.seed(rng_seed)
    surr_target = iaaft_surrogate(target)
    # Z-score surrogate target
    s = surr_target.astype(np.float64)
    s = (s - s.mean()) / max(s.std(), 1e-8)
    s = np.clip(s, -3.0, 3.0)
    r2_tr, _ = shuffled_cv_r2(h_trained, s, seed=rng_seed % (2**31))
    r2_un, _ = shuffled_cv_r2(h_untrained, s, seed=rng_seed % (2**31))
    return r2_tr - r2_un


def iaaft_significance_test(h_trained, h_untrained, target,
                              n_surrogates=200, n_jobs=-1):
    """Compute p-value for ΔR² using iAAFT phase-randomized null.
    All inputs at window level. Uses shuffled CV (same method as main probing).
    Parallelized with joblib.
    """
    # Observed ΔR² using shuffled CV
    observed = window_probe(h_trained, target, h_untrained)
    observed_delta = observed['delta_r2']

    # Warm up numba JIT on first call (compile once)
    _ = iaaft_surrogate(target[:100] if len(target) > 100 else target)

    # Parallel null distribution
    rng_seeds = np.random.randint(0, 2**31, size=n_surrogates)
    print(f"      Running {n_surrogates} surrogates ({n_jobs} jobs)...")
    null_deltas = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_one_surrogate)(h_trained, h_untrained, target, int(s))
        for s in rng_seeds
    )

    null_deltas = np.array(null_deltas)
    p_value = float(np.mean(null_deltas >= observed_delta))
    null_mean = float(np.mean(null_deltas))
    null_std = float(np.std(null_deltas))
    z_score = (observed_delta - null_mean) / max(null_std, 1e-10)

    return {
        'observed_delta_r2': float(observed_delta),
        'null_mean': null_mean,
        'null_std': null_std,
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


# ================================================================
# 1.4 INPUT DECODABILITY CONTROL
# ================================================================
def input_decodability_test(X_window, h_window, target_window):
    """Compare probe accuracy on raw input vs trained hidden states.
    All inputs at window level: (n_windows, dim).
    If R²_input ≈ R²_hidden, the LSTM didn't add value.
    """
    target = target_window.copy().astype(np.float64)
    t_std = np.std(target)
    if t_std < 1e-8:
        return {'r2_input': 0.0, 'r2_hidden': 0.0, 'added_value': 0.0,
                'input_sufficient': True}
    target = (target - np.mean(target)) / t_std
    target = np.clip(target, -3.0, 3.0)

    r2_input, _ = shuffled_cv_r2(X_window, target)
    r2_hidden, _ = shuffled_cv_r2(h_window, target)

    added_value = r2_hidden - r2_input

    return {
        'r2_input': float(r2_input),
        'r2_hidden': float(r2_hidden),
        'added_value': float(added_value),
        'input_sufficient': r2_input > 0 and added_value < 0.02,
    }


# ================================================================
# MAIN: RUN PHASE 1 ON CS48
# ================================================================
def run_phase1(patient_id='sub-CS48'):
    print("=" * 70)
    print(f"PHASE 1: Foundation Tests — {patient_id}")
    print("=" * 70)

    t0 = time.time()

    # ── Load data ──
    print("\n[data] Loading preprocessed data and probe targets...")
    data = np.load(PREPROCESSED_DIR / f'{patient_id}.npz')
    X_train = data['X_train']
    X_val = data['X_val']
    X_test = data['X_test']
    Y_train = data['Y_train']
    Y_val = data['Y_val']
    Y_test = data['Y_test']
    n_input = int(data['n_input'])

    # Load focused neuron indices
    focused_path = RESULTS_DIR / patient_id / 'focused_neurons.json'
    with open(focused_path) as f:
        focused_info = json.load(f)
    focused_idx = focused_info['focused_indices']
    n_focused = len(focused_idx)

    # Restrict output to focused neurons
    Y_train_f = Y_train[:, :, focused_idx]
    Y_val_f = Y_val[:, :, focused_idx]

    print(f"  X_test: {X_test.shape}, n_input={n_input}, n_focused={n_focused}")

    # Load probe targets
    probe_data = np.load(PROBE_TARGETS_DIR / f'{patient_id}.npz')
    target_names = [n for n in probe_data.files
                    if n not in SKIP_TARGETS and n not in CATEGORICAL_TARGETS]
    print(f"  Regression targets: {target_names}")

    # Align targets to test windows
    window_bins = WINDOW_MS // BIN_MS
    stride_bins = STRIDE_MS // BIN_MS
    test_start_window = X_train.shape[0] + X_val.shape[0]
    n_test = X_test.shape[0]

    targets = {}
    for name in target_names:
        t_raw = probe_data[name]
        t_aligned = align_target_to_windows(
            t_raw, n_test, test_start_window, window_bins, stride_bins)
        if np.std(t_aligned) < 1e-6:
            print(f"  SKIP {name}: near-constant after alignment")
            continue
        targets[name] = t_aligned

    # Flatten input for decodability test (1.4)
    X_test_flat = X_test.reshape(-1, X_test.shape[-1])

    seeds = list(range(N_SEEDS))

    # ── 1.1 Matched Baselines ──
    print("\n" + "─" * 60)
    print("1.1 MATCHED BASELINES (same-seed untrained init)")
    print("─" * 60)

    print("  Creating matched untrained models...")
    matched_untrained = create_matched_baselines(
        n_input, n_focused, X_test, seeds)

    print("  Training models and extracting hidden states...")
    trained_states = train_and_extract(
        n_input, n_focused, X_train, Y_train_f, X_val, Y_val_f,
        X_test, seeds)

    # Window dimensions for window-level probing
    bins_per_window = WINDOW_MS // BIN_MS  # 100
    n_windows = n_test  # 191

    # Compute matched-baseline ΔR² per target (shuffled CV only — fast)
    matched_results = {}
    for name, target in targets.items():
        print(f"\n  Target: {name}")
        per_seed = []
        for seed in seeds:
            h_tr = trained_states[seed]
            h_un = matched_untrained[seed]
            min_len = min(len(h_tr), len(h_un), len(target))

            # Window-level averaging
            h_tr_w, t_w = to_window_level(h_tr[:min_len], target[:min_len],
                                            n_windows, bins_per_window)
            h_un_w, _ = to_window_level(h_un[:min_len], target[:min_len],
                                          n_windows, bins_per_window)
            t = t_w.astype(np.float64)
            t_std = np.std(t)
            if t_std < 1e-8:
                per_seed.append({'delta_r2': 0.0, 'seed': seed, 'valid': False})
                continue
            t = (t - np.mean(t)) / t_std
            t = np.clip(t, -3.0, 3.0)

            r2_tr, _ = shuffled_cv_r2(h_tr_w, t, seed=seed)
            r2_un, _ = shuffled_cv_r2(h_un_w, t, seed=seed)
            delta = r2_tr - r2_un
            per_seed.append({
                'delta_r2': float(delta),
                'r2_trained': float(r2_tr),
                'r2_untrained': float(r2_un),
                'seed': seed,
                'valid': r2_tr > 0.0,
            })

        deltas = [r['delta_r2'] for r in per_seed]
        matched_results[name] = {
            'per_seed': per_seed,
            'mean_delta': float(np.mean(deltas)),
            'std_delta': float(np.std(deltas)),
            'median_delta': float(np.median(deltas)),
            'n_positive': int(sum(1 for d in deltas if d > 0)),
            'n_seeds': len(seeds),
        }
        print(f"    Mean ΔR² = {np.mean(deltas):+.4f} ± {np.std(deltas):.4f} "
              f"({sum(1 for d in deltas if d > 0)}/{len(seeds)} positive)")

    # ── 1.2 Gap CV (already computed above in matched baselines) ──
    print("\n" + "─" * 60)
    print("1.2 GAP CROSS-VALIDATION (decorrelation-aware folds)")
    print("─" * 60)

    gap_results = {}
    # Use best seed (seed 0) for detailed analysis
    best_seed = 0
    h_tr_best = trained_states[best_seed]
    h_un_best = matched_untrained[best_seed]

    for name, target in targets.items():
        min_len = min(len(h_tr_best), len(target))

        # Convert to window level
        h_tr_w, t_w = to_window_level(h_tr_best[:min_len], target[:min_len],
                                        n_windows, bins_per_window)
        t = t_w.copy().astype(np.float64)
        t_std = np.std(t)
        if t_std < 1e-8:
            continue
        t = (t - np.mean(t)) / t_std
        t = np.clip(t, -3.0, 3.0)

        # Shuffled CV R² (what original pipeline uses)
        r2_shuffled, _ = shuffled_cv_r2(h_tr_w, t)
        # Temporal CV R² (strict, no leakage)
        r2_temporal, _ = temporal_cv_r2(h_tr_w, t, gap_windows=2)

        inflation = r2_shuffled - r2_temporal
        gap_results[name] = {
            'r2_shuffled': float(r2_shuffled),
            'r2_temporal': float(r2_temporal),
            'inflation': float(inflation),
        }
        flag = " *** INFLATED ***" if inflation > 0.1 else ""
        print(f"  {name:25s}: R² shuffled={r2_shuffled:+.4f}, "
              f"temporal={r2_temporal:+.4f}, "
              f"inflation={inflation:+.4f}{flag}")

    # ── 1.3 iAAFT Significance ──
    print("\n" + "─" * 60)
    print(f"1.3 iAAFT SIGNIFICANCE ({N_SURROGATES} surrogates per target)")
    print("─" * 60)

    iaaft_results = {}
    for name, target in targets.items():
        print(f"\n  Target: {name}")
        min_len = min(len(h_tr_best), len(h_un_best), len(target))
        h_tr_w, t_w = to_window_level(h_tr_best[:min_len], target[:min_len],
                                        n_windows, bins_per_window)
        h_un_w, _ = to_window_level(h_un_best[:min_len], target[:min_len],
                                      n_windows, bins_per_window)
        # Z-score for iAAFT input
        tw = t_w.astype(np.float64)
        tw = (tw - tw.mean()) / max(tw.std(), 1e-8)
        tw = np.clip(tw, -3.0, 3.0)
        res = iaaft_significance_test(h_tr_w, h_un_w, tw,
                                        n_surrogates=N_SURROGATES)
        iaaft_results[name] = res

        sig_str = "SIGNIFICANT" if res['significant_005'] else "NOT significant"
        print(f"    ΔR² = {res['observed_delta_r2']:+.4f}, "
              f"p = {res['p_value']:.4f} ({sig_str}), "
              f"z = {res['z_score']:+.2f}")
        print(f"    Null: mean={res['null_mean']:+.4f}, "
              f"95th={res['null_percentiles']['95']:+.4f}")

    # ── 1.4 Input Decodability ──
    print("\n" + "─" * 60)
    print("1.4 INPUT DECODABILITY CONTROL")
    print("─" * 60)

    input_results = {}
    for name, target in targets.items():
        min_len = min(len(X_test_flat), len(h_tr_best), len(target))
        # Convert all to window level
        h_tr_w, t_w = to_window_level(h_tr_best[:min_len], target[:min_len],
                                        n_windows, bins_per_window)
        N_w = n_windows * bins_per_window
        X_w = X_test_flat[:N_w].reshape(n_windows, bins_per_window, -1).mean(axis=1)
        res = input_decodability_test(X_w, h_tr_w, t_w)
        input_results[name] = res

        flag = " ← INPUT SUFFICIENT" if res['input_sufficient'] else ""
        print(f"  {name:25s}: R²_input={res['r2_input']:+.4f}, "
              f"R²_hidden={res['r2_hidden']:+.4f}, "
              f"added_value={res['added_value']:+.4f}{flag}")

    # ── Summary ──
    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"PHASE 1 COMPLETE ({elapsed:.0f}s)")
    print("=" * 70)

    print("\n┌─────────────────────────┬───────────┬──────────┬─────────┬──────────┬────────────┐")
    print("│ Target                  │ ΔR²_match │ Inflat.  │ p(iAAFT)│ R²_input │ Survives?  │")
    print("├─────────────────────────┼───────────┼──────────┼─────────┼──────────┼────────────┤")

    summary = {}
    for name in targets:
        dr2 = matched_results.get(name, {}).get('mean_delta', 0)
        infl = gap_results.get(name, {}).get('inflation', 0)
        p_val = iaaft_results.get(name, {}).get('p_value', 1.0)
        inp_suff = input_results.get(name, {}).get('input_sufficient', False)
        r2_inp = input_results.get(name, {}).get('r2_input', 0)

        survives = (dr2 > 0.05 and p_val < 0.05 and not inp_suff)
        verdict = "YES" if survives else "NO"

        summary[name] = {
            'delta_r2_matched': float(dr2),
            'autocorr_inflation': float(infl),
            'iaaft_p_value': float(p_val),
            'input_r2': float(r2_inp),
            'input_sufficient': bool(inp_suff),
            'survives_phase1': survives,
        }

        print(f"│ {name:23s} │ {dr2:+.5f}  │ {infl:+.4f}  │ {p_val:.4f} │ {r2_inp:+.4f}  │ {verdict:10s} │")

    print("└─────────────────────────┴───────────┴──────────┴─────────┴──────────┴────────────┘")

    # ── Save ──
    out_path = OUTPUT_DIR / f'phase1_{patient_id}.json'
    full_results = {
        'patient_id': patient_id,
        'n_input': n_input,
        'n_focused': n_focused,
        'n_seeds': N_SEEDS,
        'n_surrogates': N_SURROGATES,
        'elapsed_seconds': elapsed,
        'matched_baselines': {k: {kk: vv for kk, vv in v.items()
                                    if kk != 'per_seed'}
                               for k, v in matched_results.items()},
        'autocorrelation_diagnostic': gap_results,
        'iaaft_significance': {k: {kk: vv for kk, vv in v.items()}
                                for k, v in iaaft_results.items()},
        'input_decodability': input_results,
        'summary': summary,
    }

    with open(out_path, 'w') as f:
        json.dump(full_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    return full_results


if __name__ == '__main__':
    run_phase1('sub-CS48')
