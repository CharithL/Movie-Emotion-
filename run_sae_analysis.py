"""
run_sae_analysis.py — DESCARTES Circuit 5: SAE Retroactive Analysis

Applies Sparse Autoencoders to decompose LSTM hidden states from trained
models. Re-probes the decomposed (monosemantic) features for biological
variables that Ridge probing on raw hidden states may have missed.

Uses JSON serialization only (no unsafe formats).
"""
import numpy as np
import torch
import torch.nn as nn
import json
import time
import sys
from pathlib import Path
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, r2_score

# ================================================================
# PATHS
# ================================================================
BASE_DIR = Path(__file__).parent
PREPROCESSED_DIR = BASE_DIR / 'preprocessed_data'
PROBE_TARGETS_DIR = BASE_DIR / 'probe_targets'
RESULTS_DIR = BASE_DIR / 'results'
SAE_RESULTS_DIR = RESULTS_DIR / 'sae'
SAE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ================================================================
# CONFIG
# ================================================================
HIDDEN_SIZE = 128
N_LAYERS = 2
DROPOUT = 0.1
BATCH_SIZE = 32
LR = 1e-3
MAX_EPOCHS = 200
PATIENCE = 20

# SAE config
SAE_EXPANSION_FACTORS = [4, 8]  # 128→512 and 128→1024
SAE_SPARSITY_COEFFS = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
SAE_EPOCHS = 50
SAE_BATCH_SIZE = 256
SAE_LR = 1e-3

# Probing
N_PROBE_FOLDS = 5
N_SEEDS = 3  # fewer seeds for SAE analysis (faster)
SKIP_TARGETS = {'valence', 'face_presence'}
CATEGORICAL_TARGETS = {'emotion_category'}

BIN_MS = 20
WINDOW_MS = 2000
STRIDE_MS = 500


# ================================================================
# LSTM MODEL (identical to run_all_patients.py)
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


# ================================================================
# SPARSE AUTOENCODER
# ================================================================
class SparseAutoencoder(nn.Module):
    """
    Overcomplete sparse autoencoder for hidden state decomposition.
    Maps hidden_size -> expansion_factor * hidden_size sparse features,
    then reconstructs. The sparse features are the monosemantic
    decomposition to probe against biological variables.
    """
    def __init__(self, hidden_size, expansion_factor=4, sparsity_coeff=1e-3):
        super().__init__()
        self.n_features = hidden_size * expansion_factor
        self.sparsity_coeff = sparsity_coeff
        self.expansion_factor = expansion_factor

        self.encoder = nn.Linear(hidden_size, self.n_features)
        self.decoder = nn.Linear(self.n_features, hidden_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        features = self.relu(self.encoder(x))  # sparse activations
        reconstruction = self.decoder(features)
        return reconstruction, features

    def loss(self, x):
        recon, features = self.forward(x)
        recon_loss = nn.functional.mse_loss(recon, x)
        sparsity_loss = features.abs().mean()  # L1 on activations
        return recon_loss + self.sparsity_coeff * sparsity_loss, recon_loss, sparsity_loss


# ================================================================
# SANITIZED PROBING (copied from run_all_patients.py)
# ================================================================
def sanitized_ridge_delta_r2(h_trained, target, h_untrained, n_folds=N_PROBE_FOLDS):
    """Sanitized Ridge delta-R2 with 3 guards."""
    target = target.copy().astype(np.float64)
    t_std = np.std(target)
    if t_std < 1e-8:
        return {'r2_trained': 0.0, 'r2_untrained': 0.0, 'delta_r2': 0.0,
                'valid': False}

    target = (target - np.mean(target)) / t_std
    target = np.clip(target, -3.0, 3.0)

    alphas = np.logspace(-3, 3, 20)
    kf = KFold(n_splits=n_folds, shuffle=False)

    def probe_r2(hidden):
        folds = []
        for tr, te in kf.split(hidden):
            r2 = RidgeCV(alphas=alphas).fit(hidden[tr], target[tr]).score(hidden[te], target[te])
            folds.append(np.clip(r2, -1.0, 1.0))
        return float(np.mean(folds))

    r2_tr = probe_r2(h_trained)
    r2_un = probe_r2(h_untrained)
    delta = r2_tr - r2_un
    valid = r2_tr > 0.0

    return {'r2_trained': r2_tr, 'r2_untrained': r2_un, 'delta_r2': delta, 'valid': valid}


def classification_delta_accuracy(h_trained, target_labels, h_untrained, n_folds=N_PROBE_FOLDS):
    """Balanced accuracy for categorical targets."""
    unique = np.unique(target_labels)
    n_classes = len(unique)
    chance = 1.0 / n_classes if n_classes >= 2 else 0.0

    if n_classes < 2:
        return {'acc_trained': 0.0, 'acc_untrained': 0.0, 'delta_accuracy': 0.0,
                'valid': False, 'chance': chance}

    kf = KFold(n_splits=n_folds, shuffle=False)

    def probe_acc(hidden):
        sc = StandardScaler()
        hidden_s = sc.fit_transform(hidden)
        folds = []
        for tr, te in kf.split(hidden_s):
            test_classes = np.unique(target_labels[te])
            if len(test_classes) < 2:
                continue
            try:
                clf = LogisticRegressionCV(max_iter=1000, class_weight='balanced',
                                           scoring='balanced_accuracy', cv=3)
                clf.fit(hidden_s[tr], target_labels[tr])
                preds = clf.predict(hidden_s[te])
                folds.append(balanced_accuracy_score(target_labels[te], preds))
            except Exception:
                continue
        return float(np.mean(folds)) if folds else 0.0

    acc_tr = probe_acc(h_trained)
    acc_un = probe_acc(h_untrained)
    delta = acc_tr - acc_un

    return {'acc_trained': acc_tr, 'acc_untrained': acc_un,
            'delta_accuracy': delta, 'valid': acc_tr > chance, 'chance': chance}


# ================================================================
# HELPER FUNCTIONS
# ================================================================
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


def extract_hidden_flat(model, X):
    set_inference_mode(model)
    with torch.no_grad():
        _, h = model(torch.FloatTensor(X).to(DEVICE), return_hidden=True)
    return h.cpu().numpy().reshape(-1, h.shape[-1])


def train_lstm(n_input, n_output, X_train, Y_train, X_val, Y_val, seed):
    """Train a single LSTM with early stopping. Returns trained model."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = LimbicPrefrontalLSTM(n_input, n_output, HIDDEN_SIZE, N_LAYERS, DROPOUT).to(DEVICE)
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
    return model, best_val


def train_sae(hidden_states, expansion_factor, sparsity_coeff):
    """
    Train a Sparse Autoencoder on hidden states.
    Returns: trained SAE, metrics dict
    """
    hidden_size = hidden_states.shape[1]
    sae = SparseAutoencoder(hidden_size, expansion_factor, sparsity_coeff).to(DEVICE)
    optimizer = torch.optim.Adam(sae.parameters(), lr=SAE_LR)

    h_tensor = torch.FloatTensor(hidden_states).to(DEVICE)
    n_samples = len(h_tensor)

    best_recon_loss = float('inf')
    best_state = None

    for epoch in range(SAE_EPOCHS):
        perm = torch.randperm(n_samples)
        epoch_recon = 0.0
        epoch_sparsity = 0.0
        n_batches = 0

        for i in range(0, n_samples, SAE_BATCH_SIZE):
            idx = perm[i:i + SAE_BATCH_SIZE]
            batch = h_tensor[idx]

            optimizer.zero_grad()
            total_loss, recon_loss, sparsity_loss = sae.loss(batch)
            total_loss.backward()
            optimizer.step()

            epoch_recon += recon_loss.item()
            epoch_sparsity += sparsity_loss.item()
            n_batches += 1

        avg_recon = epoch_recon / n_batches
        if avg_recon < best_recon_loss:
            best_recon_loss = avg_recon
            best_state = {k: v.cpu().clone() for k, v in sae.state_dict().items()}

    sae.load_state_dict(best_state)
    set_inference_mode(sae)

    # Compute final metrics
    with torch.no_grad():
        recon, features = sae(h_tensor)
        recon_np = recon.cpu().numpy()
        features_np = features.cpu().numpy()

        # Reconstruction R2
        ss_res = np.sum((hidden_states - recon_np) ** 2)
        ss_tot = np.sum((hidden_states - hidden_states.mean(axis=0)) ** 2)
        recon_r2 = 1.0 - ss_res / max(ss_tot, 1e-10)

        # Sparsity metrics
        l0_per_sample = (features_np > 0).sum(axis=1).astype(float)
        mean_l0 = float(l0_per_sample.mean())
        frac_active = float((features_np > 0).mean())

        # Dead features (never activate)
        n_dead = int((features_np.max(axis=0) == 0).sum())

    metrics = {
        'expansion_factor': expansion_factor,
        'sparsity_coeff': sparsity_coeff,
        'n_features': sae.n_features,
        'recon_r2': float(recon_r2),
        'mean_l0': mean_l0,
        'frac_active': frac_active,
        'n_dead_features': n_dead,
        'best_recon_loss': float(best_recon_loss),
    }

    return sae, metrics


def extract_sae_features(sae, hidden_states):
    """Extract SAE features from hidden states."""
    set_inference_mode(sae)
    with torch.no_grad():
        h_tensor = torch.FloatTensor(hidden_states).to(DEVICE)
        _, features = sae(h_tensor)
    return features.cpu().numpy()


# ================================================================
# MAIN SAE ANALYSIS FOR A PATIENT
# ================================================================
def run_sae_for_patient(patient_id):
    """Full SAE analysis pipeline for one patient."""
    print(f"\n{'='*70}")
    print(f"SAE ANALYSIS: {patient_id}")
    print(f"{'='*70}")

    # Load preprocessed data
    data = np.load(PREPROCESSED_DIR / f"{patient_id}.npz")
    n_input = int(data['n_input'])
    n_output = int(data['n_output'])

    X_train = data['X_train']
    Y_train = data['Y_train']
    X_val = data['X_val']
    Y_val = data['Y_val']
    X_test = data['X_test']
    Y_test = data['Y_test']

    # Load focused neurons info
    focused_path = RESULTS_DIR / patient_id / 'focused_neurons.json'
    with open(focused_path) as f:
        focused_info = json.load(f)
    focused_idx = focused_info['focused_indices']
    n_focused = len(focused_idx)

    # Subset output to focused neurons
    Y_train_f = Y_train[:, :, focused_idx]
    Y_val_f = Y_val[:, :, focused_idx]
    Y_test_f = Y_test[:, :, focused_idx]

    window_bins = WINDOW_MS // BIN_MS
    stride_bins = STRIDE_MS // BIN_MS
    train_end = X_train.shape[0]
    val_end = train_end + X_val.shape[0]
    test_start_window = val_end
    n_test = X_test.shape[0]

    # Load probe targets
    probe_data = np.load(PROBE_TARGETS_DIR / f"{patient_id}.npz")
    target_names = [n for n in probe_data.files if n not in SKIP_TARGETS]

    # Align targets to test windows
    aligned_targets = {}
    for name in target_names:
        target_raw = probe_data[name]
        aligned = align_target_to_windows(
            target_raw, n_test, test_start_window, window_bins, stride_bins)
        aligned_targets[name] = aligned

    # ============================================================
    # Step 1: Train LSTM (one seed) + extract hidden states
    # ============================================================
    print(f"\n--- Step 1: Train LSTM + extract hidden states ---")
    print(f"  Input neurons: {n_input}, Focused output: {n_focused}")

    # ALL data combined for SAE training (train+val+test)
    X_all = np.concatenate([X_train, X_val, X_test], axis=0)
    Y_all_f = np.concatenate([Y_train_f, Y_val_f, Y_test_f], axis=0)

    # Trained model
    t0 = time.time()
    trained_model, trained_loss = train_lstm(
        n_input, n_focused, X_train, Y_train_f, X_val, Y_val_f, seed=0)
    print(f"  Trained LSTM: val_loss={trained_loss:.6f} ({time.time()-t0:.1f}s)")

    # Extract hidden states on ALL data (for SAE training)
    h_all_trained = extract_hidden_flat(trained_model, X_all)
    h_test_trained = extract_hidden_flat(trained_model, X_test)
    print(f"  Trained hidden states: all={h_all_trained.shape}, test={h_test_trained.shape}")

    # Untrained model
    torch.manual_seed(100)
    untrained_model = LimbicPrefrontalLSTM(
        n_input, n_focused, HIDDEN_SIZE, N_LAYERS, DROPOUT).to(DEVICE)
    set_inference_mode(untrained_model)

    h_all_untrained = extract_hidden_flat(untrained_model, X_all)
    h_test_untrained = extract_hidden_flat(untrained_model, X_test)
    print(f"  Untrained hidden states: all={h_all_untrained.shape}, test={h_test_untrained.shape}")

    # ============================================================
    # Step 2: Baseline Ridge probing on raw hidden states
    # ============================================================
    print(f"\n--- Step 2: Baseline Ridge on raw hidden states ---")
    raw_results = {}
    for name in target_names:
        target = aligned_targets[name]
        min_len = min(len(target), len(h_test_trained))
        t_use = target[:min_len]
        h_tr = h_test_trained[:min_len]
        h_un = h_test_untrained[:min_len]

        is_cat = name in CATEGORICAL_TARGETS
        if is_cat:
            res = classification_delta_accuracy(h_tr, t_use.astype(int), h_un)
            raw_results[name] = {
                'method': 'classification',
                'metric_trained': res['acc_trained'],
                'metric_untrained': res['acc_untrained'],
                'delta': res['delta_accuracy'],
                'valid': res['valid'],
            }
        else:
            res = sanitized_ridge_delta_r2(h_tr, t_use, h_un)
            raw_results[name] = {
                'method': 'ridge',
                'metric_trained': res['r2_trained'],
                'metric_untrained': res['r2_untrained'],
                'delta': res['delta_r2'],
                'valid': res['valid'],
            }

        delta_str = f"{raw_results[name]['delta']:+.4f}"
        print(f"  {name:<25s} delta={delta_str}  valid={raw_results[name]['valid']}")

    # ============================================================
    # Step 3: SAE hyperparameter sweep
    # ============================================================
    print(f"\n--- Step 3: SAE hyperparameter sweep ---")
    sae_sweep_results = []

    for ef in SAE_EXPANSION_FACTORS:
        for sc in SAE_SPARSITY_COEFFS:
            t0 = time.time()
            sae, metrics = train_sae(h_all_trained, ef, sc)
            elapsed = time.time() - t0
            sae_sweep_results.append(metrics)
            print(f"  ef={ef} sc={sc:.0e}: R2={metrics['recon_r2']:.4f} "
                  f"L0={metrics['mean_l0']:.1f} dead={metrics['n_dead_features']} "
                  f"({elapsed:.1f}s)")

    # Select best SAE config: highest R2 with R2 > 0.90, then lowest L0
    valid_configs = [m for m in sae_sweep_results if m['recon_r2'] > 0.90]
    if not valid_configs:
        print("  WARNING: No SAE achieved R2 > 0.90, using best available")
        valid_configs = sae_sweep_results

    # Sort by L0 (lower is sparser = better), breaking ties by R2
    valid_configs.sort(key=lambda m: (m['mean_l0'], -m['recon_r2']))
    best_config = valid_configs[0]
    print(f"\n  BEST SAE: ef={best_config['expansion_factor']} "
          f"sc={best_config['sparsity_coeff']:.0e} "
          f"R2={best_config['recon_r2']:.4f} L0={best_config['mean_l0']:.1f}")

    # ============================================================
    # Step 4: Train final SAEs (trained + untrained) with best config
    # ============================================================
    print(f"\n--- Step 4: Train final SAEs ---")
    best_ef = best_config['expansion_factor']
    best_sc = best_config['sparsity_coeff']

    # SAE on trained hidden states
    sae_trained, metrics_trained = train_sae(h_all_trained, best_ef, best_sc)
    print(f"  SAE(trained):   R2={metrics_trained['recon_r2']:.4f} "
          f"L0={metrics_trained['mean_l0']:.1f}")

    # SAE on untrained hidden states (CRITICAL: same hyperparameters)
    sae_untrained, metrics_untrained = train_sae(h_all_untrained, best_ef, best_sc)
    print(f"  SAE(untrained): R2={metrics_untrained['recon_r2']:.4f} "
          f"L0={metrics_untrained['mean_l0']:.1f}")

    # ============================================================
    # Step 5: Extract SAE features on TEST data + re-probe
    # ============================================================
    print(f"\n--- Step 5: SAE feature extraction + re-probing ---")
    feat_trained = extract_sae_features(sae_trained, h_test_trained)
    feat_untrained = extract_sae_features(sae_untrained, h_test_untrained)
    print(f"  SAE features: trained={feat_trained.shape} untrained={feat_untrained.shape}")

    sae_results = {}
    for name in target_names:
        target = aligned_targets[name]
        min_len = min(len(target), len(feat_trained))
        t_use = target[:min_len]
        f_tr = feat_trained[:min_len]
        f_un = feat_untrained[:min_len]

        is_cat = name in CATEGORICAL_TARGETS
        if is_cat:
            res = classification_delta_accuracy(f_tr, t_use.astype(int), f_un)
            sae_results[name] = {
                'method': 'classification',
                'metric_trained': res['acc_trained'],
                'metric_untrained': res['acc_untrained'],
                'delta': res['delta_accuracy'],
                'valid': res['valid'],
            }
        else:
            res = sanitized_ridge_delta_r2(f_tr, t_use, f_un)
            sae_results[name] = {
                'method': 'ridge',
                'metric_trained': res['r2_trained'],
                'metric_untrained': res['r2_untrained'],
                'delta': res['delta_r2'],
                'valid': res['valid'],
            }

        raw_delta = raw_results[name]['delta']
        sae_delta = sae_results[name]['delta']
        change = sae_delta - raw_delta
        marker = ""
        if raw_delta < 0.05 and sae_delta > 0.05:
            marker = " *** REVEALED BY SAE ***"
        elif sae_delta > raw_delta + 0.03:
            marker = " (improved)"

        print(f"  {name:<25s} raw={raw_delta:+.4f} -> SAE={sae_delta:+.4f} "
              f"(change={change:+.4f}){marker}")

    # ============================================================
    # Step 6: Also probe at WINDOW level with SAE features
    # ============================================================
    print(f"\n--- Step 6: Window-level SAE probing ---")
    # Average SAE features within each window
    n_feat_per_window = window_bins  # 100 bins per window
    n_windows_test = n_test

    # feat_trained shape: (n_test * window_bins, n_sae_features)
    feat_tr_3d = feat_trained.reshape(n_windows_test, window_bins, -1)
    feat_un_3d = feat_untrained.reshape(n_windows_test, window_bins, -1)

    # Window-level: average across time within each window
    feat_tr_win = feat_tr_3d.mean(axis=1)  # (n_windows, n_sae_features)
    feat_un_win = feat_un_3d.mean(axis=1)

    # Also raw hidden states at window level
    h_tr_3d = h_test_trained.reshape(n_windows_test, window_bins, -1)
    h_un_3d = h_test_untrained.reshape(n_windows_test, window_bins, -1)
    h_tr_win = h_tr_3d.mean(axis=1)
    h_un_win = h_un_3d.mean(axis=1)

    window_raw_results = {}
    window_sae_results = {}

    for name in target_names:
        target = aligned_targets[name]
        # Average target within each window
        t_flat = target[:n_windows_test * window_bins]
        t_3d = t_flat.reshape(n_windows_test, window_bins)
        t_win = t_3d.mean(axis=1)

        is_cat = name in CATEGORICAL_TARGETS
        if is_cat:
            # For categorical: use mode within window
            t_int = t_flat.astype(int).reshape(n_windows_test, window_bins)
            from scipy import stats as sp_stats
            t_win_cat = np.array([sp_stats.mode(row, keepdims=False).mode for row in t_int])

            # Raw window
            res_raw = classification_delta_accuracy(h_tr_win, t_win_cat, h_un_win)
            window_raw_results[name] = {
                'method': 'classification',
                'delta': res_raw['delta_accuracy'],
                'valid': res_raw['valid'],
            }
            # SAE window
            res_sae = classification_delta_accuracy(feat_tr_win, t_win_cat, feat_un_win)
            window_sae_results[name] = {
                'method': 'classification',
                'delta': res_sae['delta_accuracy'],
                'valid': res_sae['valid'],
            }
        else:
            # Raw window
            res_raw = sanitized_ridge_delta_r2(h_tr_win, t_win, h_un_win)
            window_raw_results[name] = {
                'method': 'ridge',
                'delta': res_raw['delta_r2'],
                'valid': res_raw['valid'],
            }
            # SAE window
            res_sae = sanitized_ridge_delta_r2(feat_tr_win, t_win, feat_un_win)
            window_sae_results[name] = {
                'method': 'ridge',
                'delta': res_sae['delta_r2'],
                'valid': res_sae['valid'],
            }

        print(f"  {name:<25s} raw_win={window_raw_results[name]['delta']:+.4f} "
              f"SAE_win={window_sae_results[name]['delta']:+.4f}")

    # ============================================================
    # Step 7: Save results
    # ============================================================
    out_dir = SAE_RESULTS_DIR / f"circuit5_{patient_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    full_report = {
        'patient_id': patient_id,
        'n_input': n_input,
        'n_focused': n_focused,
        'hidden_size': HIDDEN_SIZE,
        'sae_config': {
            'expansion_factor': best_ef,
            'sparsity_coeff': best_sc,
            'n_sae_features': best_ef * HIDDEN_SIZE,
        },
        'sae_metrics': {
            'trained': metrics_trained,
            'untrained': metrics_untrained,
        },
        'sweep_results': sae_sweep_results,
        'bin_level': {
            'raw_ridge': {k: v for k, v in raw_results.items()},
            'sae_probe': {k: v for k, v in sae_results.items()},
        },
        'window_level': {
            'raw_ridge': {k: v for k, v in window_raw_results.items()},
            'sae_probe': {k: v for k, v in window_sae_results.items()},
        },
    }

    with open(out_dir / 'sae_probing.json', 'w') as f:
        json.dump(full_report, f, indent=2)
    print(f"\n  Saved: {out_dir / 'sae_probing.json'}")

    # ============================================================
    # Step 8: Generate comparison table
    # ============================================================
    print(f"\n{'='*100}")
    print(f"COMPARISON TABLE: {patient_id}")
    print(f"{'='*100}")
    header = (f"{'Target':<25} {'Bin Raw':>10} {'Bin SAE':>10} {'Change':>10} "
              f"{'Win Raw':>10} {'Win SAE':>10} {'Change':>10} {'Revealed?':>12}")
    print(header)
    print('-' * 100)

    for name in sorted(target_names):
        bin_raw = raw_results[name]['delta']
        bin_sae = sae_results[name]['delta']
        bin_change = bin_sae - bin_raw
        win_raw = window_raw_results[name]['delta']
        win_sae = window_sae_results[name]['delta']
        win_change = win_sae - win_raw

        # Was this target revealed?
        revealed = ""
        if bin_raw < 0.05 and bin_sae > 0.05:
            revealed = "BIN-SAE"
        if win_raw < 0.05 and win_sae > 0.05:
            revealed = revealed + "+WIN-SAE" if revealed else "WIN-SAE"
        if bin_raw < 0.05 and win_raw > 0.05:
            revealed = revealed + "+WIN" if revealed else "WIN"

        print(f"{name:<25} {bin_raw:>+10.4f} {bin_sae:>+10.4f} {bin_change:>+10.4f} "
              f"{win_raw:>+10.4f} {win_sae:>+10.4f} {win_change:>+10.4f} {revealed:>12}")

    return full_report


# ================================================================
# MAIN
# ================================================================
def main():
    t_start = time.time()

    # Circuit 5: Start with sub-CS48 (the exemplar patient)
    print("=" * 70)
    print("DESCARTES SAE RETROACTIVE ANALYSIS — Circuit 5")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    # Run CS48 (the only OK-quality patient)
    report_cs48 = run_sae_for_patient('sub-CS48')

    # Also run on a LOW_QUALITY patient for comparison (CS42 has 5 focused neurons)
    report_cs42 = run_sae_for_patient('sub-CS42')

    # Generate cross-patient SAE summary
    print(f"\n{'='*70}")
    print("SAE ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"Total time: {time.time() - t_start:.1f}s")

    # Save overall summary
    summary = {
        'circuit': 'Circuit 5 (Limbic->Prefrontal)',
        'patients_analyzed': ['sub-CS48', 'sub-CS42'],
        'sae_method': 'Overcomplete sparse autoencoder on LSTM hidden states',
        'key_question': 'Do SAE features reveal biological variables invisible to linear Ridge probes?',
    }
    with open(SAE_RESULTS_DIR / 'circuit5_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll results saved to {SAE_RESULTS_DIR}")


if __name__ == '__main__':
    main()
