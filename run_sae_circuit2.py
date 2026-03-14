"""
run_sae_circuit2.py -- DESCARTES Circuit 2: SAE Retroactive Analysis (CA3->CA1)

Applies Sparse Autoencoders to decompose pre-extracted LSTM hidden states
for the hippocampal CA3->CA1 circuit. Probes SAE features against 25
biological variables. gamma_amp is the POSITIVE CONTROL: if SAE cannot
recover it, the SAE hyperparameters need tuning.

Uses JSON serialization only (no unsafe formats).
Runs from movie emotion directory to avoid AppControl DLL blocking.
"""
import numpy as np
import torch
import torch.nn as nn
import json
import time
from pathlib import Path
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

# ================================================================
# PATHS (absolute to avoid directory dependency)
# ================================================================
MIMO_DIR = Path("C:/Users/chari/OneDrive/Documents/Descartes_Cogito/La Masson 2002/MIMO/hippocampal_mimo")
HIDDEN_STATE_DIR = MIMO_DIR / 'data' / 'hidden_states'
SWEEP_DIR = MIMO_DIR
BIOLOGY_DIR = MIMO_DIR / 'checkpoints' / 'biology_test'
SAE_RESULTS_DIR = Path("C:/Users/chari/OneDrive/Documents/Descartes_Cogito/movie emotion/results/sae")
SAE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# All 25 biological targets
BIOLOGICAL_TARGETS = [
    'I_Na_CA1', 'I_KDR_CA1', 'I_Ka_CA1', 'I_h_CA1', 'I_CaL_CA1',
    'I_CaN_CA1', 'I_KCa_CA1', 'I_M_CA1', 'I_AHP_CA1',
    'Ca_i_CA1',
    'g_AMPA_SC', 'g_NMDA_SC', 'g_GABA_A', 'g_GABA_B',
    'h_Na_CA1', 'n_KDR_CA1', 'h_Ka_CA1', 'm_h_CA1', 'm_CaL_CA1', 'm_M_CA1',
    'V_basket', 'V_OLM',
    'theta_phase', 'gamma_amp',
    'V_CA1',
]

# Key targets for focused analysis
KEY_TARGETS = ['gamma_amp', 'theta_phase', 'V_CA1', 'Ca_i_CA1',
               'I_h_CA1', 'g_GABA_B', 'I_M_CA1', 'V_basket']

# SAE config
SAE_EXPANSION_FACTORS = [4, 8]
SAE_SPARSITY_COEFFS = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
SAE_EPOCHS = 50
SAE_BATCH_SIZE = 256
SAE_LR = 1e-3

N_PROBE_FOLDS = 5
N_WINDOWS = 108
STEPS_PER_WINDOW_HIDDEN = 200
STEPS_PER_WINDOW_BIO = 5000
DOWNSAMPLE_RATIO = STEPS_PER_WINDOW_BIO // STEPS_PER_WINDOW_HIDDEN  # 25


# ================================================================
# SPARSE AUTOENCODER
# ================================================================
class SparseAutoencoder(nn.Module):
    def __init__(self, hidden_size, expansion_factor=4, sparsity_coeff=1e-3):
        super().__init__()
        self.n_features = hidden_size * expansion_factor
        self.sparsity_coeff = sparsity_coeff
        self.expansion_factor = expansion_factor
        self.encoder = nn.Linear(hidden_size, self.n_features)
        self.decoder = nn.Linear(self.n_features, hidden_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        features = self.relu(self.encoder(x))
        reconstruction = self.decoder(features)
        return reconstruction, features

    def loss(self, x):
        recon, features = self.forward(x)
        recon_loss = nn.functional.mse_loss(recon, x)
        sparsity_loss = features.abs().mean()
        return recon_loss + self.sparsity_coeff * sparsity_loss, recon_loss, sparsity_loss


def set_inference_mode(model):
    model.train(False)


# ================================================================
# PROBING
# ================================================================
def sanitized_ridge_delta_r2(h_trained, target, h_untrained, n_folds=N_PROBE_FOLDS):
    target = target.copy().astype(np.float64)
    t_std = np.std(target)
    if t_std < 1e-8:
        return {'r2_trained': 0.0, 'r2_untrained': 0.0, 'delta_r2': 0.0, 'valid': False}

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


# ================================================================
# SAE TRAINING
# ================================================================
def train_sae(hidden_states, expansion_factor, sparsity_coeff):
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
        n_batches = 0
        for i in range(0, n_samples, SAE_BATCH_SIZE):
            idx = perm[i:i + SAE_BATCH_SIZE]
            batch = h_tensor[idx]
            optimizer.zero_grad()
            total_loss, recon_loss, _ = sae.loss(batch)
            total_loss.backward()
            optimizer.step()
            epoch_recon += recon_loss.item()
            n_batches += 1

        avg_recon = epoch_recon / n_batches
        if avg_recon < best_recon_loss:
            best_recon_loss = avg_recon
            best_state = {k: v.cpu().clone() for k, v in sae.state_dict().items()}

    sae.load_state_dict(best_state)
    set_inference_mode(sae)

    with torch.no_grad():
        recon, features = sae(h_tensor)
        recon_np = recon.cpu().numpy()
        features_np = features.cpu().numpy()
        ss_res = np.sum((hidden_states - recon_np) ** 2)
        ss_tot = np.sum((hidden_states - hidden_states.mean(axis=0)) ** 2)
        recon_r2 = 1.0 - ss_res / max(ss_tot, 1e-10)
        l0_per_sample = (features_np > 0).sum(axis=1).astype(float)
        mean_l0 = float(l0_per_sample.mean())
        n_dead = int((features_np.max(axis=0) == 0).sum())

    metrics = {
        'expansion_factor': expansion_factor,
        'sparsity_coeff': sparsity_coeff,
        'n_features': sae.n_features,
        'recon_r2': float(recon_r2),
        'mean_l0': mean_l0,
        'n_dead_features': n_dead,
    }
    return sae, metrics


def extract_sae_features(sae, hidden_states):
    set_inference_mode(sae)
    with torch.no_grad():
        _, features = sae(torch.FloatTensor(hidden_states).to(DEVICE))
    return features.cpu().numpy()


# ================================================================
# BIOLOGY LOADING + ALIGNMENT
# ================================================================
def load_biology_aligned(target_name, neuron_idx=0):
    """Load biology and downsample to match hidden state timesteps."""
    bio_path = BIOLOGY_DIR / f"{target_name}.npy"
    if not bio_path.exists():
        return None

    arr = np.load(bio_path)

    if arr.ndim == 3:
        arr = arr[:, :, neuron_idx]
    elif arr.ndim == 1:
        if len(arr) == N_WINDOWS * STEPS_PER_WINDOW_BIO:
            arr = arr.reshape(N_WINDOWS, STEPS_PER_WINDOW_BIO)
        else:
            return None

    n_win, n_steps = arr.shape
    n_out = n_steps // DOWNSAMPLE_RATIO
    arr_ds = arr[:, :n_out * DOWNSAMPLE_RATIO].reshape(n_win, n_out, DOWNSAMPLE_RATIO).mean(axis=2)
    return arr_ds.reshape(-1).astype(np.float32)


# ================================================================
# MAIN
# ================================================================
def run_sae_circuit2(hidden_size_label='h128'):
    """Run SAE analysis for Circuit 2 at a specific hidden size."""
    print(f"\n{'='*70}")
    print(f"SAE ANALYSIS: Circuit 2 (CA3->CA1) -- {hidden_size_label}")
    print(f"{'='*70}")

    sweep_path = SWEEP_DIR / f'sweep_{hidden_size_label}'
    h_trained = np.load(sweep_path / 'trained_hidden.npy')
    h_untrained = np.load(sweep_path / 'untrained_hidden.npy')
    hidden_dim = h_trained.shape[1]
    n_samples = h_trained.shape[0]
    print(f"  Hidden states: {h_trained.shape} (dim={hidden_dim})")

    # Step 1: Baseline Ridge
    print(f"\n--- Step 1: Baseline Ridge on raw hidden states ---")
    raw_results = {}
    for target in KEY_TARGETS:
        bio = load_biology_aligned(target)
        if bio is None:
            print(f"  {target:<20s} SKIPPED (data not found)")
            continue
        min_len = min(len(bio), n_samples)
        res = sanitized_ridge_delta_r2(h_trained[:min_len], bio[:min_len], h_untrained[:min_len])
        raw_results[target] = res
        print(f"  {target:<20s} delta={res['delta_r2']:+.4f}  valid={res['valid']}")

    # Step 2: SAE sweep
    print(f"\n--- Step 2: SAE hyperparameter sweep ---")
    sweep_results = []
    for ef in SAE_EXPANSION_FACTORS:
        for sc in SAE_SPARSITY_COEFFS:
            t0 = time.time()
            sae, metrics = train_sae(h_trained, ef, sc)
            elapsed = time.time() - t0
            sweep_results.append(metrics)
            print(f"  ef={ef} sc={sc:.0e}: R2={metrics['recon_r2']:.4f} "
                  f"L0={metrics['mean_l0']:.1f} dead={metrics['n_dead_features']} "
                  f"({elapsed:.1f}s)")

    valid_configs = [m for m in sweep_results if m['recon_r2'] > 0.90]
    if not valid_configs:
        valid_configs = sweep_results
    valid_configs.sort(key=lambda m: (m['mean_l0'], -m['recon_r2']))
    best = valid_configs[0]
    print(f"\n  BEST: ef={best['expansion_factor']} sc={best['sparsity_coeff']:.0e} "
          f"R2={best['recon_r2']:.4f} L0={best['mean_l0']:.1f}")

    # Step 3: Final SAEs
    print(f"\n--- Step 3: Train final SAEs ---")
    sae_trained, metrics_trained = train_sae(h_trained, best['expansion_factor'], best['sparsity_coeff'])
    sae_untrained, metrics_untrained = train_sae(h_untrained, best['expansion_factor'], best['sparsity_coeff'])
    print(f"  SAE(trained):   R2={metrics_trained['recon_r2']:.4f} L0={metrics_trained['mean_l0']:.1f}")
    print(f"  SAE(untrained): R2={metrics_untrained['recon_r2']:.4f} L0={metrics_untrained['mean_l0']:.1f}")

    # Step 4: Re-probe all 25 targets
    print(f"\n--- Step 4: SAE re-probing (all 25 targets) ---")
    feat_trained = extract_sae_features(sae_trained, h_trained)
    feat_untrained = extract_sae_features(sae_untrained, h_untrained)
    print(f"  SAE features: {feat_trained.shape}")

    sae_results = {}
    for target in BIOLOGICAL_TARGETS:
        bio = load_biology_aligned(target)
        if bio is None:
            continue
        min_len = min(len(bio), n_samples)

        if target not in raw_results:
            res_raw = sanitized_ridge_delta_r2(h_trained[:min_len], bio[:min_len], h_untrained[:min_len])
            raw_results[target] = res_raw

        res_sae = sanitized_ridge_delta_r2(feat_trained[:min_len], bio[:min_len], feat_untrained[:min_len])
        sae_results[target] = res_sae

        raw_delta = raw_results[target]['delta_r2']
        sae_delta = res_sae['delta_r2']
        change = sae_delta - raw_delta
        marker = ""
        if raw_delta < 0.05 and sae_delta > 0.05:
            marker = " *** REVEALED ***"
        elif sae_delta > raw_delta + 0.05:
            marker = " (improved)"

        print(f"  {target:<20s} raw={raw_delta:+.4f} SAE={sae_delta:+.4f} "
              f"({change:+.4f}){marker}")

    # Step 5: Window-level probing
    print(f"\n--- Step 5: Window-level SAE probing ---")
    h_tr_3d = h_trained.reshape(N_WINDOWS, STEPS_PER_WINDOW_HIDDEN, -1)
    h_un_3d = h_untrained.reshape(N_WINDOWS, STEPS_PER_WINDOW_HIDDEN, -1)
    h_tr_win = h_tr_3d.mean(axis=1)
    h_un_win = h_un_3d.mean(axis=1)

    feat_tr_3d = feat_trained.reshape(N_WINDOWS, STEPS_PER_WINDOW_HIDDEN, -1)
    feat_un_3d = feat_untrained.reshape(N_WINDOWS, STEPS_PER_WINDOW_HIDDEN, -1)
    feat_tr_win = feat_tr_3d.mean(axis=1)
    feat_un_win = feat_un_3d.mean(axis=1)

    window_raw_results = {}
    window_sae_results = {}

    for target in KEY_TARGETS:
        bio = load_biology_aligned(target)
        if bio is None:
            continue
        bio_3d = bio.reshape(N_WINDOWS, STEPS_PER_WINDOW_HIDDEN)
        bio_win = bio_3d.mean(axis=1)

        res_raw = sanitized_ridge_delta_r2(h_tr_win, bio_win, h_un_win)
        res_sae = sanitized_ridge_delta_r2(feat_tr_win, bio_win, feat_un_win)
        window_raw_results[target] = res_raw
        window_sae_results[target] = res_sae

        print(f"  {target:<20s} raw_win={res_raw['delta_r2']:+.4f} "
              f"SAE_win={res_sae['delta_r2']:+.4f}")

    # Step 6: Save + table
    out_dir = SAE_RESULTS_DIR / f"circuit2_{hidden_size_label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        'circuit': 'Circuit 2 (CA3->CA1)',
        'hidden_size_label': hidden_size_label,
        'hidden_dim': hidden_dim,
        'sae_config': {
            'expansion_factor': best['expansion_factor'],
            'sparsity_coeff': best['sparsity_coeff'],
            'n_sae_features': best['n_features'],
        },
        'sae_metrics': {
            'trained': metrics_trained,
            'untrained': metrics_untrained,
        },
        'sweep_results': sweep_results,
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
        json.dump(report, f, indent=2)

    print(f"\n{'='*110}")
    print(f"COMPARISON TABLE: Circuit 2 -- {hidden_size_label}")
    print(f"{'='*110}")
    header = (f"{'Target':<20} {'Bin Raw':>10} {'Bin SAE':>10} {'Change':>10} "
              f"{'Win Raw':>10} {'Win SAE':>10} {'Change':>10} {'Revealed?':>12}")
    print(header)
    print('-' * 110)

    for target in KEY_TARGETS:
        if target not in raw_results or target not in sae_results:
            continue
        bin_raw = raw_results[target]['delta_r2']
        bin_sae = sae_results[target]['delta_r2']
        bin_change = bin_sae - bin_raw

        win_raw = window_raw_results.get(target, {}).get('delta_r2', float('nan'))
        win_sae = window_sae_results.get(target, {}).get('delta_r2', float('nan'))
        win_change = win_sae - win_raw if not np.isnan(win_raw) else float('nan')

        revealed = ""
        if bin_raw < 0.05 and bin_sae > 0.05:
            revealed = "BIN-SAE"
        if not np.isnan(win_raw) and bin_raw < 0.05 and win_raw > 0.05:
            revealed = revealed + "+WIN" if revealed else "WIN"

        print(f"{target:<20} {bin_raw:>+10.4f} {bin_sae:>+10.4f} {bin_change:>+10.4f} "
              f"{win_raw:>+10.4f} {win_sae:>+10.4f} {win_change:>+10.4f} {revealed:>12}")

    print(f"\n  Saved: {out_dir / 'sae_probing.json'}")
    return report


def main():
    t_start = time.time()
    print("=" * 70)
    print("DESCARTES SAE RETROACTIVE ANALYSIS -- Circuit 2 (CA3->CA1)")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    report_h128 = run_sae_circuit2('h128')
    report_h256 = run_sae_circuit2('h256')

    print(f"\n{'='*70}")
    print(f"CIRCUIT 2 SAE COMPLETE -- {time.time()-t_start:.1f}s total")
    print(f"{'='*70}")

    for label, report in [('h128', report_h128), ('h256', report_h256)]:
        gamma_raw = report['bin_level']['raw_ridge'].get('gamma_amp', {}).get('delta_r2', None)
        gamma_sae = report['bin_level']['sae_probe'].get('gamma_amp', {}).get('delta_r2', None)
        if gamma_raw is not None and gamma_sae is not None:
            print(f"  POSITIVE CONTROL [{label}]: gamma_amp raw={gamma_raw:+.4f} SAE={gamma_sae:+.4f}")
            if gamma_raw > 0.10 and gamma_sae < gamma_raw * 0.5:
                print(f"    WARNING: SAE degraded gamma_amp recovery!")
            elif gamma_sae > gamma_raw + 0.05:
                print(f"    SAE improved gamma_amp!")
            else:
                print(f"    SAE preserved gamma_amp (expected behavior)")


if __name__ == '__main__':
    main()
