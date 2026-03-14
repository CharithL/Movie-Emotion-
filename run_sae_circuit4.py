"""
DESCARTES SAE Analysis -- Circuit 4 (Human Working Memory)

Applies Sparse Autoencoders to LSTM hidden states for the human WM circuit
(Rutishauser DANDI 000576). For each patient:
  1. Load checkpoint + NWB data
  2. Forward pass to extract hidden states (trained + untrained)
  3. Train SAE, extract features
  4. Probe at bin level AND window level (trial-averaged)

Biological probe targets (from human_wm.config):
  Level B: persistent_delay, memory_load, delay_stability, recognition_decision
  Level C: theta_modulation, gamma_modulation, population_synchrony
  Level A: concept_selectivity, mean_firing_rate

Run from movie emotion directory:
  cd "movie emotion"
  python -X utf8 run_sae_circuit4.py
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# Add WM project to path
WM_DIR = Path("C:/Users/chari/OneDrive/Documents/Descartes_Cogito/Working memory")
sys.path.insert(0, str(WM_DIR))

SAE_RESULTS_DIR = Path("C:/Users/chari/OneDrive/Documents/Descartes_Cogito/movie emotion/results/sae")
SAE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_DIR = WM_DIR / "data" / "results" / "cross_patient" / "cross_patient"

# ---------- SAE ----------
class SparseAutoencoder(nn.Module):
    def __init__(self, hidden_size, expansion_factor=4, sparsity_coeff=1e-3):
        super().__init__()
        self.n_features = hidden_size * expansion_factor
        self.sparsity_coeff = sparsity_coeff
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


def train_sae(hidden_states, expansion_factor=4, sparsity_coeff=1e-3,
              epochs=100, batch_size=256, lr=1e-3):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X = torch.tensor(hidden_states, dtype=torch.float32).to(device)
    hidden_size = X.shape[1]

    sae = SparseAutoencoder(hidden_size, expansion_factor, sparsity_coeff).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    n = X.shape[0]
    for epoch in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            batch = X[perm[i:i+batch_size]]
            total_loss, _, _ = sae.loss(batch)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

    sae.train(False)
    with torch.no_grad():
        recon, features = sae(X)
        recon_loss = nn.functional.mse_loss(recon, X).item()
        var_total = X.var().item()
        recon_r2 = 1.0 - recon_loss / (var_total + 1e-12)

        feat_np = features.cpu().numpy()
        l0 = (feat_np > 0).sum(axis=1).mean()
        n_dead = (feat_np.sum(axis=0) == 0).sum()

    return sae, {
        'expansion_factor': expansion_factor,
        'sparsity_coeff': sparsity_coeff,
        'n_features': sae.n_features,
        'recon_r2': recon_r2,
        'mean_l0': float(l0),
        'n_dead_features': int(n_dead),
    }


def extract_sae_features(sae, hidden_states):
    device = next(sae.parameters()).device
    X = torch.tensor(hidden_states, dtype=torch.float32).to(device)
    sae.train(False)
    with torch.no_grad():
        _, features = sae(X)
    return features.cpu().numpy()


# ---------- probing ----------
def sanitized_ridge_delta_r2(X_trained, y, X_untrained, n_splits=5, alpha=1.0):
    def _safe(a):
        a = np.asarray(a, dtype=np.float64)
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        mu, sd = a.mean(), a.std()
        if sd < 1e-12:
            return None
        a = (a - mu) / sd
        return np.clip(a, -5, 5)

    y_s = _safe(y)
    if y_s is None:
        return {'r2_trained': 0.0, 'r2_untrained': 0.0, 'delta_r2': 0.0, 'valid': False,
                'reason': 'zero_variance_target'}

    def _probe(X, y_s):
        X = np.asarray(X, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        if X.std() < 1e-12:
            return 0.0
        n = min(len(X), len(y_s))
        if n < 10:
            return 0.0
        X, ys = X[:n], y_s[:n]
        kf = KFold(n_splits=min(n_splits, n), shuffle=True, random_state=42)
        scores = []
        for tr, te in kf.split(X):
            sc = StandardScaler()
            X_tr = sc.fit_transform(X[tr])
            X_te = sc.transform(X[te])
            ridge = Ridge(alpha=alpha)
            ridge.fit(X_tr, ys[tr])
            scores.append(ridge.score(X_te, ys[te]))
        r2 = float(np.median(scores))
        return max(min(r2, 1.0), -1.0)

    r2_t = _probe(X_trained, y_s)
    r2_u = _probe(X_untrained, y_s)
    delta = r2_t - r2_u
    valid = not (r2_t < 0 and r2_u < 0 and abs(r2_t) > 0.3)
    return {'r2_trained': r2_t, 'r2_untrained': r2_u, 'delta_r2': delta, 'valid': valid}


# ---------- hidden state extraction ----------
def extract_hidden_states(model, X_data, device='cpu'):
    """Forward pass to get hidden states.
    X_data: (n_trials, T, input_dim)
    Returns: (n_trials * T, hidden_size) flattened hidden states
    """
    model.train(False)
    model = model.to(device)
    X_t = torch.tensor(X_data, dtype=torch.float32).to(device)

    with torch.no_grad():
        _, h_seq = model(X_t, return_hidden=True)
        # h_seq: (n_trials, T, hidden_size)
        h_np = h_seq.cpu().numpy()

    # Flatten to (n_trials * T, hidden_size) for bin-level
    n_trials, T, hidden_size = h_np.shape
    h_flat = h_np.reshape(n_trials * T, hidden_size)

    # Also compute window-level (trial averages)
    h_window = h_np.mean(axis=1)  # (n_trials, hidden_size)

    return h_flat, h_window, h_np


def compute_human_targets(Y_test, trial_info=None):
    """Compute probe targets for human WM.
    Y_test: (n_trials, T, n_neurons)
    Returns: dict of target_name -> (n_trials,) arrays for window-level probing
    """
    targets = {}
    n_trials, T, n_neurons = Y_test.shape

    # Mean firing rate per trial
    targets['mean_firing_rate'] = Y_test.mean(axis=(1, 2))

    # Population rate per trial
    targets['population_rate'] = Y_test.sum(axis=2).mean(axis=1)

    # Trial variance (how much activity varies within trial)
    targets['trial_variance'] = Y_test.var(axis=1).mean(axis=1)

    # Persistent delay proxy: variance across time (low = persistent)
    temporal_var = Y_test.var(axis=1).mean(axis=1)
    targets['temporal_stability'] = 1.0 / (temporal_var + 1e-6)

    # Population synchrony: mean pairwise correlation across neurons
    if n_neurons >= 3:
        sync_vals = []
        for i in range(n_trials):
            trial_data = Y_test[i]  # (T, n_neurons)
            if trial_data.std() < 1e-12:
                sync_vals.append(0.0)
                continue
            # Pairwise correlation of neurons
            corr_matrix = np.corrcoef(trial_data.T)
            # Upper triangle mean (excluding diagonal)
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            vals = corr_matrix[mask]
            vals = vals[np.isfinite(vals)]
            sync_vals.append(np.abs(vals).mean() if len(vals) > 0 else 0.0)
        targets['population_synchrony'] = np.array(sync_vals)

    return targets


def numpy_to_python(obj):
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [numpy_to_python(v) for v in obj]
    return obj


# ---------- per-patient analysis ----------
def run_sae_for_patient(patient_dir, nwb_path, schema, hidden_size=128):
    """Run full SAE analysis for one human patient."""
    from human_wm.surrogate.models import HumanLSTMSurrogate
    from human_wm.data.nwb_loader import extract_patient_data, split_data

    patient_id = patient_dir.name
    checkpoint_path = patient_dir / f"lstm_h{hidden_size}_s0_best.pt"

    if not checkpoint_path.exists():
        print(f"  SKIP {patient_id}: no checkpoint")
        return None

    # Check existing result for CC quality
    result_path = patient_dir / f"results_lstm_h{hidden_size}_s0.json"
    if result_path.exists():
        with open(result_path) as f:
            prev = json.load(f)
        cc = prev.get('cc', 0)
        if cc < 0.3:
            print(f"  SKIP {patient_id}: CC={cc:.3f} < 0.3 (low quality)")
            return None
    else:
        cc = None

    cc_str = f"{cc:.3f}" if cc is not None else "?"
    print(f"\n  {patient_id} (CC={cc_str}):")

    # Load NWB data
    try:
        X, Y, trial_info = extract_patient_data(nwb_path, schema)
    except Exception as e:
        print(f"    Error loading NWB: {e}")
        return None

    splits = split_data(X, Y, trial_info, seed=42)
    X_test = splits['test']['X']
    Y_test = splits['test']['Y']

    n_trials, T, nwb_input_dim = X_test.shape
    nwb_output_dim = Y_test.shape[2]
    print(f"    Data: X_test={X_test.shape} Y_test={Y_test.shape}")

    # Load trained model — infer dims from checkpoint weights
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    ckpt_input_dim = state_dict['lstm.weight_ih_l0'].shape[1]
    ckpt_output_dim = state_dict['output_proj.weight'].shape[0]

    # Use checkpoint dims for model; truncate/pad NWB data to match
    input_dim = ckpt_input_dim
    output_dim = ckpt_output_dim
    if nwb_input_dim != input_dim:
        print(f"    [dim mismatch] NWB input={nwb_input_dim}, checkpoint={input_dim} — truncating/padding input")
        if nwb_input_dim > input_dim:
            X_test = X_test[:, :, :input_dim]
        else:
            pad = np.zeros((n_trials, T, input_dim - nwb_input_dim))
            X_test = np.concatenate([X_test, pad], axis=2)
    if nwb_output_dim != output_dim:
        print(f"    [dim mismatch] NWB output={nwb_output_dim}, checkpoint={output_dim} — truncating/padding output")
        if nwb_output_dim > output_dim:
            Y_test = Y_test[:, :, :output_dim]
        else:
            pad = np.zeros((n_trials, T, output_dim - nwb_output_dim))
            Y_test = np.concatenate([Y_test, pad], axis=2)

    model_trained = HumanLSTMSurrogate(input_dim, output_dim, hidden_size)
    model_trained.load_state_dict(state_dict)

    # Create untrained model (same architecture, random weights)
    torch.manual_seed(999)
    model_untrained = HumanLSTMSurrogate(input_dim, output_dim, hidden_size)

    # Extract hidden states
    H_bin_trained, H_win_trained, H_3d_trained = extract_hidden_states(model_trained, X_test)
    H_bin_untrained, H_win_untrained, _ = extract_hidden_states(model_untrained, X_test)

    print(f"    Hidden: bin={H_bin_trained.shape} window={H_win_trained.shape}")

    # SAE sweep (quick -- only a few configs for speed)
    sweep_configs = [(4, 1e-3), (4, 5e-3), (4, 1e-2), (8, 1e-3), (8, 1e-2)]
    sweep_results = []
    for ef, sc in sweep_configs:
        _, metrics = train_sae(H_bin_trained, ef, sc, epochs=50)
        sweep_results.append(metrics)
        tag = "*" if metrics['recon_r2'] > 0.90 else " "
        print(f"    {tag} ef={ef} sc={sc:.0e}: R2={metrics['recon_r2']:.4f} L0={metrics['mean_l0']:.1f}")

    # Select best
    viable = [s for s in sweep_results if s['recon_r2'] > 0.90]
    if not viable:
        viable = sorted(sweep_results, key=lambda s: -s['recon_r2'])[:2]
    best = min(viable, key=lambda s: s['mean_l0'])

    # Train final SAEs
    sae_trained, met_tr = train_sae(H_bin_trained, best['expansion_factor'],
                                     best['sparsity_coeff'], epochs=100)
    sae_untrained, met_un = train_sae(H_bin_untrained, best['expansion_factor'],
                                       best['sparsity_coeff'], epochs=100)

    # Extract features at bin and window levels
    F_bin_trained = extract_sae_features(sae_trained, H_bin_trained)
    F_bin_untrained = extract_sae_features(sae_untrained, H_bin_untrained)

    # For window-level SAE: train separate SAE on window-level hidden states
    sae_win_tr, met_win_tr = train_sae(H_win_trained, best['expansion_factor'],
                                        best['sparsity_coeff'], epochs=200)
    sae_win_un, met_win_un = train_sae(H_win_untrained, best['expansion_factor'],
                                        best['sparsity_coeff'], epochs=200)
    F_win_trained = extract_sae_features(sae_win_tr, H_win_trained)
    F_win_untrained = extract_sae_features(sae_win_un, H_win_untrained)

    print(f"    SAE bin: R2={met_tr['recon_r2']:.4f} L0={met_tr['mean_l0']:.1f}")
    print(f"    SAE win: R2={met_win_tr['recon_r2']:.4f} L0={met_win_tr['mean_l0']:.1f}")

    # Compute targets at window level
    targets = compute_human_targets(Y_test, trial_info)

    # 4-level probing: bin_raw, bin_sae, window_raw, window_sae
    probing = {'bin_raw': {}, 'bin_sae': {}, 'window_raw': {}, 'window_sae': {}}

    for name, target_win in targets.items():
        n = min(len(target_win), n_trials)

        # Window-level probing (raw)
        r = sanitized_ridge_delta_r2(H_win_trained[:n], target_win[:n], H_win_untrained[:n])
        probing['window_raw'][name] = r

        # Window-level probing (SAE)
        r = sanitized_ridge_delta_r2(F_win_trained[:n], target_win[:n], F_win_untrained[:n])
        probing['window_sae'][name] = r

        # Bin-level: need to expand target to match bins
        # target_win is (n_trials,) -> repeat T times -> (n_trials*T,)
        target_bin = np.repeat(target_win[:n], T)
        n_bin = min(len(target_bin), H_bin_trained.shape[0])

        r = sanitized_ridge_delta_r2(H_bin_trained[:n_bin], target_bin[:n_bin],
                                     H_bin_untrained[:n_bin])
        probing['bin_raw'][name] = r

        r = sanitized_ridge_delta_r2(F_bin_trained[:n_bin], target_bin[:n_bin],
                                     F_bin_untrained[:n_bin])
        probing['bin_sae'][name] = r

        # Print comparison
        wr = probing['window_raw'][name]['delta_r2']
        ws = probing['window_sae'][name]['delta_r2']
        br = probing['bin_raw'][name]['delta_r2']
        bs = probing['bin_sae'][name]['delta_r2']
        print(f"    {name:25s}: bin_raw={br:+.3f} bin_sae={bs:+.3f} "
              f"win_raw={wr:+.3f} win_sae={ws:+.3f}")

    result = {
        'circuit': 'Circuit 4 (Human WM)',
        'patient_id': patient_id,
        'cc': cc,
        'hidden_size': hidden_size,
        'n_trials': n_trials,
        'n_bins_per_trial': T,
        'input_dim': input_dim,
        'output_dim': output_dim,
        'sae_config': {
            'expansion_factor': best['expansion_factor'],
            'sparsity_coeff': best['sparsity_coeff'],
        },
        'sae_metrics': {
            'bin_trained': met_tr,
            'bin_untrained': met_un,
            'win_trained': met_win_tr,
            'win_untrained': met_win_un,
        },
        'probing': probing,
    }

    return result


def find_nwb_for_patient(patient_id, raw_dir):
    """Map patient directory name to NWB file path."""
    # Patient dirs are like sub-1_ses-1_ecephys+image
    # NWB files are in raw/000469/sub-1/sub-1_ses-1_ecephys+image.nwb
    nwb_name = f"{patient_id}.nwb"
    for nwb in raw_dir.rglob(nwb_name):
        return nwb
    # Try matching by sub-ID only
    sub_id = patient_id.split('_')[0]
    for nwb in raw_dir.rglob(f"{sub_id}*.nwb"):
        return nwb
    return None


def main():
    from human_wm.config import RAW_NWB_DIR, load_nwb_schema

    schema = load_nwb_schema()
    if schema is None:
        print("ERROR: NWB schema not found")
        return

    # Find all patient directories with checkpoints
    patient_dirs = sorted([d for d in CHECKPOINT_DIR.iterdir() if d.is_dir()])
    print(f"Found {len(patient_dirs)} patient directories")

    all_results = {}

    for patient_dir in patient_dirs:
        patient_id = patient_dir.name

        # Find NWB file
        nwb_path = find_nwb_for_patient(patient_id, RAW_NWB_DIR)
        if nwb_path is None:
            print(f"  SKIP {patient_id}: NWB file not found")
            continue

        result = run_sae_for_patient(patient_dir, nwb_path, schema)
        if result is None:
            continue

        # Save per-patient result
        out_dir = SAE_RESULTS_DIR / f"circuit4_{patient_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "sae_probing.json", 'w') as f:
            json.dump(numpy_to_python(result), f, indent=2)

        all_results[patient_id] = result

    # Summary
    print(f"\n{'='*60}")
    print("Circuit 4 SAE Summary")
    print(f"{'='*60}")

    summary = {}
    for pid, res in all_results.items():
        print(f"\n  {pid} (CC={res['cc']}):")
        s = {'cc': res['cc']}
        for level in ['bin_raw', 'bin_sae', 'window_raw', 'window_sae']:
            for name, probe_res in res['probing'][level].items():
                key = f"{level}_{name}"
                s[key] = probe_res['delta_r2']
        summary[pid] = s

        # Print key targets
        for name in res['probing']['window_raw']:
            wr = res['probing']['window_raw'][name]['delta_r2']
            ws = res['probing']['window_sae'][name]['delta_r2']
            diff = ws - wr
            arrow = "^" if diff > 0.02 else ("v" if diff < -0.02 else "=")
            print(f"    {name:25s}: win_raw={wr:+.3f} -> win_sae={ws:+.3f} [{arrow}]")

    with open(SAE_RESULTS_DIR / "circuit4_summary.json", 'w') as f:
        json.dump(numpy_to_python(summary), f, indent=2)
    print(f"\n  Summary saved: {SAE_RESULTS_DIR / 'circuit4_summary.json'}")


if __name__ == '__main__':
    main()
