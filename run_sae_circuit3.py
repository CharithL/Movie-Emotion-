"""
DESCARTES SAE Analysis -- Circuit 3 (ALM -> Thalamus, Mouse WM)

Applies Sparse Autoencoders to pre-extracted LSTM hidden states from the
Chen/Svoboda working memory experiment. Since hidden states are already at
window/trial level (79 or 51 trials x hidden_dim), there is no bin vs window
distinction -- we probe at trial level only.

Probe targets (7 total):
  Level B: choice_signal, ramp_signal, population_rate, choice_magnitude
  Level C: delay_stability, theta_modulation, population_synchrony

Run from the 'movie emotion' directory to avoid AppControl DLL issues:
  cd "movie emotion"
  python -X utf8 run_sae_circuit3.py
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

# ---------- paths ----------
WM_DIR = Path("C:/Users/chari/OneDrive/Documents/Descartes_Cogito/Working memory")
HIDDEN_DIR = WM_DIR / "hidden_states"
SAE_RESULTS_DIR = Path("C:/Users/chari/OneDrive/Documents/Descartes_Cogito/movie emotion/results/sae")

SESSIONS = [
    "sub-440956_ses-20190210T155629_behavior+ecephys+ogen",
    "sub-440959_ses-20190223T173853_behavior+ecephys+image+ogen",
]

HIDDEN_SIZES = [64, 128, 256]

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
              epochs=200, batch_size=32, lr=1e-3):
    """Train SAE on hidden states. More epochs for small datasets."""
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

    # Metrics
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
    """Sanitized Ridge probe with z-score, clamp, validity guard."""
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


# ---------- target computation ----------
def compute_proxy_targets(H_trained, H_untrained):
    """
    Compute proxy targets when biological targets are unavailable.
    These are structural properties of the hidden states themselves.
    """
    from sklearn.decomposition import PCA

    targets = {}
    if H_trained.shape[0] >= 5:
        pca = PCA(n_components=min(5, H_trained.shape[1]))
        scores = pca.fit_transform(H_trained)
        targets['PC1_trained'] = scores[:, 0]
        if scores.shape[1] > 1:
            targets['PC2_trained'] = scores[:, 1]
        targets['trial_variance'] = H_trained.var(axis=1)
        targets['trial_norm'] = np.linalg.norm(H_trained, axis=1)

    return targets


# ---------- main ----------
def run_sae_for_session(session_name, hs_label=128):
    print(f"\n{'='*60}")
    print(f"Circuit 3 SAE: {session_name[:15]}... h={hs_label}")
    print(f"{'='*60}")

    # Load hidden states
    trained_path = HIDDEN_DIR / session_name / f"wm_h{hs_label}_trained.npz"
    untrained_path = HIDDEN_DIR / session_name / f"wm_h{hs_label}_untrained.npz"

    if not trained_path.exists() or not untrained_path.exists():
        print(f"  SKIP: hidden states not found for h={hs_label}")
        return None

    H_trained = np.load(trained_path)['hidden_states']
    H_untrained = np.load(untrained_path)['hidden_states']
    hidden_dim = H_trained.shape[1]
    n_samples = H_trained.shape[0]

    print(f"  Hidden states: trained={H_trained.shape}, untrained={H_untrained.shape}")
    print(f"  NOTE: Data is at trial/window level ({n_samples} trials)")

    # SAE sweep
    expansion_factors = [4, 8]
    sparsity_coeffs = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

    sweep_results = []
    print(f"\n  SAE Hyperparameter Sweep (hidden_dim={hidden_dim}):")

    for ef in expansion_factors:
        for sc in sparsity_coeffs:
            _, metrics = train_sae(H_trained, ef, sc, epochs=300)
            sweep_results.append(metrics)
            tag = "*" if metrics['recon_r2'] > 0.90 else " "
            print(f"  {tag} ef={ef} sc={sc:.0e}: R2={metrics['recon_r2']:.4f} "
                  f"L0={metrics['mean_l0']:.1f} dead={metrics['n_dead_features']}")

    # Select best: R2 > 0.90 and lowest L0
    viable = [s for s in sweep_results if s['recon_r2'] > 0.90]
    if not viable:
        viable = sorted(sweep_results, key=lambda s: -s['recon_r2'])[:3]
        print("  WARNING: No config achieved R2 > 0.90, using top-3 by R2")

    best = min(viable, key=lambda s: s['mean_l0'])
    print(f"\n  BEST: ef={best['expansion_factor']} sc={best['sparsity_coeff']:.0e} "
          f"R2={best['recon_r2']:.4f} L0={best['mean_l0']:.1f}")

    # Train SAE on both trained and untrained with best config
    sae_trained, metrics_trained = train_sae(
        H_trained, best['expansion_factor'], best['sparsity_coeff'], epochs=500
    )
    sae_untrained, metrics_untrained = train_sae(
        H_untrained, best['expansion_factor'], best['sparsity_coeff'], epochs=500
    )

    # Extract SAE features
    F_trained = extract_sae_features(sae_trained, H_trained)
    F_untrained = extract_sae_features(sae_untrained, H_untrained)

    print(f"\n  SAE features: trained={F_trained.shape} untrained={F_untrained.shape}")
    print(f"  Trained SAE: R2={metrics_trained['recon_r2']:.4f} L0={metrics_trained['mean_l0']:.1f}")
    print(f"  Untrained SAE: R2={metrics_untrained['recon_r2']:.4f} L0={metrics_untrained['mean_l0']:.1f}")

    # Build results dict
    results = {
        'circuit': 'Circuit 3 (ALM->Thalamus)',
        'session': session_name,
        'hidden_size_label': f'h{hs_label}',
        'hidden_dim': hidden_dim,
        'n_samples': n_samples,
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
        'note': 'Data already at trial/window level. No bin-level dimension exists.',
    }

    # Compute proxy targets from hidden states
    proxy_targets = compute_proxy_targets(H_trained, H_untrained)

    if proxy_targets:
        print(f"\n  Probing {len(proxy_targets)} proxy targets...")

        raw_probe = {}
        sae_probe = {}

        for name, target in proxy_targets.items():
            n = min(len(target), n_samples)
            raw_result = sanitized_ridge_delta_r2(H_trained[:n], target[:n], H_untrained[:n])
            raw_probe[name] = raw_result

            sae_result = sanitized_ridge_delta_r2(F_trained[:n], target[:n], F_untrained[:n])
            sae_probe[name] = sae_result

            raw_dr2 = raw_result['delta_r2']
            sae_dr2 = sae_result['delta_r2']
            diff = sae_dr2 - raw_dr2
            arrow = "^" if diff > 0.02 else ("v" if diff < -0.02 else "=")
            print(f"    {name:25s}: raw={raw_dr2:+.3f} SAE={sae_dr2:+.3f} [{arrow}]")

        results['proxy_probing'] = {
            'raw_ridge': raw_probe,
            'sae_probe': sae_probe,
        }

    # Try to load biological targets from WM package
    try:
        sys.path.insert(0, str(WM_DIR))
        from wm.data.preprocessing import load_processed_session
        from wm.analysis.run_probing import compute_all_targets

        session_data_dir = WM_DIR / 'data' / 'processed' / session_name
        if session_data_dir.exists():
            splits, _ = load_processed_session(session_data_dir)
            Y_test = splits['test']['Y']
            trial_types_test = splits['test']['trial_types']
            targets = compute_all_targets(Y_test, trial_types_test)

            bio_raw = {}
            bio_sae = {}
            print(f"\n  Probing biological targets...")

            for level_name, level_targets in targets.items():
                for name, target in level_targets.items():
                    n = min(len(target), n_samples)
                    raw_r = sanitized_ridge_delta_r2(H_trained[:n], target[:n], H_untrained[:n])
                    sae_r = sanitized_ridge_delta_r2(F_trained[:n], target[:n], F_untrained[:n])
                    bio_raw[name] = raw_r
                    bio_sae[name] = sae_r
                    diff = sae_r['delta_r2'] - raw_r['delta_r2']
                    arrow = "^" if diff > 0.02 else ("v" if diff < -0.02 else "=")
                    print(f"    [{level_name}] {name:25s}: raw={raw_r['delta_r2']:+.3f} "
                          f"SAE={sae_r['delta_r2']:+.3f} [{arrow}]")

            results['biological_probing'] = {
                'raw_ridge': bio_raw,
                'sae_probe': bio_sae,
            }
        else:
            print(f"  No processed data for biological targets (data dir missing)")
            print(f"  Using proxy targets only")
    except Exception as e:
        print(f"  Could not load biological targets: {e}")
        print(f"  Using proxy targets only")

    return results


def numpy_to_python(obj):
    """Convert numpy types to Python natives for JSON serialization."""
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


def main():
    all_results = {}

    for session in SESSIONS:
        session_short = session.split('_')[0]

        for hs in [128]:  # Focus on h128 per the guide
            result = run_sae_for_session(session, hs)
            if result is None:
                continue

            # Save per-session result
            out_dir = SAE_RESULTS_DIR / f"circuit3_{session_short}_h{hs}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "sae_probing.json"

            with open(out_path, 'w') as f:
                json.dump(numpy_to_python(result), f, indent=2)
            print(f"\n  Saved: {out_path}")

            all_results[f"{session_short}_h{hs}"] = result

    # Summary
    print(f"\n{'='*60}")
    print("Circuit 3 SAE Summary")
    print(f"{'='*60}")
    for key, res in all_results.items():
        print(f"\n  {key}:")
        print(f"    SAE: R2={res['sae_metrics']['trained']['recon_r2']:.4f} "
              f"L0={res['sae_metrics']['trained']['mean_l0']:.1f}")
        if 'proxy_probing' in res:
            for name in res['proxy_probing']['raw_ridge']:
                raw_dr2 = res['proxy_probing']['raw_ridge'][name]['delta_r2']
                sae_dr2 = res['proxy_probing']['sae_probe'][name]['delta_r2']
                print(f"    {name:25s}: raw={raw_dr2:+.3f} -> SAE={sae_dr2:+.3f}")

    # Save summary
    summary_path = SAE_RESULTS_DIR / "circuit3_summary.json"
    summary = {}
    for key, res in all_results.items():
        summary[key] = {
            'sae_recon_r2': res['sae_metrics']['trained']['recon_r2'],
            'sae_mean_l0': res['sae_metrics']['trained']['mean_l0'],
        }
        if 'proxy_probing' in res:
            summary[key]['proxy_results'] = {
                name: {
                    'raw_delta_r2': res['proxy_probing']['raw_ridge'][name]['delta_r2'],
                    'sae_delta_r2': res['proxy_probing']['sae_probe'][name]['delta_r2'],
                }
                for name in res['proxy_probing']['raw_ridge']
            }

    with open(summary_path, 'w') as f:
        json.dump(numpy_to_python(summary), f, indent=2)
    print(f"\n  Summary saved: {summary_path}")


if __name__ == '__main__':
    main()
