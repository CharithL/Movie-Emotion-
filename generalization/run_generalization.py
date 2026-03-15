"""
DESCARTES Generalization Testing -- All 5 Levels

Tests whether trained LSTM surrogates generalize beyond training data.
Five levels from temporal splits to perturbation robustness.

Key metric: Generalization Ratio = CC_novel / CC_trained
  > 0.8  = STRONG (prosthetic-grade)
  0.5-0.8 = PARTIAL (needs improvement)
  < 0.5  = FAILURE (memorized training distribution)

Run from 'movie emotion' directory:
    python -X utf8 generalization/run_generalization.py
"""

import gc
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings('ignore')

# Paths
MOVIE_DIR = Path("C:/Users/chari/OneDrive/Documents/Descartes_Cogito/movie emotion")
WM_DIR = Path("C:/Users/chari/OneDrive/Documents/Descartes_Cogito/Working memory")
PREPROCESSED_DIR = MOVIE_DIR / "preprocessed_data"
RESULTS_DIR = MOVIE_DIR / "results" / "generalization"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_DIR = WM_DIR / "data" / "results" / "cross_patient" / "cross_patient"
RAW_NWB_DIR = WM_DIR / "data" / "raw" / "000469"

# Use CPU locally to avoid GPU OOM crashes; CUDA on cloud instances
import os
DEVICE = 'cuda' if (torch.cuda.is_available() and os.environ.get('ALLOW_GPU', '0') == '1') else 'cpu'


def cleanup_memory():
    """Aggressively free memory between levels."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Add project paths
sys.path.insert(0, str(MOVIE_DIR))
sys.path.insert(0, str(WM_DIR))


# ================================================================
# UTILITIES
# ================================================================

def compute_cc(y_pred, y_true):
    """Mean per-neuron Pearson correlation across time and trials."""
    pred_flat = y_pred.reshape(-1, y_pred.shape[-1])
    true_flat = y_true.reshape(-1, y_true.shape[-1])
    ccs = []
    for j in range(pred_flat.shape[1]):
        p, t = pred_flat[:, j], true_flat[:, j]
        if np.std(p) < 1e-10 or np.std(t) < 1e-10:
            continue
        r = np.corrcoef(p, t)[0, 1]
        if np.isfinite(r):
            ccs.append(r)
    return float(np.mean(ccs)) if ccs else 0.0


def train_and_evaluate(model, X_tr, Y_tr, X_val, Y_val, X_test, Y_test,
                       n_epochs=200, patience=20, lr=1e-3, device=DEVICE):
    """Train model with early stopping, return CC on val (same) and test (novel)."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_tr_t = torch.FloatTensor(X_tr).to(device)
    Y_tr_t = torch.FloatTensor(Y_tr).to(device)

    best_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(n_epochs):
        model.train()

        # Mini-batch training
        n = X_tr_t.shape[0]
        perm = torch.randperm(n)
        batch_size = min(32, n)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            optimizer.zero_grad()
            y_pred = model(X_tr_t[idx])
            if isinstance(y_pred, tuple):
                y_pred = y_pred[0]
            loss = criterion(y_pred, Y_tr_t[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        epoch_loss /= max(n_batches, 1)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.train(False)

    with torch.no_grad():
        y_val = model(torch.FloatTensor(X_val).to(device))
        if isinstance(y_val, tuple):
            y_val = y_val[0]
        cc_same = compute_cc(y_val.cpu().numpy(), Y_val)

        y_test = model(torch.FloatTensor(X_test).to(device))
        if isinstance(y_test, tuple):
            y_test = y_test[0]
        cc_novel = compute_cc(y_test.cpu().numpy(), Y_test)

    return cc_same, cc_novel


# ================================================================
# LEVEL 1: CROSS-CONDITION (TEMPORAL SPLIT)
# ================================================================

def level1_circuit5(train_fraction=0.6, n_seeds=5):
    """Level 1 for Circuit 5: train on early movie, test on late movie."""
    from phase1_foundation import LimbicPrefrontalLSTM

    print("\n" + "=" * 70)
    print("LEVEL 1: Cross-Condition — Circuit 5 (Movie Temporal Split)")
    print("=" * 70)

    data = np.load(PREPROCESSED_DIR / 'sub-CS48.npz')

    # Concatenate all splits to get full timeline in order
    X_all = np.concatenate([data['X_train'], data['X_val'], data['X_test']], axis=0)
    Y_all = np.concatenate([data['Y_train'], data['Y_val'], data['Y_test']], axis=0)
    n_input = int(data['n_input'])
    n_output = int(data['n_output'])

    n_total = len(X_all)
    n_train_end = int(train_fraction * n_total)

    # Training period: first 60%
    X_early = X_all[:n_train_end]
    Y_early = Y_all[:n_train_end]

    # Novel test: last 40%
    X_late = X_all[n_train_end:]
    Y_late = Y_all[n_train_end:]

    # Hold out 20% of early for validation (reference CC)
    n_val = int(0.2 * len(X_early))
    X_tr = X_early[:len(X_early) - n_val]
    Y_tr = Y_early[:len(Y_early) - n_val]
    X_val_same = X_early[len(X_early) - n_val:]
    Y_val_same = Y_early[len(Y_early) - n_val:]

    print(f"  Train: {len(X_tr)} windows (early movie)")
    print(f"  Val (same period): {len(X_val_same)} windows")
    print(f"  Test (novel period): {len(X_late)} windows")

    results_per_seed = []
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = LimbicPrefrontalLSTM(n_input, n_output, hidden_size=128)
        cc_same, cc_novel = train_and_evaluate(
            model, X_tr, Y_tr, X_val_same, Y_val_same, X_late, Y_late,
            n_epochs=200, patience=20
        )
        ratio = cc_novel / max(cc_same, 0.01)
        results_per_seed.append({
            'seed': seed, 'cc_same': cc_same, 'cc_novel': cc_novel,
            'ratio': ratio,
        })
        print(f"  Seed {seed}: CC_same={cc_same:.3f}, CC_novel={cc_novel:.3f}, "
              f"ratio={ratio:.3f}")

    ratios = [r['ratio'] for r in results_per_seed]
    mean_ratio = float(np.mean(ratios))
    status = 'STRONG' if mean_ratio > 0.8 else 'PARTIAL' if mean_ratio > 0.5 else 'FAILURE'

    result = {
        'circuit': 'circuit5_CS48',
        'test': 'temporal_split',
        'train_fraction': train_fraction,
        'train_windows': len(X_tr),
        'val_windows': len(X_val_same),
        'test_windows': len(X_late),
        'cc_same_mean': float(np.mean([r['cc_same'] for r in results_per_seed])),
        'cc_novel_mean': float(np.mean([r['cc_novel'] for r in results_per_seed])),
        'ratio_mean': mean_ratio,
        'ratio_std': float(np.std(ratios)),
        'status': status,
        'per_seed': results_per_seed,
    }

    print(f"\n  Circuit 5 Level 1: ratio={mean_ratio:.3f} ({status})")
    return result


def level1_circuit4(n_seeds=5, train_fraction=0.6):
    """Level 1 for Circuit 4: train on early trials, test on late trials."""
    from human_wm.surrogate.models import HumanLSTMSurrogate
    from human_wm.data.nwb_loader import extract_patient_data
    from human_wm.config import load_nwb_schema

    print("\n" + "=" * 70)
    print("LEVEL 1: Cross-Condition — Circuit 4 (Temporal Split)")
    print("=" * 70)

    schema = load_nwb_schema()

    # Use patients with highest CC and downloaded NWBs
    test_patients = ['sub-10_ses-1_ecephys+image', 'sub-4_ses-1_ecephys+image',
                     'sub-8_ses-1_ecephys+image']

    all_patient_results = []

    for patient_id in test_patients:
        nwb_path = None
        for p in RAW_NWB_DIR.rglob(f"{patient_id}.nwb"):
            nwb_path = p
            break
        if nwb_path is None:
            continue

        # Check CC
        ckpt_dir = CHECKPOINT_DIR / patient_id
        result_path = ckpt_dir / 'results_lstm_h128_s0.json'
        cc_ref = 0.0
        if result_path.exists():
            with open(result_path) as f:
                cc_ref = json.load(f).get('cc', 0)

        print(f"\n  Patient: {patient_id} (CC_ref={cc_ref:.3f})")

        X, Y, trial_info = extract_patient_data(nwb_path, schema)
        n_trials, T, n_input = X.shape
        n_output = Y.shape[2]

        n_train_end = int(train_fraction * n_trials)
        X_early = X[:n_train_end]
        Y_early = Y[:n_train_end]
        X_late = X[n_train_end:]
        Y_late = Y[n_train_end:]

        n_val = int(0.2 * len(X_early))
        X_tr = X_early[:len(X_early) - n_val]
        Y_tr = Y_early[:len(Y_early) - n_val]
        X_val = X_early[len(X_early) - n_val:]
        Y_val = Y_early[len(Y_early) - n_val:]

        print(f"    Data: ({n_trials}, {T}, {n_input}) -> ({n_trials}, {T}, {n_output})")
        print(f"    Train: {len(X_tr)}, Val: {len(X_val)}, Test (novel): {len(X_late)}")

        seed_results = []
        for seed in range(n_seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = HumanLSTMSurrogate(n_input, n_output, hidden_size=128)
            cc_same, cc_novel = train_and_evaluate(
                model, X_tr, Y_tr, X_val, Y_val, X_late, Y_late,
                n_epochs=200, patience=20
            )
            ratio = cc_novel / max(cc_same, 0.01)
            seed_results.append({
                'seed': seed, 'cc_same': cc_same, 'cc_novel': cc_novel,
                'ratio': ratio,
            })
            print(f"    Seed {seed}: CC_same={cc_same:.3f}, CC_novel={cc_novel:.3f}, "
                  f"ratio={ratio:.3f}")

        ratios = [r['ratio'] for r in seed_results]
        mean_ratio = float(np.mean(ratios))
        status = 'STRONG' if mean_ratio > 0.8 else 'PARTIAL' if mean_ratio > 0.5 else 'FAILURE'

        all_patient_results.append({
            'patient_id': patient_id,
            'cc_reference': cc_ref,
            'n_trials': n_trials,
            'n_input': n_input,
            'n_output': n_output,
            'cc_same_mean': float(np.mean([r['cc_same'] for r in seed_results])),
            'cc_novel_mean': float(np.mean([r['cc_novel'] for r in seed_results])),
            'ratio_mean': mean_ratio,
            'ratio_std': float(np.std(ratios)),
            'status': status,
            'per_seed': seed_results,
        })
        print(f"    {patient_id}: ratio={mean_ratio:.3f} ({status})")

    overall_ratios = [r['ratio_mean'] for r in all_patient_results]
    overall_mean = float(np.mean(overall_ratios)) if overall_ratios else 0.0
    overall_status = 'STRONG' if overall_mean > 0.8 else 'PARTIAL' if overall_mean > 0.5 else 'FAILURE'

    return {
        'circuit': 'circuit4_human_wm',
        'test': 'temporal_split',
        'train_fraction': train_fraction,
        'n_patients_tested': len(all_patient_results),
        'overall_ratio_mean': overall_mean,
        'overall_status': overall_status,
        'per_patient': all_patient_results,
    }


# ================================================================
# LEVEL 2: CROSS-SESSION
# ================================================================

def level2_cross_session(n_seeds=3):
    """Train on session 1, test on session 2 for patients with both sessions."""
    from human_wm.surrogate.models import HumanLSTMSurrogate
    from human_wm.data.nwb_loader import extract_patient_data
    from human_wm.config import load_nwb_schema

    print("\n" + "=" * 70)
    print("LEVEL 2: Cross-Session Generalization")
    print("=" * 70)

    schema = load_nwb_schema()

    # Find patients with both ses-1 and ses-2 NWBs downloaded
    patients_with_both = []
    for sub_dir in sorted(RAW_NWB_DIR.iterdir()):
        if not sub_dir.is_dir():
            continue
        nwbs = sorted(sub_dir.glob('*.nwb'))
        ses1 = [n for n in nwbs if '_ses-1_' in n.name]
        ses2 = [n for n in nwbs if '_ses-2_' in n.name]
        if ses1 and ses2:
            patients_with_both.append((sub_dir.name, ses1[0], ses2[0]))

    if not patients_with_both:
        print("  No patients with both sessions downloaded locally.")
        print("  Skipping Level 2 — download ses-2 NWBs manually first.")
        return {
            'level': 2, 'test': 'cross_session',
            'status': 'SKIPPED', 'reason': 'no_ses2_nwbs_downloaded',
            'n_patients': 0,
        }

    print(f"  Patients with both sessions: {len(patients_with_both)}")

    all_results = []

    for sub_name, ses1_path, ses2_path in patients_with_both:
        print(f"\n  Patient: {sub_name}")

        try:
            X1, Y1, info1 = extract_patient_data(ses1_path, schema)
            X2, Y2, info2 = extract_patient_data(ses2_path, schema)
        except Exception as e:
            print(f"    Error loading: {e}")
            continue

        n1_in, n1_out = X1.shape[2], Y1.shape[2]
        n2_in, n2_out = X2.shape[2], Y2.shape[2]

        # Use minimum neuron count (population-level approach)
        n_in = min(n1_in, n2_in)
        n_out = min(n1_out, n2_out)

        if n_in < 2 or n_out < 2:
            print(f"    SKIP: insufficient neurons (in={n_in}, out={n_out})")
            continue

        # Truncate to shared dimensions (first N neurons)
        X1_s, Y1_s = X1[:, :, :n_in], Y1[:, :, :n_out]
        X2_s, Y2_s = X2[:, :, :n_in], Y2[:, :, :n_out]

        print(f"    Ses-1: {X1_s.shape} -> {Y1_s.shape}")
        print(f"    Ses-2: {X2_s.shape} -> {Y2_s.shape}")

        # Train on ses-1, validate on held-out ses-1, test on ses-2
        n_val = int(0.2 * len(X1_s))
        X_tr = X1_s[:len(X1_s) - n_val]
        Y_tr = Y1_s[:len(Y1_s) - n_val]
        X_val = X1_s[len(X1_s) - n_val:]
        Y_val = Y1_s[len(Y1_s) - n_val:]

        seed_results = []
        for seed in range(n_seeds):
            torch.manual_seed(seed)
            model = HumanLSTMSurrogate(n_in, n_out, hidden_size=128)
            cc_same, cc_novel = train_and_evaluate(
                model, X_tr, Y_tr, X_val, Y_val, X2_s, Y2_s,
                n_epochs=200, patience=20
            )
            ratio = cc_novel / max(cc_same, 0.01)
            seed_results.append({'seed': seed, 'cc_same': cc_same,
                                 'cc_novel': cc_novel, 'ratio': ratio})
            print(f"    Seed {seed}: CC_ses1={cc_same:.3f}, CC_ses2={cc_novel:.3f}, "
                  f"ratio={ratio:.3f}")

        ratios = [r['ratio'] for r in seed_results]
        mean_ratio = float(np.mean(ratios))
        status = 'STRONG' if mean_ratio > 0.8 else 'PARTIAL' if mean_ratio > 0.5 else 'FAILURE'

        all_results.append({
            'patient': sub_name,
            'n_shared_input': n_in,
            'n_shared_output': n_out,
            'n_trials_ses1': len(X1_s),
            'n_trials_ses2': len(X2_s),
            'ratio_mean': mean_ratio,
            'status': status,
            'per_seed': seed_results,
        })
        print(f"    {sub_name}: ratio={mean_ratio:.3f} ({status})")

    overall_ratios = [r['ratio_mean'] for r in all_results]
    overall_mean = float(np.mean(overall_ratios)) if overall_ratios else 0.0

    return {
        'level': 2,
        'test': 'cross_session',
        'n_patients': len(all_results),
        'overall_ratio_mean': overall_mean,
        'per_patient': all_results,
    }


# ================================================================
# LEVEL 3: CROSS-PATIENT (PatientAgnosticSurrogate)
# ================================================================

class PatientAgnosticSurrogate(nn.Module):
    """Surrogate with population embedding for variable neuron counts."""

    def __init__(self, embed_dim=32, hidden_size=128, n_layers=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size

        self.neuron_encoder = nn.Sequential(
            nn.Linear(1, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.lstm = nn.LSTM(
            input_size=embed_dim, hidden_size=hidden_size,
            num_layers=n_layers, batch_first=True,
            dropout=0.1 if n_layers > 1 else 0,
        )
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, embed_dim), nn.ReLU(),
        )
        self.neuron_decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, x, n_output_neurons=None):
        batch, time, n_input = x.shape
        # Embed each neuron independently
        x_flat = x.reshape(-1, 1)
        embedded = self.neuron_encoder(x_flat)
        embedded = embedded.reshape(batch, time, n_input, self.embed_dim)
        # Average across neurons
        pop_vector = embedded.mean(dim=2)
        # LSTM
        lstm_out, _ = self.lstm(pop_vector)
        # Decode
        out_embed = self.output_proj(lstm_out)
        if n_output_neurons is None:
            n_output_neurons = n_input
        out_expand = out_embed.unsqueeze(2).expand(-1, -1, n_output_neurons, -1)
        y = self.neuron_decoder(out_expand).squeeze(-1)
        return y


def _get_patient_manifest():
    """Build manifest of patient paths and reference CCs without loading data."""
    manifest = []
    for sub_dir in sorted(RAW_NWB_DIR.iterdir()):
        if not sub_dir.is_dir():
            continue
        nwbs = sorted(sub_dir.glob('*_ses-1_*.nwb'))
        if not nwbs:
            continue
        pid = nwbs[0].stem
        ckpt_dir = CHECKPOINT_DIR / pid
        if not ckpt_dir.exists():
            continue
        result_path = ckpt_dir / 'results_lstm_h128_s0.json'
        cc_ref = 0.0
        if result_path.exists():
            with open(result_path) as f:
                cc_ref = json.load(f).get('cc', 0)
        if cc_ref < 0.3:
            continue
        manifest.append({'pid': pid, 'nwb_path': nwbs[0], 'cc_ref': cc_ref})
    return manifest


def _load_one_patient(nwb_path, schema):
    """Load a single patient, return X, Y arrays."""
    from human_wm.data.nwb_loader import extract_patient_data
    X, Y, info = extract_patient_data(nwb_path, schema)
    return X, Y


def _preload_all_patients(manifest, schema, max_trials=64):
    """Load all patient data ONCE into RAM, cap trials to limit memory.
    sub-11 has 378 trials × 1247 bins = 177 MB — we subsample trials."""
    cache = {}
    for m in manifest:
        try:
            X, Y = _load_one_patient(m['nwb_path'], schema)
            # Cap trials to avoid memory explosion (sub-11: 378×1247×47)
            if X.shape[0] > max_trials:
                idx = np.linspace(0, X.shape[0]-1, max_trials, dtype=int)
                X, Y = X[idx], Y[idx]
            cache[m['pid']] = (X, Y)
            mb = (X.nbytes + Y.nbytes) / 1e6
            print(f"    Cached {m['pid']}: {X.shape} + {Y.shape} = {mb:.1f} MB")
        except Exception as e:
            print(f"    Failed to load {m['pid']}: {e}")
    return cache


def level3_cross_patient(n_epochs=50):
    """Leave-one-patient-out with PatientAgnosticSurrogate.
    Pre-loads all data to avoid repeated NWB reads."""
    from human_wm.config import load_nwb_schema

    print("\n" + "=" * 70)
    print("LEVEL 3: Cross-Patient Generalization")
    print("=" * 70)

    schema = load_nwb_schema()
    manifest = _get_patient_manifest()
    n_patients = len(manifest)
    print(f"  Total eligible patients: {n_patients}")

    if n_patients < 3:
        print("  Too few patients for leave-one-out. Skipping.")
        return {'level': 3, 'status': 'SKIPPED', 'reason': 'too_few_patients'}

    # Limit to top 6 patients by CC to keep runtime reasonable
    manifest = sorted(manifest, key=lambda m: m['cc_ref'], reverse=True)[:6]
    n_patients = len(manifest)
    print(f"  Using top {n_patients} patients by CC")

    # Pre-load ALL patient data once (fixes the 250-NWB-opens bug)
    print("  Pre-loading patient data...")
    cache = _preload_all_patients(manifest, schema, max_trials=64)
    manifest = [m for m in manifest if m['pid'] in cache]
    n_patients = len(manifest)
    print(f"  Successfully cached {n_patients} patients")

    results = []

    for test_idx, test_entry in enumerate(manifest):
        test_pid = test_entry['pid']
        cc_ref = test_entry['cc_ref']
        train_entries = [m for i, m in enumerate(manifest) if i != test_idx]

        print(f"\n  Test: {test_pid} (CC_ref={cc_ref:.3f})")

        X_test, Y_test = cache[test_pid]
        n_out_test = Y_test.shape[2]

        model = PatientAgnosticSurrogate(embed_dim=32, hidden_size=128).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        best_loss = float('inf')
        best_state = None

        for epoch in range(n_epochs):
            model.train()
            epoch_loss = 0.0
            n_trained = 0

            for train_entry in train_entries:
                if train_entry['pid'] not in cache:
                    continue
                X_tr, Y_tr = cache[train_entry['pid']]
                n_out = Y_tr.shape[-1]

                # Mini-batch to avoid GPU OOM on large patients
                batch_size = min(16, X_tr.shape[0])
                perm = np.random.permutation(X_tr.shape[0])[:batch_size]

                X = torch.FloatTensor(X_tr[perm]).to(DEVICE)
                Y = torch.FloatTensor(Y_tr[perm]).to(DEVICE)

                optimizer.zero_grad()
                y_pred = model(X, n_output_neurons=n_out)
                loss = criterion(y_pred, Y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_trained += 1

                del X, Y

            if n_trained > 0:
                epoch_loss /= n_trained
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)
        model.train(False)

        # Evaluate in batches to avoid OOM on sub-11
        with torch.no_grad():
            all_preds = []
            batch_sz = 16
            for i in range(0, len(X_test), batch_sz):
                X_b = torch.FloatTensor(X_test[i:i+batch_sz]).to(DEVICE)
                yp = model(X_b, n_output_neurons=n_out_test).cpu().numpy()
                all_preds.append(yp)
                del X_b
            y_pred = np.concatenate(all_preds, axis=0)
            cc_novel = compute_cc(y_pred, Y_test)

        gen_ratio = cc_novel / max(cc_ref, 0.01)

        results.append({
            'test_patient': test_pid,
            'n_train_patients': len(train_entries),
            'cc_novel': float(cc_novel),
            'cc_reference': float(cc_ref),
            'generalization_ratio': float(gen_ratio),
        })
        print(f"    CC_novel={cc_novel:.3f}, CC_ref={cc_ref:.3f}, "
              f"ratio={gen_ratio:.3f}")

        del model, best_state
        cleanup_memory()

    # Free the cache
    del cache
    cleanup_memory()

    ratios = [r['generalization_ratio'] for r in results]
    mean_ratio = float(np.mean(ratios)) if ratios else 0.0
    status = 'STRONG' if mean_ratio > 0.8 else 'PARTIAL' if mean_ratio > 0.5 else 'FAILURE'

    return {
        'level': 3,
        'test': 'leave_one_patient_out',
        'n_patients': n_patients,
        'overall_ratio_mean': mean_ratio,
        'overall_status': status,
        'per_patient': results,
    }


# ================================================================
# LEVEL 5: PERTURBATION ROBUSTNESS
# ================================================================

def level5_perturbation(model, X_test, Y_test, label=''):
    """Run all three perturbation tests on a trained model."""

    print(f"\n  {'='*60}")
    print(f"  Perturbation Tests: {label}")
    print(f"  {'='*60}")

    model.train(False)
    model = model.to(DEVICE)

    # Baseline CC
    with torch.no_grad():
        y_base = model(torch.FloatTensor(X_test).to(DEVICE))
        if isinstance(y_base, tuple):
            y_base = y_base[0]
        y_base = y_base.cpu().numpy()
    cc_baseline = compute_cc(y_base, Y_test)
    print(f"  Baseline CC: {cc_baseline:.3f}")

    results = {'baseline_cc': float(cc_baseline)}

    # --- Neuron Dropout ---
    print("\n  Neuron Dropout:")
    dropout_results = []
    for frac in [0.1, 0.2, 0.3, 0.5]:
        n_input = X_test.shape[-1]
        n_drop = max(1, int(frac * n_input))
        ccs = []
        for rep in range(10):
            X_c = X_test.copy()
            drop_idx = np.random.choice(n_input, n_drop, replace=False)
            X_c[:, :, drop_idx] = 0.0
            with torch.no_grad():
                yp = model(torch.FloatTensor(X_c).to(DEVICE))
                if isinstance(yp, tuple):
                    yp = yp[0]
                cc = compute_cc(yp.cpu().numpy(), Y_test)
            ccs.append(cc)
        retention = np.mean(ccs) / max(cc_baseline, 0.01)
        dropout_results.append({
            'fraction': frac, 'n_dropped': n_drop,
            'cc_mean': float(np.mean(ccs)), 'retention': float(retention),
            'robust': retention > 0.8,
        })
        print(f"    {frac:.0%} dropout: CC={np.mean(ccs):.3f} "
              f"(retention={retention:.3f}) {'OK' if retention > 0.8 else 'DEGRADED'}")
    results['neuron_dropout'] = dropout_results

    # --- Noise Injection ---
    print("\n  Noise Injection:")
    noise_results = []
    signal_power = float(np.mean(X_test ** 2)) + 1e-10
    for snr in [20, 10, 5, 2, 1]:
        noise_power = signal_power / snr
        ccs = []
        for rep in range(10):
            noise = np.random.randn(*X_test.shape) * np.sqrt(noise_power)
            X_n = X_test + noise.astype(np.float32)
            with torch.no_grad():
                yp = model(torch.FloatTensor(X_n).to(DEVICE))
                if isinstance(yp, tuple):
                    yp = yp[0]
                cc = compute_cc(yp.cpu().numpy(), Y_test)
            ccs.append(cc)
        retention = np.mean(ccs) / max(cc_baseline, 0.01)
        noise_results.append({
            'snr': snr, 'cc_mean': float(np.mean(ccs)),
            'retention': float(retention),
        })
        print(f"    SNR={snr:2d}: CC={np.mean(ccs):.3f} (retention={retention:.3f})")
    results['noise_injection'] = noise_results

    # --- Gain Drift ---
    print("\n  Gain Drift:")
    drift_results = []
    n_windows = X_test.shape[0]
    n_input = X_test.shape[-1]
    for mag in [0.1, 0.2, 0.5, 1.0]:
        X_d = X_test.copy()
        for w in range(n_windows):
            gain = 1.0 + mag * (w / max(n_windows - 1, 1))
            per_neuron = gain * (1.0 + 0.1 * np.random.randn(n_input))
            X_d[w] *= per_neuron.astype(np.float32)
        with torch.no_grad():
            yp = model(torch.FloatTensor(X_d).to(DEVICE))
            if isinstance(yp, tuple):
                yp = yp[0]
            cc = compute_cc(yp.cpu().numpy(), Y_test)
        retention = cc / max(cc_baseline, 0.01)
        drift_results.append({
            'magnitude': mag, 'cc': float(cc), 'retention': float(retention),
        })
        print(f"    Drift {mag}: CC={cc:.3f} (retention={retention:.3f})")
    results['gain_drift'] = drift_results

    return results


def level5_circuit5():
    """Level 5 for Circuit 5 CS48."""
    from phase1_foundation import LimbicPrefrontalLSTM

    print("\n" + "=" * 70)
    print("LEVEL 5: Perturbation Robustness — Circuit 5 CS48")
    print("=" * 70)

    data = np.load(PREPROCESSED_DIR / 'sub-CS48.npz')
    X_test = data['X_test']
    Y_test = data['Y_test']
    n_input = int(data['n_input'])
    n_output = int(data['n_output'])

    # Load trained model
    ckpt_path = MOVIE_DIR / 'results' / 'sub-CS48' / 'best_model.pt'
    if not ckpt_path.exists():
        # Train a quick model
        print("  Training fresh model for perturbation testing...")
        X_train = data['X_train']
        Y_train = data['Y_train']
        X_val = data['X_val']
        Y_val = data['Y_val']

        torch.manual_seed(0)
        model = LimbicPrefrontalLSTM(n_input, n_output, hidden_size=128)
        train_and_evaluate(model, X_train, Y_train, X_val, Y_val,
                          X_test, Y_test, n_epochs=200, patience=20)
    else:
        model = LimbicPrefrontalLSTM(n_input, n_output, hidden_size=128)
        state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state)

    return level5_perturbation(model, X_test, Y_test, label='Circuit 5 CS48')


def level5_circuit4():
    """Level 5 for Circuit 4 sub-10 (best patient)."""
    from human_wm.surrogate.models import HumanLSTMSurrogate
    from human_wm.data.nwb_loader import extract_patient_data
    from human_wm.config import load_nwb_schema

    print("\n" + "=" * 70)
    print("LEVEL 5: Perturbation Robustness — Circuit 4 sub-10")
    print("=" * 70)

    schema = load_nwb_schema()
    patient_id = 'sub-10_ses-1_ecephys+image'
    nwb_path = RAW_NWB_DIR / 'sub-10' / f'{patient_id}.nwb'

    X, Y, info = extract_patient_data(nwb_path, schema)
    n_input, n_output = X.shape[2], Y.shape[2]

    # Load or train model
    ckpt_path = CHECKPOINT_DIR / patient_id / 'lstm_h128_s0_best.pt'
    state = torch.load(ckpt_path, map_location='cpu', weights_only=True)

    ckpt_input = state['lstm.weight_ih_l0'].shape[1]
    ckpt_output = state['output_proj.weight'].shape[0]

    # Align dimensions
    if n_input != ckpt_input:
        if n_input > ckpt_input:
            X = X[:, :, :ckpt_input]
        else:
            pad = np.zeros((X.shape[0], X.shape[1], ckpt_input - n_input), dtype=np.float32)
            X = np.concatenate([X, pad], axis=2)
        n_input = ckpt_input

    model = HumanLSTMSurrogate(n_input, ckpt_output, hidden_size=128)
    model.load_state_dict(state)

    # Use last 20% as test
    n_test = int(0.2 * len(X))
    X_test = X[-n_test:]
    Y_test = Y[-n_test:, :, :min(Y.shape[2], ckpt_output)]

    return level5_perturbation(model, X_test, Y_test, label='Circuit 4 sub-10')


# ================================================================
# MAIN ORCHESTRATOR
# ================================================================

def run_all():
    print("=" * 70)
    print("DESCARTES GENERALIZATION TESTING — ALL LEVELS")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    t0 = time.time()
    report = {}

    # LEVEL 1: Cross-Condition (Temporal Split)
    print("\n" + "#" * 70)
    print("# LEVEL 1: CROSS-CONDITION GENERALIZATION")
    print("#" * 70)

    l1_c5 = level1_circuit5()
    report['level1_circuit5'] = l1_c5
    cleanup_memory()

    l1_c4 = level1_circuit4()
    report['level1_circuit4'] = l1_c4
    cleanup_memory()

    # Checkpoint
    l1_ratios = [l1_c5['ratio_mean']]
    if l1_c4.get('overall_ratio_mean'):
        l1_ratios.append(l1_c4['overall_ratio_mean'])
    l1_mean = float(np.mean(l1_ratios))

    print(f"\n{'='*70}")
    print(f"LEVEL 1 CHECKPOINT: mean ratio = {l1_mean:.3f}")
    if l1_mean < 0.5:
        print("FAILURE — model is a lookup table. Stopping.")
        report['stopped_at'] = 'level1'
        report['reason'] = f'mean_ratio={l1_mean:.3f} < 0.5'
    else:
        status = 'STRONG' if l1_mean > 0.8 else 'PARTIAL'
        print(f"{status} — proceeding to higher levels.")

    # Save intermediate
    with open(RESULTS_DIR / 'generalization_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)

    if l1_mean < 0.5:
        elapsed = time.time() - t0
        report['elapsed_seconds'] = elapsed
        with open(RESULTS_DIR / 'generalization_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        return report

    # LEVEL 2: Cross-Session
    print("\n" + "#" * 70)
    print("# LEVEL 2: CROSS-SESSION GENERALIZATION")
    print("#" * 70)

    l2 = level2_cross_session()
    report['level2_cross_session'] = l2
    cleanup_memory()

    # LEVEL 3: Cross-Patient
    print("\n" + "#" * 70)
    print("# LEVEL 3: CROSS-PATIENT GENERALIZATION")
    print("#" * 70)

    l3 = level3_cross_patient()
    report['level3_cross_patient'] = l3
    cleanup_memory()

    # LEVEL 4: Cross-Task — NOT TESTABLE
    print("\n" + "#" * 70)
    print("# LEVEL 4: CROSS-TASK GENERALIZATION")
    print("#" * 70)
    print("  NOT TESTABLE: No shared patients between Circuit 4 (DANDI 000469)")
    print("  and Circuit 5 (movie emotion). Different species/brain regions.")
    report['level4_cross_task'] = {
        'status': 'NOT_TESTABLE',
        'reason': 'No shared patients between WM (human MTL/frontal) and movie (human limbic/prefrontal from different dataset)',
    }

    # LEVEL 5: Perturbation Robustness
    print("\n" + "#" * 70)
    print("# LEVEL 5: PERTURBATION ROBUSTNESS")
    print("#" * 70)

    l5_c5 = level5_circuit5()
    report['level5_circuit5'] = l5_c5
    cleanup_memory()

    l5_c4 = level5_circuit4()
    report['level5_circuit4'] = l5_c4
    cleanup_memory()

    # FINAL SUMMARY
    elapsed = time.time() - t0
    report['elapsed_seconds'] = elapsed

    print("\n" + "=" * 70)
    print("GENERALIZATION TESTING COMPLETE")
    print("=" * 70)

    # Overall assessment
    prosthetic_viable = True
    strongest = 0
    weakest = 5

    for level_key, level_data in report.items():
        if not isinstance(level_data, dict):
            continue
        if 'overall_ratio_mean' in level_data or 'ratio_mean' in level_data:
            ratio = level_data.get('overall_ratio_mean', level_data.get('ratio_mean', 0))
            level_num = int(level_key.replace('level', '')[0]) if level_key[0:5] == 'level' else 0
            if ratio < 0.5:
                prosthetic_viable = False
            if ratio > 0.5 and level_num > strongest:
                strongest = level_num
            if ratio < 0.5 and level_num < weakest:
                weakest = level_num

    report['overall_assessment'] = {
        'prosthetic_viable': prosthetic_viable,
        'strongest_generalization_level': strongest,
        'elapsed_minutes': elapsed / 60,
    }

    # Print summary table
    print(f"\n{'Level':<30} {'Ratio':>8} {'Status':>10}")
    print("-" * 52)

    for key in ['level1_circuit5', 'level1_circuit4', 'level2_cross_session',
                'level3_cross_patient', 'level4_cross_task',
                'level5_circuit5', 'level5_circuit4']:
        if key not in report:
            continue
        d = report[key]
        ratio = d.get('ratio_mean', d.get('overall_ratio_mean', None))
        status = d.get('status', d.get('overall_status', 'N/A'))
        if ratio is not None:
            print(f"  {key:<28} {ratio:>8.3f} {status:>10}")
        else:
            print(f"  {key:<28} {'N/A':>8} {status:>10}")

    print(f"\n  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Prosthetic viable: {prosthetic_viable}")

    with open(RESULTS_DIR / 'generalization_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Saved: {RESULTS_DIR / 'generalization_report.json'}")

    return report


if __name__ == '__main__':
    run_all()
