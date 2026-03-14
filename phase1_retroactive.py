"""
phase1_retroactive.py -- Phase 1 Statistical Hardening for Circuits 2, 3, 4

Runs matched baselines, gap CV, iAAFT significance, and input decodability
retroactively on all circuits with existing hidden states.

Circuit 2: CA3->CA1 (21600 bins, 108 windows, 25 targets) -- CRITICAL: gamma_amp
Circuit 3: ALM->Thalamus (79/51 trials, proxy targets only)
Circuit 4: Human WM (NWB + checkpoints, 5 targets per patient)

Run from "movie emotion" directory.
"""
import numpy as np
import json
import time
import sys
import warnings
from pathlib import Path
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import KFold
from numba import njit
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')

# Paths
MIMO_DIR = Path("C:/Users/chari/OneDrive/Documents/Descartes_Cogito/La Masson 2002/MIMO/hippocampal_mimo")
WM_DIR = Path("C:/Users/chari/OneDrive/Documents/Descartes_Cogito/Working memory")
OUTPUT_DIR = Path("C:/Users/chari/OneDrive/Documents/Descartes_Cogito/movie emotion/results/statistical_hardening")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Import iAAFT helpers from Phase 1
sys.path.insert(0, str(Path("C:/Users/chari/OneDrive/Documents/Descartes_Cogito/movie emotion")))
from phase1_foundation import (
    _rank_reorder, iaaft_surrogate, shuffled_cv_r2, temporal_cv_r2,
    to_window_level,
)

N_SURROGATES = 1000


# ================================================================
# SHARED PHASE 1 LOGIC
# ================================================================

def _one_surrogate_generic(h_trained, h_untrained, target, rng_seed):
    """Single iAAFT surrogate: phase-randomize target, probe, get null dR2."""
    np.random.seed(rng_seed)
    surr = iaaft_surrogate(target)
    s = surr.astype(np.float64)
    s_std = np.std(s)
    if s_std < 1e-8:
        return 0.0
    s = (s - np.mean(s)) / s_std
    s = np.clip(s, -3.0, 3.0)
    r2_tr, _ = shuffled_cv_r2(h_trained, s, seed=rng_seed % (2**31))
    r2_un, _ = shuffled_cv_r2(h_untrained, s, seed=rng_seed % (2**31))
    return r2_tr - r2_un


def run_iaaft_significance(h_trained, h_untrained, target, n_surrogates=200, n_jobs=-1):
    """iAAFT significance test. All inputs must be at probe level (window/trial)."""
    # Observed
    t = target.astype(np.float64)
    t_std = np.std(t)
    if t_std < 1e-8:
        return {'observed_delta_r2': 0.0, 'r2_trained': 0.0, 'r2_untrained': 0.0,
                'null_mean': 0.0, 'null_std': 0.0, 'null_percentiles': {'5':0,'50':0,'95':0,'99':0},
                'p_value': 1.0, 'z_score': 0.0, 'significant_005': False, 'significant_001': False,
                'n_surrogates': 0}
    t = (t - np.mean(t)) / t_std
    t = np.clip(t, -3.0, 3.0)

    r2_tr, _ = shuffled_cv_r2(h_trained, t)
    r2_un, _ = shuffled_cv_r2(h_untrained, t)
    observed_delta = r2_tr - r2_un

    # Warm up numba
    _ = iaaft_surrogate(t[:50] if len(t) > 50 else t)

    rng_seeds = np.random.randint(0, 2**31, size=n_surrogates)
    null_deltas = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_one_surrogate_generic)(h_trained, h_untrained, t, int(s))
        for s in rng_seeds
    )
    null_deltas = np.array(null_deltas)

    p_value = float(np.mean(null_deltas >= observed_delta))
    null_mean = float(np.mean(null_deltas))
    null_std = float(np.std(null_deltas))
    z_score = (observed_delta - null_mean) / max(null_std, 1e-10)

    return {
        'observed_delta_r2': float(observed_delta),
        'r2_trained': float(r2_tr),
        'r2_untrained': float(r2_un),
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


def run_gap_cv_diagnostic(h_trained, target, gap_windows=2):
    """Gap CV diagnostic: compare shuffled vs temporal CV."""
    t = target.astype(np.float64)
    t_std = np.std(t)
    if t_std < 1e-8:
        return {'r2_shuffled': 0.0, 'r2_temporal': 0.0, 'inflation': 0.0}
    t = (t - np.mean(t)) / t_std
    t = np.clip(t, -3.0, 3.0)

    r2_shuffled, _ = shuffled_cv_r2(h_trained, t)
    r2_temporal, _ = temporal_cv_r2(h_trained, t, gap_windows=gap_windows)
    inflation = r2_shuffled - r2_temporal

    return {
        'r2_shuffled': float(r2_shuffled),
        'r2_temporal': float(r2_temporal),
        'inflation': float(inflation),
    }


# ================================================================
# CIRCUIT 2: CA3->CA1 (gamma_amp cornerstone)
# ================================================================

# Circuit 2 constants
C2_N_WINDOWS = 108
C2_STEPS_PER_WINDOW_HIDDEN = 200
C2_STEPS_PER_WINDOW_BIO = 5000
C2_DOWNSAMPLE_RATIO = C2_STEPS_PER_WINDOW_BIO // C2_STEPS_PER_WINDOW_HIDDEN  # 25

C2_BIOLOGY_DIR = MIMO_DIR / 'checkpoints' / 'biology_test'

C2_ALL_TARGETS = [
    'I_Na_CA1', 'I_KDR_CA1', 'I_Ka_CA1', 'I_h_CA1', 'I_CaL_CA1',
    'I_CaN_CA1', 'I_KCa_CA1', 'I_M_CA1', 'I_AHP_CA1',
    'Ca_i_CA1',
    'g_AMPA_SC', 'g_NMDA_SC', 'g_GABA_A', 'g_GABA_B',
    'h_Na_CA1', 'n_KDR_CA1', 'h_Ka_CA1', 'm_h_CA1', 'm_CaL_CA1', 'm_M_CA1',
    'V_basket', 'V_OLM',
    'theta_phase', 'gamma_amp',
    'V_CA1',
]

# Priority targets (run iAAFT on these; others get gap CV only)
C2_KEY_TARGETS = ['gamma_amp', 'theta_phase', 'V_CA1', 'Ca_i_CA1',
                   'I_h_CA1', 'g_GABA_B', 'I_M_CA1', 'V_basket']


def load_c2_biology(target_name, neuron_idx=0):
    """Load Circuit 2 biology and downsample to match hidden state timesteps."""
    bio_path = C2_BIOLOGY_DIR / f"{target_name}.npy"
    if not bio_path.exists():
        return None
    arr = np.load(bio_path)
    if arr.ndim == 3:
        arr = arr[:, :, neuron_idx]
    elif arr.ndim == 1:
        if len(arr) == C2_N_WINDOWS * C2_STEPS_PER_WINDOW_BIO:
            arr = arr.reshape(C2_N_WINDOWS, C2_STEPS_PER_WINDOW_BIO)
        else:
            return None
    n_win, n_steps = arr.shape
    n_out = n_steps // C2_DOWNSAMPLE_RATIO
    arr_ds = arr[:, :n_out * C2_DOWNSAMPLE_RATIO].reshape(n_win, n_out, C2_DOWNSAMPLE_RATIO).mean(axis=2)
    return arr_ds.reshape(-1).astype(np.float32)


def run_phase1_circuit2():
    """Phase 1 for Circuit 2. Uses h128 (256-dim hidden states)."""
    print("\n" + "=" * 70)
    print("PHASE 1 RETROACTIVE: Circuit 2 (CA3->CA1) -- gamma_amp cornerstone")
    print("=" * 70)
    t0 = time.time()

    # Load hidden states
    h_trained = np.load(MIMO_DIR / 'sweep_h128' / 'trained_hidden.npy')
    h_untrained = np.load(MIMO_DIR / 'sweep_h128' / 'untrained_hidden.npy')
    print(f"  Hidden states: trained={h_trained.shape}, untrained={h_untrained.shape}")

    n_bins = h_trained.shape[0]  # 21600
    n_windows = C2_N_WINDOWS  # 108
    bins_per_window = C2_STEPS_PER_WINDOW_HIDDEN  # 200

    results = {
        'circuit': 'circuit2_ca3_ca1',
        'hidden_dim': h_trained.shape[1],
        'n_bins': n_bins,
        'n_windows': n_windows,
        'bins_per_window': bins_per_window,
        'matched_baselines': {},
        'autocorrelation_diagnostic': {},
        'iaaft_significance': {},
        'summary': {},
    }

    # Load all targets
    targets = {}
    for tname in C2_ALL_TARGETS:
        bio = load_c2_biology(tname)
        if bio is not None and len(bio) == n_bins and np.std(bio) > 1e-8:
            targets[tname] = bio
    print(f"  Loaded {len(targets)}/{len(C2_ALL_TARGETS)} targets")

    # 1.1 Matched Baselines (Circuit 2 has 1 trained + 1 untrained -- no seed matching needed)
    print("\n" + "-" * 60)
    print("1.1 MATCHED BASELINES")
    print("-" * 60)

    for tname, target in targets.items():
        h_tr_w, t_w = to_window_level(h_trained, target, n_windows, bins_per_window)
        h_un_w, _ = to_window_level(h_untrained, target, n_windows, bins_per_window)

        t = t_w.astype(np.float64)
        t_std = np.std(t)
        if t_std < 1e-8:
            continue
        t = (t - np.mean(t)) / t_std
        t = np.clip(t, -3.0, 3.0)

        r2_tr, _ = shuffled_cv_r2(h_tr_w, t)
        r2_un, _ = shuffled_cv_r2(h_un_w, t)
        delta = r2_tr - r2_un

        results['matched_baselines'][tname] = {
            'r2_trained': float(r2_tr),
            'r2_untrained': float(r2_un),
            'delta_r2': float(delta),
        }

        marker = " ***" if tname in ['gamma_amp', 'theta_phase'] else ""
        print(f"  {tname:20s}: dR2={delta:+.4f} (trained={r2_tr:.4f}, untrained={r2_un:.4f}){marker}")

    # 1.2 Gap CV Diagnostic
    print("\n" + "-" * 60)
    print("1.2 GAP CV DIAGNOSTIC")
    print("-" * 60)

    for tname, target in targets.items():
        h_tr_w, t_w = to_window_level(h_trained, target, n_windows, bins_per_window)
        gap_res = run_gap_cv_diagnostic(h_tr_w, t_w, gap_windows=2)
        results['autocorrelation_diagnostic'][tname] = gap_res

        flag = " *** INFLATED" if gap_res['inflation'] > 0.1 else ""
        print(f"  {tname:20s}: shuffled={gap_res['r2_shuffled']:+.4f}, "
              f"temporal={gap_res['r2_temporal']:+.4f}, "
              f"inflation={gap_res['inflation']:+.4f}{flag}")

    # 1.3 iAAFT Significance -- KEY TARGETS (all 25 would take too long)
    print("\n" + "-" * 60)
    print(f"1.3 iAAFT SIGNIFICANCE ({N_SURROGATES} surrogates)")
    print("-" * 60)

    for tname in C2_KEY_TARGETS:
        if tname not in targets:
            continue
        target = targets[tname]
        print(f"\n  iAAFT for {tname}...")
        h_tr_w, t_w = to_window_level(h_trained, target, n_windows, bins_per_window)
        h_un_w, _ = to_window_level(h_untrained, target, n_windows, bins_per_window)

        res = run_iaaft_significance(h_tr_w, h_un_w, t_w, n_surrogates=N_SURROGATES)
        results['iaaft_significance'][tname] = res

        sig = "SIGNIFICANT" if res['significant_005'] else "NOT significant"
        print(f"    dR2={res['observed_delta_r2']:+.4f}, p={res['p_value']:.4f} ({sig}), z={res['z_score']:+.2f}")

    # Summary
    elapsed = time.time() - t0
    results['elapsed_seconds'] = elapsed

    print("\n" + "=" * 70)
    print(f"CIRCUIT 2 PHASE 1 COMPLETE ({elapsed:.0f}s)")
    print("=" * 70)

    print("\n+----------------------+---------+---------+---------+-----------+")
    print("| Target               | dR2     | Inflat. | p(iAAFT)| Survives? |")
    print("+----------------------+---------+---------+---------+-----------+")

    for tname in C2_KEY_TARGETS:
        if tname not in targets:
            continue
        dr2 = results['matched_baselines'].get(tname, {}).get('delta_r2', 0)
        infl = results['autocorrelation_diagnostic'].get(tname, {}).get('inflation', 0)
        p_val = results['iaaft_significance'].get(tname, {}).get('p_value', 1.0)
        survives = dr2 > 0.05 and p_val < 0.05
        verdict = "YES" if survives else "NO"
        results['summary'][tname] = {
            'delta_r2': float(dr2),
            'inflation': float(infl),
            'p_value': float(p_val),
            'survives_phase1': survives,
        }
        print(f"| {tname:20s} | {dr2:+.4f} | {infl:+.4f} | {p_val:.4f} | {verdict:9s} |")

    print("+----------------------+---------+---------+---------+-----------+")

    # Save
    out_path = OUTPUT_DIR / 'phase1_circuit2.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")

    return results


# ================================================================
# CIRCUIT 3: ALM->Thalamus (Mouse WM)
# ================================================================

C3_HIDDEN_DIR = WM_DIR / "hidden_states"
C3_SESSIONS = [
    "sub-440956_ses-20190210T155629_behavior+ecephys+ogen",
    "sub-440959_ses-20190223T173853_behavior+ecephys+image+ogen",
]

# DANDI streaming config for Circuit 3
C3_DANDISET_ID = '000363'
C3_THALAMUS_SESSIONS = [
    'sub-440959/sub-440959_ses-20190223T173853_behavior+ecephys+image+ogen.nwb',
]


def compute_c3_proxy_targets(h_trained):
    """Compute proxy targets from hidden states (real targets unavailable)."""
    from sklearn.decomposition import PCA
    targets = {}
    pca = PCA(n_components=min(10, h_trained.shape[1], h_trained.shape[0] - 1))
    pcs = pca.fit_transform(h_trained)
    targets['PC1'] = pcs[:, 0]
    targets['PC2'] = pcs[:, 1]
    targets['trial_variance'] = np.var(h_trained, axis=1)
    targets['trial_norm'] = np.linalg.norm(h_trained, axis=1)
    return targets


def stream_and_preprocess_c3_session(session_name):
    """Stream NWB from DANDI 000363, extract spikes, bin, compute bio targets.

    Returns (Y_test, trial_types_test, n_trials) or None on failure.
    Y_test: (n_test_trials, n_bins, n_thal_neurons)
    """
    try:
        import h5py
        import remfile
        from dandi.dandiapi import DandiAPIClient
    except ImportError as e:
        print(f"    Cannot stream: {e}")
        return None

    # Check for cached streamed data first
    cache_dir = WM_DIR / 'data' / 'raw' / session_name
    if (cache_dir / 'spike_data.npz').exists():
        print(f"    Using cached streamed data: {cache_dir}")
        sys.path.insert(0, str(WM_DIR))
        from wm.data.preprocessing import extract_from_streamed, filter_correct_trials, split_data as wm_split
        X, Y, trial_types, trial_outcomes, info = extract_from_streamed(cache_dir)
        X_f, Y_f, tt_f = filter_correct_trials(X, Y, trial_types, trial_outcomes)
        splits = wm_split(X_f, Y_f, tt_f, seed=42)
        # Use ALL data for probing (maximize trial count)
        Y_all = np.concatenate([splits[s]['Y'] for s in ['train', 'val', 'test']], axis=0)
        tt_all = np.concatenate([splits[s]['trial_types'] for s in ['train', 'val', 'test']], axis=0)
        return Y_all, tt_all, Y_all.shape[0]

    # Find the matching DANDI asset
    asset_path = None
    for known in C3_THALAMUS_SESSIONS:
        if session_name in known:
            asset_path = known
            break

    if asset_path is None:
        print(f"    No known DANDI asset for {session_name}")
        return None

    print(f"    Streaming from DANDI {C3_DANDISET_ID}: {asset_path}...")
    try:
        client = DandiAPIClient()
        ds = client.get_dandiset(C3_DANDISET_ID)

        target_asset = None
        for asset in ds.get_assets():
            if asset.path == asset_path:
                target_asset = asset
                break

        if target_asset is None:
            print(f"    Asset not found in DANDI")
            return None

        url = target_asset.get_content_url(follow_redirects=1, strip_query=True)

        # Stream-extract to cache
        sys.path.insert(0, str(WM_DIR))
        from wm.data.nwb_loader import _classify_region
        from wm.config import BIN_SIZE_S, DELAY_START_S, N_DELAY_BINS

        rfile = remfile.File(url)
        with h5py.File(rfile, 'r') as f:
            units = f['units']
            anno_names = units['anno_name'][:]
            anno_str = [v.decode() if isinstance(v, bytes) else str(v) for v in anno_names]
            classes = [_classify_region(a) for a in anno_str]
            n_units = len(anno_str)

            spike_times_data = units['spike_times'][:]
            spike_times_index = units['spike_times_index'][:]

            all_spike_times = []
            prev_idx = 0
            for i in range(n_units):
                end_idx = spike_times_index[i]
                all_spike_times.append(spike_times_data[prev_idx:end_idx].astype(np.float64))
                prev_idx = end_idx

            trials = f['intervals']['trials']
            trial_starts = trials['start_time'][:]
            n_trials_raw = len(trial_starts)

            trial_instructions = []
            if 'trial_instruction' in trials:
                raw = trials['trial_instruction'][:]
                trial_instructions = [v.decode() if isinstance(v, bytes) else str(v) for v in raw]
            else:
                trial_instructions = ['unknown'] * n_trials_raw

            outcomes = []
            if 'outcome' in trials:
                raw = trials['outcome'][:]
                outcomes = [v.decode() if isinstance(v, bytes) else str(v) for v in raw]
            else:
                outcomes = ['unknown'] * n_trials_raw

        # Save cache
        cache_dir.mkdir(parents=True, exist_ok=True)
        spike_dict = {f'unit_{i}': s for i, s in enumerate(all_spike_times)}
        np.savez_compressed(cache_dir / 'spike_data.npz', **spike_dict)

        import json as json_mod
        with open(cache_dir / 'unit_metadata.json', 'w') as fp:
            json_mod.dump({
                'n_units': n_units, 'anno_names': anno_str, 'region_classes': classes,
                'n_alm': sum(1 for c in classes if c == 'alm'),
                'n_thal': sum(1 for c in classes if c == 'thal'),
            }, fp, indent=2)
        with open(cache_dir / 'trial_metadata.json', 'w') as fp:
            json_mod.dump({
                'n_trials': n_trials_raw,
                'start_times': trial_starts.tolist(),
                'stop_times': [0.0] * n_trials_raw,
                'trial_instructions': trial_instructions,
                'outcomes': outcomes,
            }, fp, indent=2)

        print(f"    Cached to {cache_dir}")

        # Now preprocess
        from wm.data.preprocessing import extract_from_streamed, filter_correct_trials, split_data as wm_split
        X, Y, trial_types, trial_outcomes, info = extract_from_streamed(cache_dir)
        X_f, Y_f, tt_f = filter_correct_trials(X, Y, trial_types, trial_outcomes)
        splits = wm_split(X_f, Y_f, tt_f, seed=42)
        Y_all = np.concatenate([splits[s]['Y'] for s in ['train', 'val', 'test']], axis=0)
        tt_all = np.concatenate([splits[s]['trial_types'] for s in ['train', 'val', 'test']], axis=0)
        print(f"    Preprocessed: Y={Y_all.shape}, trials={Y_all.shape[0]}")
        return Y_all, tt_all, Y_all.shape[0]

    except Exception as e:
        print(f"    Streaming failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def compute_c3_bio_targets(Y, trial_types):
    """Compute real biological probe targets for Circuit 3.

    Y: (n_trials, n_timesteps, n_thal_neurons)
    trial_types: (n_trials,) binary labels

    Returns dict of target_name -> (n_trials,) scalar arrays.
    """
    sys.path.insert(0, str(WM_DIR))
    from wm.targets.emergent import compute_theta_modulation, compute_population_synchrony, compute_population_rate
    from wm.targets.choice_signal import compute_choice_axis, compute_choice_magnitude, compute_delay_stability, trial_average_choice_signal
    from wm.targets.ramp_signal import compute_ramp_signal, trial_average_ramp_signal

    targets = {}

    # Population rate (trial-averaged scalar)
    pop_rate = compute_population_rate(Y)  # (n_trials, n_timesteps)
    targets['population_rate'] = pop_rate.mean(axis=1)

    # Theta modulation
    targets['theta_modulation'] = compute_theta_modulation(Y)

    # Population synchrony
    targets['population_synchrony'] = compute_population_synchrony(Y)

    # Choice signal (requires 2 trial types)
    try:
        choice_sig, _ = compute_choice_axis(Y, trial_types)
        targets['choice_signal'] = trial_average_choice_signal(choice_sig)
        targets['choice_magnitude'] = compute_choice_magnitude(choice_sig).mean(axis=1)
        targets['delay_stability'] = compute_delay_stability(choice_sig)
    except ValueError as e:
        print(f"    choice_signal skipped: {e}")

    # Ramp signal
    try:
        ramp_sig, _ = compute_ramp_signal(Y)
        targets['ramp_signal'] = trial_average_ramp_signal(ramp_sig)
    except Exception as e:
        print(f"    ramp_signal skipped: {e}")

    return targets


def run_phase1_circuit3():
    """Phase 1 for Circuit 3. Trial-level only (79/51 trials).

    Attempts to stream real bio targets from DANDI; falls back to proxy targets.
    """
    print("\n" + "=" * 70)
    print("PHASE 1 RETROACTIVE: Circuit 3 (ALM->Thalamus, Mouse WM)")
    print("=" * 70)
    t0 = time.time()

    all_results = {}

    for session in C3_SESSIONS:
        session_dir = C3_HIDDEN_DIR / session
        if not session_dir.exists():
            print(f"  SKIP {session}: directory not found")
            continue

        short_name = session.split('_')[0]
        print(f"\n--- Session: {short_name} ---")

        # Load hidden states
        trained_path = session_dir / 'wm_h128_trained.npz'
        untrained_path = session_dir / 'wm_h128_untrained.npz'
        if not trained_path.exists() or not untrained_path.exists():
            print(f"  SKIP: missing hidden state files")
            continue

        h_trained = np.load(trained_path)['hidden_states']
        h_untrained = np.load(untrained_path)['hidden_states']
        n_trials = h_trained.shape[0]
        print(f"  Hidden states: trained={h_trained.shape}, untrained={h_untrained.shape}")

        if n_trials < 20:
            print(f"  SKIP: too few trials ({n_trials})")
            continue

        # Try to get real biological targets
        bio_data = stream_and_preprocess_c3_session(session)
        if bio_data is not None:
            Y_thal, trial_types, n_bio_trials = bio_data
            print(f"  Bio data available: Y={Y_thal.shape}, {n_bio_trials} trials")

            # n_bio_trials may differ from n_trials (hidden state trials)
            # Use the minimum and align
            n_use = min(n_trials, n_bio_trials)
            h_tr_use = h_trained[:n_use]
            h_un_use = h_untrained[:n_use]
            Y_use = Y_thal[:n_use]
            tt_use = trial_types[:n_use]

            targets = compute_c3_bio_targets(Y_use, tt_use)
            target_note = f'real biological targets from DANDI {C3_DANDISET_ID}'
        else:
            print(f"  Falling back to proxy targets")
            targets = compute_c3_proxy_targets(h_trained)
            h_tr_use = h_trained
            h_un_use = h_untrained
            n_use = n_trials
            target_note = 'proxy targets only (real bio targets unavailable)'

        session_results = {
            'session': session,
            'n_trials': n_use,
            'hidden_dim': h_trained.shape[1],
            'matched_baselines': {},
            'autocorrelation_diagnostic': {},
            'iaaft_significance': {},
            'summary': {},
            'note': target_note,
        }

        # 1.1 Matched baselines
        print("\n  1.1 Matched Baselines:")
        for tname, target in targets.items():
            t = target.astype(np.float64)
            t_std = np.std(t)
            if t_std < 1e-8:
                print(f"    {tname:20s}: SKIP (zero variance)")
                continue
            t = (t - np.mean(t)) / t_std
            t = np.clip(t, -3.0, 3.0)

            r2_tr, _ = shuffled_cv_r2(h_tr_use, t)
            r2_un, _ = shuffled_cv_r2(h_un_use, t)
            delta = r2_tr - r2_un

            session_results['matched_baselines'][tname] = {
                'r2_trained': float(r2_tr),
                'r2_untrained': float(r2_un),
                'delta_r2': float(delta),
            }
            print(f"    {tname:20s}: dR2={delta:+.4f}")

        # 1.2 Gap CV
        print("\n  1.2 Gap CV (trial-level, limited value):")
        for tname, target in targets.items():
            if np.std(target) < 1e-8:
                continue
            gap_res = run_gap_cv_diagnostic(h_tr_use, target, gap_windows=1)
            session_results['autocorrelation_diagnostic'][tname] = gap_res
            print(f"    {tname:20s}: inflation={gap_res['inflation']:+.4f}")

        # 1.3 iAAFT
        n_surr = min(N_SURROGATES, 200)  # Cap at 200 for small N trial-level data
        print(f"\n  1.3 iAAFT Significance ({n_surr} surrogates):")
        for tname, target in targets.items():
            t = target.astype(np.float64)
            t_std = np.std(t)
            if t_std < 1e-8:
                continue
            t = (t - np.mean(t)) / t_std
            t = np.clip(t, -3.0, 3.0)

            res = run_iaaft_significance(h_tr_use, h_un_use, t, n_surrogates=n_surr)
            session_results['iaaft_significance'][tname] = res

            sig = "SIGNIFICANT" if res['significant_005'] else "NOT significant"
            print(f"    {tname:20s}: dR2={res['observed_delta_r2']:+.4f}, "
                  f"p={res['p_value']:.4f} ({sig})")

        # Summary
        for tname in targets:
            dr2 = session_results['matched_baselines'].get(tname, {}).get('delta_r2', 0)
            p_val = session_results['iaaft_significance'].get(tname, {}).get('p_value', 1.0)
            session_results['summary'][tname] = {
                'delta_r2': float(dr2),
                'p_value': float(p_val),
                'survives_phase1': dr2 > 0.05 and p_val < 0.05,
            }

        all_results[short_name] = session_results

    elapsed = time.time() - t0
    all_results['elapsed_seconds'] = elapsed

    print(f"\n  Circuit 3 Phase 1 complete ({elapsed:.0f}s)")

    out_path = OUTPUT_DIR / 'phase1_circuit3.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Saved to {out_path}")

    return all_results


# ================================================================
# CIRCUIT 4: Human WM (Rutishauser)
# ================================================================

CHECKPOINT_DIR = WM_DIR / "data" / "results" / "cross_patient" / "cross_patient"
RAW_NWB_BASE = WM_DIR / "data" / "raw"


def find_nwb_for_patient(patient_id, raw_dir):
    """Recursively search raw_dir for NWB file matching patient_id.

    Priority:
    1. Exact match: patient_id.nwb (correct session)
    2. Sub-ID + session match in 000469 subdirectory structure
    3. NEVER fall back to wrong session -- returns None instead
    """
    # Try exact match first: patient_id.nwb
    nwb_name = f"{patient_id}.nwb"
    for nwb in raw_dir.rglob(nwb_name):
        return nwb
    # Try matching by sub-ID within the correct session subdirectory
    # e.g., sub-10_ses-2_ecephys+image -> look for sub-10/sub-10_ses-2*.nwb
    parts = patient_id.split('_')
    sub_id = parts[0]  # e.g., "sub-10"
    ses_id = None
    for p in parts:
        if p.startswith('ses-'):
            ses_id = p
            break
    if ses_id:
        pattern = f"{sub_id}_{ses_id}*.nwb"
        for nwb in raw_dir.rglob(pattern):
            return nwb
    # No fallback to wrong session
    return None


def run_phase1_circuit4():
    """Phase 1 for Circuit 4. Loads NWB data + checkpoints per patient."""
    print("\n" + "=" * 70)
    print("PHASE 1 RETROACTIVE: Circuit 4 (Human WM, Rutishauser)")
    print("=" * 70)
    t0 = time.time()

    # Import WM tools
    sys.path.insert(0, str(WM_DIR))
    try:
        from human_wm.surrogate.models import HumanLSTMSurrogate
        from human_wm.data.nwb_loader import extract_patient_data, split_data
    except ImportError as e:
        print(f"  ERROR: Cannot import WM tools: {e}")
        print("  Skipping Circuit 4.")
        return {'error': str(e)}

    # Load NWB schema
    schema_path = WM_DIR / 'data' / 'nwb_schema.json'
    if not schema_path.exists():
        print("  ERROR: nwb_schema.json not found")
        return {'error': 'schema not found'}
    with open(schema_path) as f:
        schema = json.load(f)

    import torch

    # Find all patient directories with checkpoints
    patient_dirs = sorted([d for d in CHECKPOINT_DIR.iterdir() if d.is_dir()])
    print(f"  Found {len(patient_dirs)} patient directories")

    all_results = {}

    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        ckpt_path = patient_dir / 'lstm_h128_s0_best.pt'

        if not ckpt_path.exists():
            continue

        # Check CC quality
        result_path = patient_dir / 'results_lstm_h128_s0.json'
        cc = None
        if result_path.exists():
            with open(result_path) as f:
                prev = json.load(f)
            cc = prev.get('cc', 0)
            if cc < 0.3:
                print(f"  SKIP {patient_id}: CC={cc:.3f} < 0.3")
                continue

        # Find matching NWB file -- search ALL raw subdirectories (000469 + 000576)
        nwb_path = find_nwb_for_patient(patient_id, RAW_NWB_BASE)
        if nwb_path is None:
            continue

        cc_str = f"{cc:.3f}" if cc is not None else "?"
        print(f"\n  {patient_id} (CC={cc_str}), NWB={nwb_path.name}:")

        try:
            X, Y, trial_info = extract_patient_data(nwb_path, schema)
            # Use ALL data for probing (not just test split) to maximize trial count
            X_all = X
            Y_all = Y
        except Exception as e:
            print(f"    Error loading data: {e}")
            continue

        n_trials, T, input_dim = X_all.shape
        output_dim = Y_all.shape[2] if Y_all.ndim == 3 else 0
        print(f"    X_all={X_all.shape}, Y_all={Y_all.shape}")

        if n_trials < 10:
            print(f"    SKIP: too few trials ({n_trials})")
            continue

        # Keep original Y for target computation before any padding
        Y_for_targets = Y_all.copy()
        X_test = X_all
        Y_test = Y_all

        # Load trained model
        state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        ckpt_input = state_dict['lstm.weight_ih_l0'].shape[1]
        ckpt_output = state_dict['output_proj.weight'].shape[0]

        # Dim alignment
        if input_dim != ckpt_input:
            if input_dim > ckpt_input:
                X_test = X_test[:, :, :ckpt_input]
            else:
                pad = np.zeros((n_trials, T, ckpt_input - input_dim))
                X_test = np.concatenate([X_test, pad], axis=2)
            input_dim = ckpt_input

        if output_dim != ckpt_output:
            if output_dim > ckpt_output:
                Y_test = Y_test[:, :, :ckpt_output]
            else:
                # Pad output to match checkpoint (probe targets come from real Y_test)
                pad = np.zeros((n_trials, T, ckpt_output - output_dim))
                Y_test = np.concatenate([Y_test, pad], axis=2)
                output_dim = ckpt_output

        model_trained = HumanLSTMSurrogate(input_dim, ckpt_output, hidden_size=128)
        model_trained.load_state_dict(state_dict)
        model_trained.train(False)

        # Extract trained hidden states
        X_t = torch.tensor(X_test, dtype=torch.float32)
        with torch.no_grad():
            _, h_seq = model_trained(X_t, return_hidden=True)
        h_np = h_seq.cpu().numpy()
        h_window_trained = h_np.mean(axis=1)  # (n_trials, 128)

        # Create untrained baseline (same architecture, random init)
        torch.manual_seed(999)
        np.random.seed(999)
        model_untrained = HumanLSTMSurrogate(input_dim, ckpt_output, hidden_size=128)
        model_untrained.train(False)
        with torch.no_grad():
            _, h_seq_un = model_untrained(X_t, return_hidden=True)
        h_window_untrained = h_seq_un.cpu().numpy().mean(axis=1)

        # Compute targets from original (unpadded) Y data
        n_real_neurons = Y_for_targets.shape[2] if Y_for_targets.ndim == 3 else 0
        if n_real_neurons < 2:
            print(f"    SKIP: too few output neurons ({n_real_neurons})")
            continue

        probe_targets = compute_human_targets_local(Y_for_targets)

        print(f"    Targets: {list(probe_targets.keys())}")

        patient_results = {
            'patient_id': patient_id,
            'cc': float(cc) if cc else None,
            'n_trials': n_trials,
            'n_time_bins': T,
            'matched_baselines': {},
            'autocorrelation_diagnostic': {},
            'iaaft_significance': {},
            'summary': {},
        }

        # 1.1 + 1.2 + 1.3 for each target
        for tname, target in probe_targets.items():
            t = target.astype(np.float64)
            t_std = np.std(t)
            if t_std < 1e-8:
                continue
            t = (t - np.mean(t)) / t_std
            t = np.clip(t, -3.0, 3.0)

            # Matched baseline
            r2_tr, _ = shuffled_cv_r2(h_window_trained, t)
            r2_un, _ = shuffled_cv_r2(h_window_untrained, t)
            delta = r2_tr - r2_un
            patient_results['matched_baselines'][tname] = {
                'r2_trained': float(r2_tr),
                'r2_untrained': float(r2_un),
                'delta_r2': float(delta),
            }

            # Gap CV (trial-level)
            gap_res = run_gap_cv_diagnostic(h_window_trained, target, gap_windows=1)
            patient_results['autocorrelation_diagnostic'][tname] = gap_res

            # iAAFT (reduced surrogates for small N)
            n_surr = min(N_SURROGATES, 100)
            iaaft_res = run_iaaft_significance(h_window_trained, h_window_untrained, target, n_surrogates=n_surr)
            patient_results['iaaft_significance'][tname] = iaaft_res

            survives = delta > 0.05 and iaaft_res['p_value'] < 0.05
            patient_results['summary'][tname] = {
                'delta_r2': float(delta),
                'inflation': float(gap_res['inflation']),
                'p_value': float(iaaft_res['p_value']),
                'survives_phase1': survives,
            }

            sig = "SIG" if iaaft_res['significant_005'] else "ns"
            print(f"    {tname:25s}: dR2={delta:+.4f}, p={iaaft_res['p_value']:.3f} ({sig})")

        all_results[patient_id] = patient_results

    elapsed = time.time() - t0
    all_results['elapsed_seconds'] = elapsed

    print(f"\n  Circuit 4 Phase 1 complete ({elapsed:.0f}s)")

    out_path = OUTPUT_DIR / 'phase1_circuit4.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Saved to {out_path}")

    return all_results


def compute_human_targets_local(Y_test):
    """Compute probe targets for human WM (local copy to avoid circular imports)."""
    targets = {}
    n_trials, T, n_neurons = Y_test.shape
    targets['mean_firing_rate'] = Y_test.mean(axis=(1, 2))
    targets['population_rate'] = Y_test.sum(axis=2).mean(axis=1)
    targets['trial_variance'] = Y_test.var(axis=1).mean(axis=1)
    temporal_var = Y_test.var(axis=1).mean(axis=1)
    targets['temporal_stability'] = 1.0 / (temporal_var + 1e-6)
    if n_neurons >= 3:
        sync_vals = []
        for i in range(n_trials):
            trial_data = Y_test[i]
            if trial_data.std() < 1e-12:
                sync_vals.append(0.0)
                continue
            corr_matrix = np.corrcoef(trial_data.T)
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            vals = corr_matrix[mask]
            vals = vals[np.isfinite(vals)]
            sync_vals.append(np.abs(vals).mean() if len(vals) > 0 else 0.0)
        targets['population_synchrony'] = np.array(sync_vals)
    return targets


# ================================================================
# MAIN
# ================================================================

def run_all():
    print("=" * 70)
    print("PHASE 1 RETROACTIVE: ALL CIRCUITS")
    print("=" * 70)
    t0 = time.time()

    # Circuit 2 -- PRIORITY (gamma_amp cornerstone)
    c2 = run_phase1_circuit2()

    # Circuit 3
    c3 = run_phase1_circuit3()

    # Circuit 4
    c4 = run_phase1_circuit4()

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"ALL CIRCUITS COMPLETE ({elapsed:.0f}s)")
    print(f"{'='*70}")

    return c2, c3, c4


if __name__ == '__main__':
    run_all()
