"""
run_all_patients.py — DESCARTES Circuit 5: 14-Patient Scaling Pipeline

Follows DESCARTES_CIRCUIT5_SCALING_GUIDE.md:
  Task A: Sanitized Ridge delta-R2, balanced accuracy for categorical, target validation
  Task B: Per-patient pipeline (Steps 1-6) + cross-patient summary

NO ablation. NO valence. NO face_presence.
"""
import numpy as np
import torch
import torch.nn as nn
import pynwb
import json
import time
import sys
import traceback
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert, butter, filtfilt
from scipy.interpolate import interp1d
from scipy.signal import detrend
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
import pandas as pd

# ================================================================
# GLOBAL CONFIG
# ================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data' / '000623'
ANNOTATIONS_DIR = BASE_DIR / 'bmovie-release-NWB-BIDS' / 'assets' / 'annotations'
PREPROCESSED_DIR = BASE_DIR / 'preprocessed_data'
PROBE_TARGETS_DIR = BASE_DIR / 'probe_targets'
RESULTS_DIR = BASE_DIR / 'results'

PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
PROBE_TARGETS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Preprocessing
BIN_MS = 20
WINDOW_MS = 2000
STRIDE_MS = 500
SIGMA_BINS = 2.0
MIN_FIRING_RATE = 0.5

INPUT_LABELS = ['Left amygdala', 'Right amygdala',
                'Left hippocampus', 'Right hippocampus']
OUTPUT_LABELS = ['Left ACC', 'Right ACC',
                 'Left preSMA', 'Right preSMA',
                 'Left vmPFC', 'Right vmPFC']

# Model
HIDDEN_SIZE = 128
N_LAYERS = 2
N_SEEDS = 10
MAX_EPOCHS = 200
BATCH_SIZE = 32
LR = 1e-3
PATIENCE = 20
DROPOUT = 0.1

# Pilot
PILOT_EPOCHS = 200
PILOT_PATIENCE = 30
CC_THRESHOLD_PRIMARY = 0.3
CC_THRESHOLD_FALLBACK = 0.2
MIN_FOCUSED_NEURONS = 3
TOP_K_FALLBACK = 5  # if <3 neurons pass threshold, take top-K anyway (LOW_QUALITY)

# Probing
SKIP_TARGETS = {'valence', 'face_presence'}
CATEGORICAL_TARGETS = {'emotion_category'}
N_PROBE_FOLDS = 5

# Untrained seeds
UNTRAINED_SEED_OFFSET = 100

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ================================================================
# MODEL
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
    """Switch model to inference mode (no dropout, no grad tracking)."""
    model.train(False)


# ================================================================
# STEP 1: PREPROCESSING
# ================================================================
def get_neuron_region(nwb, unit_idx):
    electrode_row = nwb.units['electrodes'][unit_idx]
    if hasattr(electrode_row, 'iloc') and 'group_name' in electrode_row.columns:
        gname = electrode_row['group_name'].iloc[0]
        if gname in nwb.electrode_groups:
            return nwb.electrode_groups[gname].location
    if hasattr(electrode_row, 'iloc') and 'location' in electrode_row.columns:
        return electrode_row['location'].iloc[0]
    return 'unknown'


def preprocess_patient(nwb_path, patient_id):
    """Full preprocessing: spike extraction, binning, smoothing, windows, split."""
    out_file = PREPROCESSED_DIR / f"{patient_id}.npz"
    if out_file.exists():
        print(f"  [preprocess] Using cached {out_file.name}")
        return out_file

    with pynwb.NWBHDF5IO(str(nwb_path), 'r') as io:
        nwb = io.read()
        movie_start = nwb.trials['start_time'][0]
        movie_stop = nwb.trials['stop_time'][0]
        movie_dur = movie_stop - movie_start

        n_units = len(nwb.units)
        input_ids, output_ids = [], []
        input_spikes, output_spikes = [], []
        input_regions, output_regions = [], []

        for i in range(n_units):
            region = get_neuron_region(nwb, i)
            spikes = nwb.units['spike_times'][i]
            movie_spk = spikes[(spikes >= movie_start) & (spikes <= movie_stop)] - movie_start
            if len(movie_spk) / movie_dur < MIN_FIRING_RATE:
                continue
            if region in INPUT_LABELS:
                input_ids.append(i)
                input_spikes.append(movie_spk)
                input_regions.append(region)
            elif region in OUTPUT_LABELS:
                output_ids.append(i)
                output_spikes.append(movie_spk)
                output_regions.append(region)

    print(f"  [preprocess] {len(input_ids)} input, {len(output_ids)} output neurons")

    dur_ms = movie_dur * 1000.0
    n_bins = int(np.ceil(dur_ms / BIN_MS))
    bin_edges = np.arange(0, dur_ms + BIN_MS, BIN_MS) / 1000.0

    def bin_spikes(spike_list):
        arr = np.zeros((len(spike_list), n_bins), dtype=np.float32)
        for i, st in enumerate(spike_list):
            if len(st) > 0:
                counts, _ = np.histogram(st, bins=bin_edges)
                arr[i, :len(counts)] = counts
        return arr

    input_binned = bin_spikes(input_spikes)
    output_binned = bin_spikes(output_spikes)

    input_rates = gaussian_filter1d(input_binned.astype(np.float32), sigma=SIGMA_BINS, axis=1)
    output_rates = gaussian_filter1d(output_binned.astype(np.float32), sigma=SIGMA_BINS, axis=1)

    def zscore_neurons(rates):
        mu = rates.mean(axis=1, keepdims=True)
        std = rates.std(axis=1, keepdims=True)
        std[std < 1e-6] = 1.0
        return ((rates - mu) / std).astype(np.float32)

    input_rates = zscore_neurons(input_rates)
    output_rates = zscore_neurons(output_rates)

    window_bins = WINDOW_MS // BIN_MS
    stride_bins = STRIDE_MS // BIN_MS
    T = input_rates.shape[1]
    n_windows = (T - window_bins) // stride_bins + 1

    X = np.zeros((n_windows, window_bins, len(input_ids)), dtype=np.float32)
    Y = np.zeros((n_windows, window_bins, len(output_ids)), dtype=np.float32)
    for w in range(n_windows):
        s = w * stride_bins
        X[w] = input_rates[:, s:s+window_bins].T
        Y[w] = output_rates[:, s:s+window_bins].T

    train_end = int(n_windows * 0.60)
    val_end = int(n_windows * 0.80)

    np.savez_compressed(
        out_file,
        X_train=X[:train_end], Y_train=Y[:train_end],
        X_val=X[train_end:val_end], Y_val=Y[train_end:val_end],
        X_test=X[val_end:], Y_test=Y[val_end:],
        input_rates=input_rates, output_rates=output_rates,
        input_ids=np.array(input_ids), output_ids=np.array(output_ids),
        input_regions=np.array(input_regions),
        output_regions=np.array(output_regions),
        movie_start=movie_start, movie_stop=movie_stop,
        bin_ms=BIN_MS, window_ms=WINDOW_MS, stride_ms=STRIDE_MS,
        n_input=len(input_ids), n_output=len(output_ids))
    print(f"  [preprocess] Saved {out_file.name} "
          f"({X[:train_end].shape[0]}/{X[train_end:val_end].shape[0]}/{X[val_end:].shape[0]} windows)")
    return out_file


# ================================================================
# STEP 2: PILOT LSTM -> FOCUSED NEURON SELECTION
# ================================================================
def run_pilot(X_train, Y_train, X_val, Y_val, n_input, n_output, patient_id):
    """Quick pilot LSTM to identify focused output neurons."""
    result_file = RESULTS_DIR / patient_id / 'focused_neurons.json'
    if result_file.exists():
        with open(result_file) as f:
            info = json.load(f)
        print(f"  [pilot] Using cached: {len(info['focused_indices'])} neurons")
        return info

    torch.manual_seed(42)
    model = LimbicPrefrontalLSTM(n_input, n_output, HIDDEN_SIZE, N_LAYERS, DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    X_tr = torch.FloatTensor(X_train).to(DEVICE)
    Y_tr = torch.FloatTensor(Y_train).to(DEVICE)
    X_v = torch.FloatTensor(X_val).to(DEVICE)
    Y_v = torch.FloatTensor(Y_val).to(DEVICE)

    best_val = float('inf')
    best_state = None
    patience_ctr = 0
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=PILOT_PATIENCE//2, factor=0.5)

    for ep in range(PILOT_EPOCHS):
        model.train()
        perm = torch.randperm(len(X_tr))
        n_b = max(1, len(X_tr) // BATCH_SIZE)
        for b in range(n_b):
            idx = perm[b*BATCH_SIZE:(b+1)*BATCH_SIZE]
            optimizer.zero_grad()
            loss = criterion(model(X_tr[idx]), Y_tr[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        set_inference_mode(model)
        with torch.no_grad():
            vl = criterion(model(X_v), Y_v).item()
        scheduler.step(vl)
        if vl < best_val:
            best_val = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PILOT_PATIENCE:
                break

    model.load_state_dict(best_state)
    set_inference_mode(model)
    with torch.no_grad():
        pred = model(X_v).cpu().numpy()

    cc_per_neuron = []
    for j in range(n_output):
        p = pred[:, :, j].flatten()
        t = Y_val[:, :, j].flatten()
        if np.std(p) > 0 and np.std(t) > 0:
            cc_per_neuron.append(float(np.corrcoef(p, t)[0, 1]))
        else:
            cc_per_neuron.append(0.0)

    cc_arr = np.array(cc_per_neuron)
    threshold_used = CC_THRESHOLD_PRIMARY
    mask = cc_arr >= CC_THRESHOLD_PRIMARY
    if mask.sum() < 5:
        threshold_used = CC_THRESHOLD_FALLBACK
        mask = cc_arr >= CC_THRESHOLD_FALLBACK
    used_topk_fallback = False
    if mask.sum() < MIN_FOCUSED_NEURONS:
        # Top-K fallback: take best neurons regardless of absolute CC
        used_topk_fallback = True
        k = min(TOP_K_FALLBACK, n_output)
        top_k_idx = np.argsort(cc_arr)[::-1][:k]
        focused_idx = top_k_idx.tolist()
        focused_ccs = cc_arr[top_k_idx].tolist()
        threshold_used = 0.0
        print(f"  [pilot] Top-K fallback: taking best {k} neurons "
              f"(max CC={focused_ccs[0]:.3f}, min CC={focused_ccs[-1]:.3f})")
    else:
        focused_idx = np.where(mask)[0].tolist()
        focused_ccs = cc_arr[focused_idx].tolist()
        order = np.argsort(focused_ccs)[::-1]
        focused_idx = [focused_idx[i] for i in order]
        focused_ccs = [focused_ccs[i] for i in order]

    info = {
        'patient_id': patient_id,
        'n_output_total': n_output,
        'n_focused': len(focused_idx),
        'focused_indices': focused_idx,
        'focused_ccs': focused_ccs,
        'all_ccs': cc_per_neuron,
        'threshold_used': float(threshold_used),
        'used_topk_fallback': used_topk_fallback,
    }

    result_file.parent.mkdir(parents=True, exist_ok=True)
    with open(result_file, 'w') as f:
        json.dump(info, f, indent=2)
    if focused_idx:
        print(f"  [pilot] {len(focused_idx)} focused neurons (best CC={focused_ccs[0]:.3f})")
    else:
        print(f"  [pilot] 0 focused neurons")
    return info


# ================================================================
# STEP 3: PROBE TARGETS
# ================================================================
def compute_probe_targets(nwb_path, preprocessed_path, patient_id):
    """Compute all probe targets (excluding SKIP_TARGETS)."""
    out_file = PROBE_TARGETS_DIR / f"{patient_id}.npz"
    if out_file.exists():
        print(f"  [targets] Using cached {out_file.name}")
        return out_file

    data = np.load(preprocessed_path, allow_pickle=True)
    input_rates = data['input_rates']
    movie_start = float(data['movie_start'])
    movie_stop = float(data['movie_stop'])
    n_bins = input_rates.shape[1]

    targets = {}

    # Scene cuts
    scene_csv = ANNOTATIONS_DIR / 'scenecut_info.csv'
    if scene_csv.exists():
        scene_df = pd.read_csv(scene_csv)
        scene_cut_times = scene_df['shot_start_t'].values.astype(float)
    else:
        scene_cut_times = np.array([])
        print("  WARNING: scenecut_info.csv not found")

    # Arousal: 0.7 x pupil + 0.3 x cuts (per guide)
    arousal_cuts = _arousal_from_cuts(scene_cut_times, n_bins)

    with pynwb.NWBHDF5IO(str(nwb_path), 'r') as io:
        nwb = io.read()
        pupil_signal = _arousal_from_pupil(nwb, movie_start, movie_stop, n_bins)

    if pupil_signal is not None:
        def norm01(x):
            mn, mx = x.min(), x.max()
            return (x - mn) / max(mx - mn, 1e-8)
        arousal = 0.7 * norm01(pupil_signal) + 0.3 * norm01(arousal_cuts)
    else:
        print("  [targets] Pupil unavailable, using cuts only for arousal")
        arousal = arousal_cuts

    targets['arousal'] = arousal.astype(np.float32)
    targets['threat_pe'] = _threat_pe(arousal)

    valence_internal = -arousal  # only for category derivation
    targets['emotion_category'] = _emotion_category(arousal, valence_internal)

    targets['event_boundaries'] = _event_boundaries(scene_cut_times, n_bins)
    targets['population_synchrony'] = _population_synchrony(input_rates)
    targets['temporal_stability'] = _temporal_stability(input_rates)

    with pynwb.NWBHDF5IO(str(nwb_path), 'r') as io:
        nwb = io.read()
        theta = _extract_lfp_band(nwb, movie_start, movie_stop, n_bins, 4.0, 8.0, "Theta")
        gamma = _extract_lfp_band(nwb, movie_start, movie_stop, n_bins, 30.0, 80.0, "Gamma")

    if theta is not None:
        targets['theta_power'] = theta
    else:
        targets['theta_power'] = gaussian_filter1d(
            targets['population_synchrony'], sigma=50).astype(np.float32)

    if gamma is not None:
        targets['gamma_power'] = gamma
    else:
        targets['gamma_power'] = gaussian_filter1d(
            targets['temporal_stability'], sigma=10).astype(np.float32)

    with pynwb.NWBHDF5IO(str(nwb_path), 'r') as io:
        nwb = io.read()
        enc = _encoding_success(nwb, movie_start, movie_stop, n_bins)
    if enc is not None:
        targets['encoding_success'] = enc
    else:
        targets['encoding_success'] = np.zeros(n_bins, dtype=np.float32)

    np.savez_compressed(out_file, **targets)
    print(f"  [targets] Saved {len(targets)} targets to {out_file.name}")
    return out_file


# -- Probe target helper functions --

def _arousal_from_cuts(scene_cut_times, n_bins):
    signal = np.zeros(n_bins, dtype=np.float32)
    bins = (scene_cut_times * 1000.0 / BIN_MS).astype(int)
    bins = bins[(bins >= 0) & (bins < n_bins)]
    signal[bins] = 1.0
    return gaussian_filter1d(signal, sigma=500).astype(np.float32)


def _arousal_from_pupil(nwb, movie_start, movie_stop, n_bins):
    try:
        pupil = nwb.processing['behavior']['PupilTracking']
        ts = pupil.time_series['TimeSeries']
        data = ts.data[:].astype(np.float64)
        if data.ndim == 2:
            data = data[:, 0]
        rate = float(ts.rate)
        t0 = float(ts.starting_time) if ts.starting_time else 0.0
        timestamps = t0 + np.arange(len(data)) / rate
        mask = (timestamps >= movie_start) & (timestamps <= movie_stop)
        pupil_movie = data[mask].copy()
        ts_movie = timestamps[mask] - movie_start

        if len(pupil_movie) < 100:
            return None

        blink_mask = pupil_movie < 10.0
        if blink_mask.sum() / len(pupil_movie) > 0.5:
            return None
        valid = ~blink_mask & np.isfinite(pupil_movie)
        if valid.sum() < 100:
            return None
        f_interp = interp1d(ts_movie[valid], pupil_movie[valid],
                            kind='linear', fill_value='extrapolate', bounds_error=False)
        pupil_clean = f_interp(ts_movie)
        pupil_detrended = detrend(pupil_clean, type='linear')

        std = np.std(pupil_detrended)
        if std < 1e-6:
            return None
        pupil_z = (pupil_detrended - np.mean(pupil_detrended)) / std

        bin_edges_s = np.arange(0, (movie_stop - movie_start) * 1000.0 + BIN_MS, BIN_MS) / 1000.0
        binned = np.zeros(n_bins, dtype=np.float32)
        for i in range(n_bins):
            if i + 1 < len(bin_edges_s):
                m = (ts_movie >= bin_edges_s[i]) & (ts_movie < bin_edges_s[i+1])
                if m.any():
                    binned[i] = np.mean(pupil_z[m])
                elif i > 0:
                    binned[i] = binned[i-1]
        return gaussian_filter1d(binned, sigma=100).astype(np.float32)
    except Exception as e:
        print(f"  [pupil] Failed: {e}")
        return None


def _threat_pe(arousal, sigma=25.0):
    predicted = gaussian_filter1d(arousal, sigma=sigma)
    return (arousal - predicted).astype(np.float32)


def _emotion_category(arousal, valence, threshold=0.5):
    n = len(arousal)
    cats = np.zeros(n, dtype=np.int32)
    cats[(arousal > threshold) & (valence < -threshold)] = 1
    deriv = np.gradient(arousal)
    cats[deriv > np.percentile(deriv, 90)] = 2
    relief_mask = (deriv < np.percentile(deriv, 10)) & (np.roll(arousal, 5) > threshold)
    cats[relief_mask] = 3
    return cats


def _event_boundaries(scene_cut_times, n_bins, sigma=5.0):
    signal = np.zeros(n_bins, dtype=np.float32)
    bins = (scene_cut_times * 1000.0 / BIN_MS).astype(int)
    bins = bins[(bins >= 0) & (bins < n_bins)]
    signal[bins] = 1.0
    return gaussian_filter1d(signal, sigma=sigma).astype(np.float32)


def _population_synchrony(spike_rates, window_bins=10):
    n_neurons, T = spike_rates.shape
    if n_neurons < 2:
        return np.zeros(T, dtype=np.float32)
    n_win = T - window_bins + 1
    sync = np.zeros(n_win, dtype=np.float32)
    for t in range(n_win):
        w = spike_rates[:, t:t+window_bins]
        active = np.std(w, axis=1) > 0
        if active.sum() < 2:
            continue
        corr = np.corrcoef(w[active])
        upper = corr[np.triu_indices(active.sum(), k=1)]
        sync[t] = np.nanmean(np.abs(upper))
    pad = T - n_win
    return np.concatenate([sync, np.full(pad, sync[-1] if len(sync) > 0 else 0)]).astype(np.float32)


def _temporal_stability(spike_rates, lag=5):
    n_neurons, T = spike_rates.shape
    stab = np.zeros(T, dtype=np.float32)
    for t in range(lag, T):
        c, p = spike_rates[:, t], spike_rates[:, t - lag]
        nc, np_ = np.linalg.norm(c), np.linalg.norm(p)
        if nc > 0 and np_ > 0:
            stab[t] = np.dot(c, p) / (nc * np_)
    stab[:lag] = stab[lag]
    return stab


def _get_limbic_cols(es):
    keywords = ['amygdala', 'hippocampus']
    elec_indices = es.electrodes.data[:]
    etable = es.electrodes.table.to_dataframe()
    cols, locs = [], []
    for ci, ei in enumerate(elec_indices):
        loc = str(etable.iloc[ei]['location']).lower()
        if any(kw in loc for kw in keywords):
            cols.append(ci)
            locs.append(etable.iloc[ei]['location'])
    return cols, locs


def _extract_lfp_band(nwb, movie_start, movie_stop, n_bins,
                       band_low, band_high, name):
    try:
        lfp_container = nwb.processing['ecephys']['LFP_macro']
        es = list(lfp_container.electrical_series.values())[0]
        rate = float(es.rate) if es.rate else 1000.0
        data = es.data
        cols, locs = _get_limbic_cols(es)
        if not cols:
            print(f"  [{name}] No limbic channels")
            return None
        print(f"  [{name}] {len(cols)} limbic channels")

        t0 = float(es.starting_time) if es.starting_time is not None else 0.0
        s0 = max(0, int((movie_start - t0) * rate))
        s1 = min(data.shape[0], int((movie_stop - t0) * rate))
        if s1 - s0 < 2000:
            return None

        lfp_movie = data[s0:s1, :][:, cols].astype(np.float64)
        nyq = rate / 2.0
        if band_high / nyq >= 1.0:
            return None

        b, a = butter(4, [band_low / nyq, band_high / nyq], btype='band')
        envs = np.zeros_like(lfp_movie)
        for ch in range(lfp_movie.shape[1]):
            filt = filtfilt(b, a, lfp_movie[:, ch])
            envs[:, ch] = np.abs(hilbert(filt))
        mean_env = np.mean(envs, axis=1).astype(np.float32)

        spb = int(BIN_MS / 1000.0 * rate)
        n_use = min(n_bins, len(mean_env) // spb)
        binned = np.array([np.mean(mean_env[i*spb:(i+1)*spb]) for i in range(n_use)],
                          dtype=np.float32)
        if len(binned) < n_bins:
            binned = np.pad(binned, (0, n_bins - len(binned)), mode='edge')
        else:
            binned = binned[:n_bins]

        mu, std = binned.mean(), binned.std()
        if std > 1e-8:
            binned = (binned - mu) / std
        return binned.astype(np.float32)
    except Exception as e:
        print(f"  [{name}] Failed: {e}")
        return None


def _encoding_success(nwb, movie_start, movie_stop, n_bins):
    try:
        n_trials = len(nwb.trials)
        if n_trials <= 1:
            return None
        stim_phases = [str(nwb.trials['stim_phase'][i]) for i in range(n_trials)]
        rec_trials = [i for i in range(n_trials) if stim_phases[i] != 'encoding']
        if not rec_trials:
            return None

        responses = []
        for i in range(n_trials):
            try:
                responses.append(int(nwb.trials['response_correct'][i]))
            except:
                responses.append(0)

        movie_dur = movie_stop - movie_start
        signal = np.zeros(n_bins, dtype=np.float32)
        n_rec = len(rec_trials)
        for idx, ti in enumerate(rec_trials):
            t = (idx / max(n_rec - 1, 1)) * movie_dur
            bi = min(int(t * 1000.0 / BIN_MS), n_bins - 1)
            signal[bi] = 1.0 if responses[ti] else -1.0

        signal = gaussian_filter1d(signal, sigma=250)
        mx = np.max(np.abs(signal))
        if mx > 0:
            signal /= mx
        return signal.astype(np.float32)
    except Exception as e:
        print(f"  [encoding] Failed: {e}")
        return None


# ================================================================
# SANITIZED PROBING (Task A from guide)
# ================================================================
def validate_probe_target(name, target):
    """Check for pathological target properties."""
    report = {
        'name': name, 'length': len(target),
        'has_nan': bool(np.any(np.isnan(target))),
        'has_inf': bool(np.any(np.isinf(target))),
        'std': float(np.nanstd(target)),
    }
    if report['has_nan'] or report['has_inf']:
        report['skip'] = True
        report['skip_reason'] = 'contains NaN or Inf'
        return report
    if report['std'] < 1e-6:
        report['skip'] = True
        report['skip_reason'] = 'near-constant'
        return report
    unique_vals, counts = np.unique(target, return_counts=True)
    max_frac = counts.max() / len(target)
    if max_frac > 0.95:
        report['skip'] = True
        report['skip_reason'] = f'{max_frac:.1%} identical values'
        return report
    report['skip'] = False
    report['skip_reason'] = 'OK'
    return report


def sanitized_ridge_delta_r2(h_trained, target, h_untrained,
                              n_folds=N_PROBE_FOLDS):
    """Sanitized Ridge delta-R2 with 3 guards: z-score+clip, clamp, validity."""
    target = target.copy().astype(np.float64)
    t_std = np.std(target)
    if t_std < 1e-8:
        return {'r2_trained': 0.0, 'r2_untrained': 0.0, 'delta_r2': 0.0,
                'valid': False, 'reason': 'near-constant'}

    # Guard 1: z-score + clip to plus/minus 3 sigma
    target = (target - np.mean(target)) / t_std
    n_clipped = int(np.sum(np.abs(target) > 3.0))
    target = np.clip(target, -3.0, 3.0)

    alphas = np.logspace(-3, 3, 20)
    kf = KFold(n_splits=n_folds, shuffle=False)

    def probe_r2(hidden):
        folds = []
        for tr, te in kf.split(hidden):
            r2 = RidgeCV(alphas=alphas).fit(hidden[tr], target[tr]).score(hidden[te], target[te])
            folds.append(np.clip(r2, -1.0, 1.0))  # Guard 2: clamp
        return float(np.mean(folds)), folds

    r2_tr, folds_tr = probe_r2(h_trained)
    r2_un, folds_un = probe_r2(h_untrained)
    delta = r2_tr - r2_un
    valid = r2_tr > 0.0  # Guard 3

    return {
        'r2_trained': r2_tr, 'r2_untrained': r2_un, 'delta_r2': delta,
        'valid': valid, 'n_outliers_clipped': n_clipped,
        'reason': 'OK' if valid else 'r2_trained <= 0',
    }


def classification_delta_accuracy(h_trained, target_labels, h_untrained,
                                   n_folds=N_PROBE_FOLDS):
    """Balanced accuracy for categorical targets."""
    unique, counts = np.unique(target_labels, return_counts=True)
    n_classes = len(unique)
    class_dist = {int(c): int(n) for c, n in zip(unique, counts)}
    chance = 1.0 / n_classes if n_classes >= 2 else 0.0

    if n_classes < 2:
        return {'acc_trained': 0.0, 'acc_untrained': 0.0,
                'delta_accuracy': 0.0, 'valid': False,
                'reason': f'only {n_classes} class(es)',
                'class_distribution': class_dist, 'chance_balanced': chance}

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

    return {
        'acc_trained': acc_tr, 'acc_untrained': acc_un,
        'delta_accuracy': delta, 'chance_balanced': chance,
        'valid': acc_tr > chance,
        'reason': 'OK' if acc_tr > chance else 'trained below chance',
        'class_distribution': class_dist,
    }


# ================================================================
# STEP 4: TRAIN 10 SEEDS + 10 UNTRAINED
# ================================================================
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
    final_ep = 0

    for ep in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(len(X_tr))
        nb = max(1, len(X_tr) // BATCH_SIZE)
        for b in range(nb):
            idx = perm[b*BATCH_SIZE:(b+1)*BATCH_SIZE]
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
                final_ep = ep + 1
                break
        final_ep = ep + 1

    model.load_state_dict(best_state)
    set_inference_mode(model)

    with torch.no_grad():
        pred = model(X_v).cpu().numpy()
    ccs = []
    for j in range(Y_val.shape[2]):
        p = pred[:, :, j].flatten()
        t = Y_val[:, :, j].flatten()
        ccs.append(float(np.corrcoef(p, t)[0, 1]) if np.std(p) > 0 and np.std(t) > 0 else 0.0)

    return float(np.nanmean(ccs)), ccs, best_val, final_ep


def extract_hidden_flat(model, X):
    set_inference_mode(model)
    with torch.no_grad():
        _, h = model(torch.FloatTensor(X).to(DEVICE), return_hidden=True)
    return h.cpu().numpy().reshape(-1, h.shape[-1])


# ================================================================
# STEP 5-6: PROBING + PATIENT REPORT
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


def classify_status(delta, valid, consistency, is_categorical=False):
    """Classify probe result per guide B.2."""
    if not valid:
        return 'zombie'
    if is_categorical:
        if delta > 0.10 and consistency >= 7:
            return 'STRONG'
        if delta > 0.03 and consistency >= 5:
            return 'CANDIDATE'
        return 'zombie'
    else:
        if delta > 0.15 and consistency >= 7:
            return 'STRONG'
        if delta > 0.05 and consistency >= 5:
            return 'CANDIDATE'
        return 'zombie'


def run_probing_for_patient(patient_id, X_train, Y_train, X_val, Y_val,
                             X_test, Y_test, n_input, n_focused,
                             probe_targets_path, focused_idx):
    """Steps 4-6: Train 10 seeds, probe, generate report."""
    window_bins = WINDOW_MS // BIN_MS
    stride_bins = STRIDE_MS // BIN_MS
    train_end = X_train.shape[0]
    val_end = train_end + X_val.shape[0]
    test_start_window = val_end
    n_test = X_test.shape[0]

    # Step 4: Train 10 seeds
    print(f"  [train] Training {N_SEEDS} seeds...")
    seed_ccs = []
    trained_models = []
    for s in range(N_SEEDS):
        model = LimbicPrefrontalLSTM(n_input, n_focused, HIDDEN_SIZE, N_LAYERS, DROPOUT)
        mean_cc, per_cc, val_loss, epochs = train_model(
            model, X_train, Y_train, X_val, Y_val, s)
        seed_ccs.append(mean_cc)
        trained_models.append(model)

    mean_cc = float(np.mean(seed_ccs))
    std_cc = float(np.std(seed_ccs))
    quality = 'OK' if mean_cc >= 0.3 else 'LOW_QUALITY'
    print(f"  [train] Mean CC = {mean_cc:.4f} +/- {std_cc:.4f} [{quality}]")

    # Step 4b: Create untrained models
    untrained_models = []
    for s in range(N_SEEDS):
        torch.manual_seed(UNTRAINED_SEED_OFFSET + s)
        um = LimbicPrefrontalLSTM(n_input, n_focused, HIDDEN_SIZE, N_LAYERS, DROPOUT).to(DEVICE)
        set_inference_mode(um)
        untrained_models.append(um)

    # Load probe targets
    probe_data = np.load(probe_targets_path)
    target_names = [n for n in probe_data.files if n not in SKIP_TARGETS]

    # Step 5: Probe per seed
    print(f"  [probe] Probing {len(target_names)} targets x {N_SEEDS} seeds...")
    per_seed_results = {name: [] for name in target_names}
    target_validations = {}
    excluded_targets = []

    for name in target_names:
        target_raw = probe_data[name]
        target_aligned = align_target_to_windows(
            target_raw, n_test, test_start_window, window_bins, stride_bins)

        val_report = validate_probe_target(name, target_aligned)
        target_validations[name] = val_report
        if val_report['skip']:
            print(f"    {name}: SKIPPED ({val_report['skip_reason']})")
            excluded_targets.append(name)
            continue

        is_cat = name in CATEGORICAL_TARGETS

        for s in range(N_SEEDS):
            h_tr = extract_hidden_flat(trained_models[s], X_test)
            h_un = extract_hidden_flat(untrained_models[s], X_test)

            min_len = min(len(target_aligned), len(h_tr))
            t_use = target_aligned[:min_len]
            h_tr_use = h_tr[:min_len]
            h_un_use = h_un[:min_len]

            if is_cat:
                res = classification_delta_accuracy(h_tr_use, t_use.astype(int), h_un_use)
                per_seed_results[name].append({
                    'delta': res['delta_accuracy'], 'valid': res['valid'],
                    'metric_trained': res['acc_trained'],
                    'metric_untrained': res['acc_untrained'],
                    'full': res})
            else:
                res = sanitized_ridge_delta_r2(h_tr_use, t_use, h_un_use)
                per_seed_results[name].append({
                    'delta': res['delta_r2'], 'valid': res['valid'],
                    'metric_trained': res['r2_trained'],
                    'metric_untrained': res['r2_untrained'],
                    'full': res})

    # Step 6: Aggregate + report
    probe_results = {}
    for name in target_names:
        if name in excluded_targets:
            probe_results[name] = {'status': 'INVALID', 'reason': 'failed validation'}
            continue

        seeds = per_seed_results[name]
        if not seeds:
            probe_results[name] = {'status': 'INVALID', 'reason': 'no seed results'}
            continue

        deltas = [s['delta'] for s in seeds]
        valids = [s['valid'] for s in seeds]
        is_cat = name in CATEGORICAL_TARGETS

        mean_delta = float(np.mean(deltas))
        std_delta = float(np.std(deltas))
        thresh = 0.03 if is_cat else 0.05
        consistency = sum(1 for d in deltas if d > thresh)
        sign_positive = sum(1 for d in deltas if d > 0)
        any_valid = any(valids)

        status = classify_status(abs(mean_delta), any_valid, consistency, is_cat)

        # Only report if at least 7/10 agree on sign
        if sign_positive < 7 and (N_SEEDS - sign_positive) < 7:
            status = 'zombie'

        method = 'classification' if is_cat else 'ridge'
        probe_results[name] = {
            'method': method,
            'metric_trained': float(np.mean([s['metric_trained'] for s in seeds])),
            'metric_untrained': float(np.mean([s['metric_untrained'] for s in seeds])),
            'delta': mean_delta,
            'delta_std_across_seeds': std_delta,
            'seed_consistency': f"{consistency}/{N_SEEDS}",
            'sign_positive_seeds': sign_positive,
            'valid': any_valid,
            'status': status,
        }
        print(f"    {name:<25s} delta={mean_delta:+.4f} +/- {std_delta:.4f}  "
              f"[{status}]  consistency={consistency}/{N_SEEDS}")

    report = {
        'patient_id': patient_id,
        'n_input_neurons': n_input,
        'n_focused_neurons': n_focused,
        'focused_neuron_ids': focused_idx,
        'mean_cc': mean_cc,
        'cc_std': std_cc,
        'cc_per_seed': [float(c) for c in seed_ccs],
        'quality_flag': quality,
        'probe_results': probe_results,
        'target_validation': target_validations,
        'excluded_targets': excluded_targets,
    }

    report_path = RESULTS_DIR / patient_id / 'patient_report.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  [report] Saved {report_path}")
    return report


# ================================================================
# CROSS-PATIENT SUMMARY
# ================================================================
def build_cross_patient_summary():
    patient_reports = []
    for rf in sorted(RESULTS_DIR.glob("sub-*/patient_report.json")):
        with open(rf) as f:
            patient_reports.append(json.load(f))

    n_patients = len(patient_reports)
    print(f"\n{'='*90}")
    print(f"CROSS-PATIENT PROBING SUMMARY -- CIRCUIT 5 (n={n_patients})")
    print(f"{'='*90}")

    all_targets = set()
    for r in patient_reports:
        all_targets.update(r.get('probe_results', {}).keys())

    summary = {}
    for target in sorted(all_targets):
        strong = candidate = zombie = invalid = 0
        deltas = []
        low_q = 0

        for r in patient_reports:
            if r.get('quality_flag') == 'LOW_QUALITY':
                low_q += 1
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
        frac = (strong + candidate) / max(n_valid, 1)
        summary[target] = {
            'n_patients': n_patients, 'n_valid': n_valid,
            'n_strong': strong, 'n_candidate_or_better': strong + candidate,
            'n_zombie': zombie, 'n_invalid': invalid,
            'fraction_learned': frac,
            'mean_delta': float(np.mean(deltas)) if deltas else None,
            'std_delta': float(np.std(deltas)) if deltas else None,
            'universally_learned': frac > 0.5,
        }

    print(f"{'Target':<25} {'Strong':>6} {'Cand+':>6} {'Zombie':>6} "
          f"{'Invld':>5} {'Mean D':>10} {'Learned?':>10}")
    print('-' * 80)
    for t in sorted(summary.keys(),
                    key=lambda x: summary[x].get('mean_delta') or -999,
                    reverse=True):
        s = summary[t]
        md = f"{s['mean_delta']:+.4f}" if s['mean_delta'] is not None else "N/A"
        lr = "YES" if s['universally_learned'] else "no"
        print(f"{t:<25} {s['n_strong']:>6} {s['n_candidate_or_better']:>6} "
              f"{s['n_zombie']:>6} {s['n_invalid']:>5} {md:>10} {lr:>10}")

    low_q_total = sum(1 for r in patient_reports if r.get('quality_flag') == 'LOW_QUALITY')
    print(f"\nLow quality patients (CC < 0.3): {low_q_total}/{n_patients}")

    summary_path = RESULTS_DIR / 'cross_patient_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {summary_path}")
    return summary


# ================================================================
# MAIN
# ================================================================
def find_nwb_path(patient_id):
    """Find R1 NWB file for patient."""
    patient_dir = DATA_DIR / patient_id
    cs_num = patient_id.replace('sub-CS', '')
    nwb_name = f"{patient_id}_ses-P{cs_num}CSR1_behavior+ecephys.nwb"
    nwb_path = patient_dir / nwb_name
    if nwb_path.exists():
        return nwb_path
    for f in patient_dir.glob("*R1*ecephys.nwb"):
        return f
    for f in patient_dir.glob("*.nwb"):
        return f
    return None


def main():
    t_total = time.time()

    inv_path = BASE_DIR / 'patient_inventory.json'
    with open(inv_path) as f:
        inventory = json.load(f)

    included = [p for p in inventory if p.get('included', False)]
    print(f"\n{'='*70}")
    print(f"DESCARTES CIRCUIT 5 -- 14-PATIENT SCALING PIPELINE")
    print(f"{'='*70}")
    print(f"Patients: {len(included)}")
    print(f"Device: {DEVICE}")
    print(f"Seeds: {N_SEEDS} trained + {N_SEEDS} untrained")
    print(f"Skip targets: {SKIP_TARGETS}")
    print(f"NO ablation (holographic LSTM confirmed)")
    print()

    completed = []
    failed = []
    excluded = []
    patient_times = []

    for pi, patient_info in enumerate(included):
        patient_id = patient_info['subject_id']
        t0 = time.time()
        print(f"\n{'#'*70}")
        print(f"# Patient {pi+1}/{len(included)}: {patient_id} "
              f"({patient_info['n_input']}in / {patient_info['n_output']}out)")
        print(f"{'#'*70}")

        try:
            nwb_path = find_nwb_path(patient_id)
            if nwb_path is None:
                print(f"  ERROR: NWB file not found for {patient_id}")
                failed.append((patient_id, 'NWB not found'))
                continue

            # Step 1: Preprocess
            preprocessed = preprocess_patient(nwb_path, patient_id)
            data = np.load(preprocessed)
            n_input = int(data['n_input'])
            n_output = int(data['n_output'])

            if n_input < 5 or n_output < 3:
                print(f"  EXCLUDE: too few neurons ({n_input}in/{n_output}out)")
                excluded.append((patient_id, f'{n_input}in/{n_output}out'))
                continue

            # Step 2: Pilot + focused selection
            pilot = run_pilot(data['X_train'], data['Y_train'],
                              data['X_val'], data['Y_val'],
                              n_input, n_output, patient_id)

            if pilot.get('used_topk_fallback', False):
                print(f"  WARNING: used top-K fallback ({pilot['n_focused']} neurons, "
                      f"max CC={max(pilot['focused_ccs']):.3f}) -- LOW_QUALITY")

            focused_idx = pilot['focused_indices']
            n_focused = len(focused_idx)

            Y_train_f = data['Y_train'][:, :, focused_idx]
            Y_val_f = data['Y_val'][:, :, focused_idx]
            Y_test_f = data['Y_test'][:, :, focused_idx]

            # Step 3: Probe targets
            targets_path = compute_probe_targets(nwb_path, preprocessed, patient_id)

            # Steps 4-6: Train, probe, report
            report = run_probing_for_patient(
                patient_id, data['X_train'], Y_train_f,
                data['X_val'], Y_val_f, data['X_test'], Y_test_f,
                n_input, n_focused, targets_path, focused_idx)

            elapsed = time.time() - t0
            patient_times.append(elapsed)
            completed.append(patient_id)
            print(f"\n  DONE: {patient_id} in {elapsed:.1f}s  "
                  f"(CC={report['mean_cc']:.3f}, {report['quality_flag']})")

            if len(patient_times) == 1:
                remaining = len(included) - (pi + 1)
                est = elapsed * remaining
                print(f"  Estimated remaining: {est/60:.1f} min "
                      f"({elapsed:.0f}s x {remaining} patients)")

        except Exception as e:
            elapsed = time.time() - t0
            print(f"\n  FAILED: {patient_id} after {elapsed:.1f}s")
            print(f"  Error: {e}")
            traceback.print_exc()
            failed.append((patient_id, str(e)))

    # Cross-patient summary
    print(f"\n\n{'='*70}")
    print(f"ALL PATIENTS COMPLETE")
    print(f"{'='*70}")
    print(f"Completed: {len(completed)}")
    print(f"Excluded:  {len(excluded)} -- {excluded}")
    print(f"Failed:    {len(failed)} -- {failed}")
    if patient_times:
        print(f"Total time: {(time.time() - t_total)/60:.1f} min "
              f"(mean {np.mean(patient_times):.0f}s/patient)")

    if completed:
        summary = build_cross_patient_summary()

        table_path = RESULTS_DIR / 'cross_patient_table.txt'
        with open(table_path, 'w') as f:
            f.write(f"DESCARTES Circuit 5 -- Cross-Patient Probing Summary\n")
            f.write(f"Patients: {len(completed)} completed, "
                    f"{len(excluded)} excluded, {len(failed)} failed\n")
            f.write(f"{'='*80}\n")
            f.write(f"{'Target':<25} {'Strong':>6} {'Cand+':>6} {'Zombie':>6} "
                    f"{'Invld':>5} {'Mean D':>10} {'Learned?':>10}\n")
            f.write(f"{'-'*80}\n")
            for t in sorted(summary.keys(),
                            key=lambda x: summary[x].get('mean_delta') or -999,
                            reverse=True):
                s = summary[t]
                md = f"{s['mean_delta']:+.4f}" if s['mean_delta'] is not None else "N/A"
                lr = "YES" if s['universally_learned'] else "no"
                f.write(f"{t:<25} {s['n_strong']:>6} {s['n_candidate_or_better']:>6} "
                        f"{s['n_zombie']:>6} {s['n_invalid']:>5} {md:>10} {lr:>10}\n")
        print(f"\nSaved: {table_path}")


if __name__ == '__main__':
    main()
