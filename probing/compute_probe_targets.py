"""
probing/compute_probe_targets.py

Compute all 11 probe targets for one patient and save to probe_targets/<patient_id>.npz.
Probe targets are continuous time series at the same 20ms bin resolution as the neural data.

Targets:
  Domain 1 - Emotion:     valence, arousal, threat_pe, emotion_category
  Domain 2 - Social:      event_boundaries, face_presence
  Domain 3 - Dynamics:    theta_power, gamma_power, population_synchrony, temporal_stability
  Domain 4 - Memory:      encoding_success

Note: short_faceannots.pkl is loaded from the official Rutishauser lab repository
(bmovie-release-NWB-BIDS), a trusted source within this analysis pipeline.
"""
import numpy as np
import pandas as pd
import pickle
import pynwb
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert, butter, filtfilt

BIN_MS = 20  # must match preprocessing


# ================================================================
# DOMAIN 1: EMOTION (derived from scene-cut arousal proxy)
# ================================================================

def compute_arousal_from_cuts(scene_cut_times, n_bins, bin_ms=20):
    """Scene cut density as arousal proxy (higher cut rate = more tension)."""
    signal = np.zeros(n_bins, dtype=np.float32)
    boundary_bins = (scene_cut_times * 1000.0 / bin_ms).astype(int)
    boundary_bins = boundary_bins[(boundary_bins >= 0) & (boundary_bins < n_bins)]
    signal[boundary_bins] = 1.0
    # 10-second Gaussian kernel for cut density
    sigma = 10.0 * 1000.0 / bin_ms  # 500 bins
    arousal = gaussian_filter1d(signal, sigma=sigma)
    if arousal.max() > 0:
        arousal = arousal / arousal.max()
    return arousal.astype(np.float32)


def compute_arousal_from_pupil(nwb, movie_start, movie_stop, n_bins, bin_ms=20):
    """Extract pupil diameter as arousal proxy.

    Pipeline: raw pixels -> interpolate blinks (zero values) ->
    detrend (high-pass) -> z-score -> bin to 20ms -> smooth 2s kernel.

    The PupilTracking TimeSeries uses uniform sampling (rate=500 Hz),
    NOT explicit timestamps — we reconstruct time from rate + starting_time.
    """
    try:
        from scipy.interpolate import interp1d
        from scipy.signal import detrend

        pupil = nwb.processing['behavior']['PupilTracking']
        pupil_ts = pupil.time_series['TimeSeries']

        data = pupil_ts.data[:].astype(np.float64)
        if data.ndim == 2:
            data = data[:, 0]

        # Reconstruct timestamps from rate (uniform sampling)
        rate = float(pupil_ts.rate)
        t0 = float(pupil_ts.starting_time) if pupil_ts.starting_time else 0.0
        timestamps = t0 + np.arange(len(data)) / rate

        # Clip to movie epoch
        mask = (timestamps >= movie_start) & (timestamps <= movie_stop)
        pupil_movie = data[mask].copy()
        ts_movie = timestamps[mask] - movie_start

        if len(pupil_movie) < 100:
            print("  Pupil: too few samples in movie epoch")
            return None

        # ── Blink interpolation ──
        # Blinks appear as 0 (pupil contour disappears) or very small values
        blink_threshold = 10.0  # pixels; real pupils are typically 50-200 px
        blink_mask = pupil_movie < blink_threshold
        n_blinks = blink_mask.sum()
        pct_blinks = 100.0 * n_blinks / len(pupil_movie)
        print(f"  Pupil: {n_blinks} blink samples ({pct_blinks:.1f}% of movie)")

        if pct_blinks > 50:
            print("  Pupil: >50% blinks, unreliable signal")
            return None

        # Replace blinks with linear interpolation
        valid = ~blink_mask & np.isfinite(pupil_movie)
        if valid.sum() < 100:
            return None
        f_interp = interp1d(ts_movie[valid], pupil_movie[valid],
                            kind='linear', fill_value='extrapolate',
                            bounds_error=False)
        pupil_clean = f_interp(ts_movie)

        # ── Detrend (remove slow drift from adaptation) ──
        pupil_detrended = detrend(pupil_clean, type='linear')

        # ── Z-score ──
        mu = np.mean(pupil_detrended)
        std = np.std(pupil_detrended)
        if std < 1e-6:
            print("  Pupil: near-zero variance after detrend")
            return None
        pupil_z = (pupil_detrended - mu) / std

        # ── Downsample to 20ms bins ──
        duration_ms = (movie_stop - movie_start) * 1000.0
        bin_edges_s = np.arange(0, duration_ms + bin_ms, bin_ms) / 1000.0
        binned = np.zeros(n_bins, dtype=np.float32)
        for i in range(n_bins):
            if i + 1 < len(bin_edges_s):
                mask_bin = (ts_movie >= bin_edges_s[i]) & (ts_movie < bin_edges_s[i+1])
                if mask_bin.any():
                    binned[i] = np.mean(pupil_z[mask_bin])
                elif i > 0:
                    binned[i] = binned[i-1]

        # ── Smooth with 2-second kernel ──
        sigma = 2.0 * 1000.0 / bin_ms  # 100 bins
        binned = gaussian_filter1d(binned, sigma=sigma)
        return binned.astype(np.float32)

    except Exception as e:
        import traceback
        print(f"  Pupil extraction failed: {e}")
        traceback.print_exc()
        return None


def compute_valence(arousal):
    """For Hitchcock suspense: high arousal = negative valence (fear)."""
    return (-arousal).astype(np.float32)


def compute_threat_pe(arousal, smooth_sigma=25.0):
    """Threat prediction error: observed arousal - slow prediction."""
    predicted = gaussian_filter1d(arousal, sigma=smooth_sigma)
    return (arousal - predicted).astype(np.float32)


def compute_emotion_category(arousal, valence, threshold=0.5):
    """Discrete categories: 0=neutral, 1=fear, 2=surprise, 3=relief."""
    n = len(arousal)
    cats = np.zeros(n, dtype=np.int32)
    cats[(arousal > threshold) & (valence < -threshold)] = 1  # fear
    deriv = np.gradient(arousal)
    cats[deriv > np.percentile(deriv, 90)] = 2  # surprise
    relief_mask = (deriv < np.percentile(deriv, 10)) & (np.roll(arousal, 5) > threshold)
    cats[relief_mask] = 3  # relief
    return cats


# ================================================================
# DOMAIN 2: SOCIAL / NARRATIVE
# ================================================================

def compute_event_boundaries(scene_cut_times, n_bins, bin_ms=20, sigma=5.0):
    """Smooth impulse at each scene boundary."""
    signal = np.zeros(n_bins, dtype=np.float32)
    boundary_bins = (scene_cut_times * 1000.0 / bin_ms).astype(int)
    boundary_bins = boundary_bins[(boundary_bins >= 0) & (boundary_bins < n_bins)]
    signal[boundary_bins] = 1.0
    return gaussian_filter1d(signal, sigma=sigma).astype(np.float32)


def compute_face_presence(face_pkl_path, n_bins, movie_duration_s, bin_ms=20, fps=24.0):
    """
    Binary face presence signal from frame annotations.
    Note: The pickle file is from the official Rutishauser lab bmovie-release-NWB-BIDS
    repository, a trusted source within this analysis pipeline.
    """
    try:
        with open(face_pkl_path, 'rb') as f:
            face_data = pickle.load(f)

        signal = np.zeros(n_bins, dtype=np.float32)
        for key, frame_info in face_data.items():
            # key like 'frame_118'
            frame_num = int(key.split('_')[1])
            time_s = frame_num / fps
            bin_idx = int(time_s * 1000.0 / bin_ms)
            if 0 <= bin_idx < n_bins:
                n_faces = len(frame_info)
                signal[bin_idx] = max(signal[bin_idx], n_faces)

        # Smooth to 100ms resolution
        sigma = 0.1 * 1000.0 / bin_ms  # 5 bins
        return gaussian_filter1d(signal, sigma=sigma).astype(np.float32)

    except Exception as e:
        print(f"  Face annotation loading failed: {e}")
        return np.zeros(n_bins, dtype=np.float32)


# ================================================================
# DOMAIN 3: EMERGENT DYNAMICS (from neural data)
# ================================================================

def compute_population_synchrony(spike_rates, window_bins=10):
    """Mean pairwise |correlation| in sliding window."""
    n_neurons, T = spike_rates.shape
    if n_neurons < 2:
        return np.zeros(T, dtype=np.float32)

    n_windows = T - window_bins + 1
    sync = np.zeros(n_windows, dtype=np.float32)

    for t in range(n_windows):
        window = spike_rates[:, t:t+window_bins]
        # Check for constant rows
        stds = np.std(window, axis=1)
        active = stds > 0
        if active.sum() < 2:
            continue
        corr = np.corrcoef(window[active])
        upper = corr[np.triu_indices(active.sum(), k=1)]
        sync[t] = np.nanmean(np.abs(upper))

    # Pad to original length
    pad = T - n_windows
    sync = np.concatenate([sync, np.full(pad, sync[-1] if len(sync) > 0 else 0)])
    return sync.astype(np.float32)


def compute_temporal_stability(spike_rates, lag=5):
    """Cosine similarity of population vector at lag (100ms at 20ms bins)."""
    n_neurons, T = spike_rates.shape
    stability = np.zeros(T, dtype=np.float32)

    for t in range(lag, T):
        curr = spike_rates[:, t]
        prev = spike_rates[:, t - lag]
        nc = np.linalg.norm(curr)
        np_ = np.linalg.norm(prev)
        if nc > 0 and np_ > 0:
            stability[t] = np.dot(curr, prev) / (nc * np_)

    stability[:lag] = stability[lag]
    return stability.astype(np.float32)


def _get_limbic_channel_columns(es):
    """Identify LFP data matrix columns corresponding to amygdala + hippocampus.

    The LFP data matrix has shape (n_samples, n_electrodes). The column index
    is NOT the same as the electrode table index — we need to map:
      electrode_indices[i] -> column i in the data matrix.

    Returns: list of column indices for limbic (amygdala + hippocampus) channels.
    """
    limbic_keywords = ['amygdala', 'hippocampus']
    elec_indices = es.electrodes.data[:]
    etable = es.electrodes.table.to_dataframe()

    limbic_cols = []
    limbic_locs = []
    for col_idx, elec_idx in enumerate(elec_indices):
        loc = str(etable.iloc[elec_idx]['location']).lower()
        if any(kw in loc for kw in limbic_keywords):
            limbic_cols.append(col_idx)
            limbic_locs.append(etable.iloc[elec_idx]['location'])

    return limbic_cols, limbic_locs


def _extract_lfp_band_power(nwb, movie_start, movie_stop, n_bins, bin_ms,
                             band_low, band_high, band_name):
    """Generic LFP band power extraction from limbic channels.

    Steps:
    1. Select amygdala + hippocampus channels via electrode metadata
    2. Extract movie epoch from LFP data
    3. Bandpass filter (Butterworth order 4)
    4. Hilbert transform -> amplitude envelope
    5. Average across limbic channels
    6. Downsample to 20ms bins
    """
    lfp_container = nwb.processing['ecephys']['LFP_macro']
    es = list(lfp_container.electrical_series.values())[0]

    lfp_rate = float(es.rate) if es.rate else 1000.0
    data = es.data  # (n_samples, n_channels) — lazy HDF5
    n_samples_total = data.shape[0]

    # ── Select limbic channels ──
    limbic_cols, limbic_locs = _get_limbic_channel_columns(es)
    if len(limbic_cols) == 0:
        print(f"  {band_name}: no limbic channels found in electrode table")
        return None
    print(f"  {band_name}: using {len(limbic_cols)} limbic channels: "
          f"{set(limbic_locs)}")

    # ── Movie epoch sample range ──
    t0_lfp = float(es.starting_time) if es.starting_time is not None else 0.0
    start_sample = int((movie_start - t0_lfp) * lfp_rate)
    end_sample = int((movie_stop - t0_lfp) * lfp_rate)
    start_sample = max(0, start_sample)
    end_sample = min(n_samples_total, end_sample)

    n_movie_samples = end_sample - start_sample
    if n_movie_samples < 2000:
        print(f"  {band_name}: only {n_movie_samples} LFP samples in movie epoch")
        return None

    # ── Load limbic channel data for movie epoch ──
    # Read full slice then index columns (HDF5 is row-major, so this is fast)
    lfp_movie = data[start_sample:end_sample, :]
    lfp_limbic = lfp_movie[:, limbic_cols].astype(np.float64)
    print(f"  {band_name}: LFP shape = {lfp_limbic.shape}, rate = {lfp_rate} Hz")

    # ── Nyquist check ──
    nyq = lfp_rate / 2.0
    if band_high / nyq >= 1.0:
        print(f"  {band_name}: LFP rate {lfp_rate} Hz too low for {band_high} Hz")
        return None

    # ── Bandpass + Hilbert per channel, then average ──
    b, a = butter(4, [band_low / nyq, band_high / nyq], btype='band')
    envelopes = np.zeros((lfp_limbic.shape[0], lfp_limbic.shape[1]),
                         dtype=np.float64)
    for ch in range(lfp_limbic.shape[1]):
        filt = filtfilt(b, a, lfp_limbic[:, ch])
        envelopes[:, ch] = np.abs(hilbert(filt))

    # Average envelope across limbic channels
    mean_env = np.mean(envelopes, axis=1).astype(np.float32)
    print(f"  {band_name}: envelope range [{mean_env.min():.6f}, {mean_env.max():.6f}], "
          f"std={mean_env.std():.6f}")

    # ── Downsample to bins ──
    samples_per_bin = int(bin_ms / 1000.0 * lfp_rate)
    n_bins_lfp = len(mean_env) // samples_per_bin
    n_use = min(n_bins, n_bins_lfp)

    binned = np.array([
        np.mean(mean_env[i*samples_per_bin:(i+1)*samples_per_bin])
        for i in range(n_use)
    ], dtype=np.float32)

    # Pad or trim to target n_bins
    if len(binned) < n_bins:
        binned = np.pad(binned, (0, n_bins - len(binned)), mode='edge')
    else:
        binned = binned[:n_bins]

    # Z-score for probe target standardization
    mu, std = binned.mean(), binned.std()
    if std > 1e-8:
        binned = (binned - mu) / std

    return binned.astype(np.float32)


def compute_theta_power(nwb, movie_start, movie_stop, n_bins, bin_ms=20,
                        input_labels=None):
    """Theta (4-8 Hz) power from limbic LFP channels."""
    try:
        return _extract_lfp_band_power(
            nwb, movie_start, movie_stop, n_bins, bin_ms,
            band_low=4.0, band_high=8.0, band_name="Theta")
    except Exception as e:
        import traceback
        print(f"  Theta extraction failed: {e}")
        traceback.print_exc()
        return None


def compute_gamma_power(nwb, movie_start, movie_stop, n_bins, bin_ms=20):
    """Gamma (30-80 Hz) power from limbic LFP channels."""
    try:
        return _extract_lfp_band_power(
            nwb, movie_start, movie_stop, n_bins, bin_ms,
            band_low=30.0, band_high=80.0, band_name="Gamma")
    except Exception as e:
        import traceback
        print(f"  Gamma extraction failed: {e}")
        traceback.print_exc()
        return None


# ================================================================
# DOMAIN 4: MEMORY
# ================================================================

def compute_encoding_success(nwb, movie_start, movie_stop, n_bins, bin_ms=20):
    """
    Map recognition hits/misses back to movie timepoints.
    Trial 0 = movie, Trials 1+ = recognition test.
    """
    try:
        n_trials = len(nwb.trials)
        if n_trials <= 1:
            return None

        # Get recognition trial data
        stim_phases = [str(nwb.trials['stim_phase'][i]) for i in range(n_trials)]
        responses_correct = []
        stim_files = []
        for i in range(n_trials):
            try:
                responses_correct.append(int(nwb.trials['response_correct'][i]))
            except:
                responses_correct.append(0)
            try:
                stim_files.append(str(nwb.trials['stimulus_file'][i]))
            except:
                stim_files.append('')

        # Find recognition trials (not encoding)
        rec_trials = [i for i in range(n_trials) if stim_phases[i] != 'encoding']
        if len(rec_trials) == 0:
            return None

        # Without explicit movie timestamps for each test frame,
        # we create a proxy: distribute recognition trials evenly
        # across the movie duration and label hits vs misses
        movie_dur_s = movie_stop - movie_start
        signal = np.zeros(n_bins, dtype=np.float32)

        n_rec = len(rec_trials)
        for idx, trial_i in enumerate(rec_trials):
            # Estimate movie position: distribute evenly
            movie_time = (idx / max(n_rec - 1, 1)) * movie_dur_s
            bin_idx = int(movie_time * 1000.0 / bin_ms)
            bin_idx = min(bin_idx, n_bins - 1)

            # Hit = 1, Miss = -1
            if responses_correct[trial_i]:
                signal[bin_idx] = 1.0
            else:
                signal[bin_idx] = -1.0

        # Smooth with 5-second kernel
        sigma = 5.0 * 1000.0 / bin_ms  # 250 bins
        signal = gaussian_filter1d(signal, sigma=sigma)

        # Normalize to [-1, 1]
        max_abs = np.max(np.abs(signal))
        if max_abs > 0:
            signal = signal / max_abs

        return signal.astype(np.float32)

    except Exception as e:
        print(f"  Encoding success computation failed: {e}")
        return None


# ================================================================
# MAIN: Compute all targets for one patient
# ================================================================

def compute_all_probe_targets(nwb_path, preprocessed_path, annotations_dir,
                               output_dir='probe_targets'):
    """Compute all 11 probe targets and save to .npz."""
    nwb_path = Path(nwb_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subject_id = nwb_path.parent.name

    # Load preprocessed data for dimensions and metadata
    data = np.load(preprocessed_path, allow_pickle=True)
    input_rates = data['input_rates']   # (n_input, T)
    output_rates = data['output_rates'] # (n_output, T)
    movie_start = float(data['movie_start'])
    movie_stop = float(data['movie_stop'])
    n_bins = input_rates.shape[1]

    print(f"\n{'='*70}")
    print(f"COMPUTING PROBE TARGETS: {subject_id}")
    print(f"{'='*70}")
    print(f"Continuous data: {n_bins} bins, movie {movie_start:.1f}-{movie_stop:.1f}s")

    targets = {}

    # ── Load scene cut annotations ──
    annotations_path = Path(annotations_dir)
    scene_csv = annotations_path / 'scenecut_info.csv'
    if scene_csv.exists():
        scene_df = pd.read_csv(scene_csv)
        scene_cut_times = scene_df['shot_start_t'].values.astype(float)
        print(f"Scene cuts: {len(scene_cut_times)} boundaries loaded")
    else:
        scene_cut_times = np.array([])
        print("WARNING: scenecut_info.csv not found")

    # ── DOMAIN 1: EMOTION ──
    print("\n--- Domain 1: Emotion ---")

    # Arousal from scene cuts
    arousal_cuts = compute_arousal_from_cuts(scene_cut_times, n_bins, BIN_MS)
    print(f"  Arousal (cut density): range [{arousal_cuts.min():.3f}, {arousal_cuts.max():.3f}]")

    # Try pupil arousal
    pupil_arousal = None
    print("  Attempting pupil diameter extraction...")
    with pynwb.NWBHDF5IO(str(nwb_path), 'r') as io:
        nwb = io.read()
        pupil_arousal = compute_arousal_from_pupil(nwb, movie_start, movie_stop,
                                                    n_bins, BIN_MS)
        if pupil_arousal is not None:
            print(f"  Pupil arousal: range [{pupil_arousal.min():.3f}, {pupil_arousal.max():.3f}]")
            # Combine: average of cut-based and pupil-based
            arousal = 0.5 * arousal_cuts + 0.5 * (pupil_arousal - pupil_arousal.min()) / \
                      max(pupil_arousal.max() - pupil_arousal.min(), 1e-6)
        else:
            print("  Pupil unavailable, using cut density only")
            arousal = arousal_cuts

    targets['arousal'] = arousal
    targets['valence'] = compute_valence(arousal)
    targets['threat_pe'] = compute_threat_pe(arousal)
    targets['emotion_category'] = compute_emotion_category(arousal, targets['valence'])
    print(f"  Valence: range [{targets['valence'].min():.3f}, {targets['valence'].max():.3f}]")
    print(f"  Threat PE: range [{targets['threat_pe'].min():.3f}, {targets['threat_pe'].max():.3f}]")
    cats, counts = np.unique(targets['emotion_category'], return_counts=True)
    print(f"  Emotion categories: {dict(zip(cats, counts))}")

    # ── DOMAIN 2: SOCIAL / NARRATIVE ──
    print("\n--- Domain 2: Social/Narrative ---")
    targets['event_boundaries'] = compute_event_boundaries(scene_cut_times, n_bins, BIN_MS)
    print(f"  Event boundaries: {(targets['event_boundaries'] > 0.01).sum()} active bins")

    face_pkl = annotations_path / 'short_faceannots.pkl'
    movie_dur_s = movie_stop - movie_start
    targets['face_presence'] = compute_face_presence(face_pkl, n_bins, movie_dur_s, BIN_MS)
    print(f"  Face presence: {(targets['face_presence'] > 0).sum()} bins with faces")

    # ── DOMAIN 3: EMERGENT DYNAMICS ──
    print("\n--- Domain 3: Emergent Dynamics ---")

    targets['population_synchrony'] = compute_population_synchrony(input_rates)
    print(f"  Pop. synchrony (input): mean={targets['population_synchrony'].mean():.4f}")

    targets['temporal_stability'] = compute_temporal_stability(input_rates)
    print(f"  Temporal stability: mean={targets['temporal_stability'].mean():.4f}")

    # LFP-based targets
    print("  Attempting LFP extraction for theta/gamma...")
    with pynwb.NWBHDF5IO(str(nwb_path), 'r') as io:
        nwb = io.read()
        theta = compute_theta_power(nwb, movie_start, movie_stop, n_bins, BIN_MS)
        if theta is not None:
            targets['theta_power'] = theta
            print(f"  Theta power: mean={theta.mean():.4f}, std={theta.std():.4f}")
        else:
            print("  Theta: UNAVAILABLE (using synchrony as fallback proxy)")
            targets['theta_power'] = gaussian_filter1d(
                targets['population_synchrony'], sigma=50).astype(np.float32)

        gamma = compute_gamma_power(nwb, movie_start, movie_stop, n_bins, BIN_MS)
        if gamma is not None:
            targets['gamma_power'] = gamma
            print(f"  Gamma power: mean={gamma.mean():.4f}, std={gamma.std():.4f}")
        else:
            print("  Gamma: UNAVAILABLE (using stability as fallback proxy)")
            targets['gamma_power'] = gaussian_filter1d(
                targets['temporal_stability'], sigma=10).astype(np.float32)

    # ── DOMAIN 4: MEMORY ──
    print("\n--- Domain 4: Memory ---")
    with pynwb.NWBHDF5IO(str(nwb_path), 'r') as io:
        nwb = io.read()
        enc_success = compute_encoding_success(nwb, movie_start, movie_stop, n_bins, BIN_MS)
        if enc_success is not None:
            targets['encoding_success'] = enc_success
            print(f"  Encoding success: range [{enc_success.min():.3f}, {enc_success.max():.3f}]")
        else:
            print("  Encoding success: UNAVAILABLE")
            targets['encoding_success'] = np.zeros(n_bins, dtype=np.float32)

    # ── Save ──
    out_file = output_dir / f"{subject_id}.npz"
    np.savez_compressed(out_file, **targets)
    print(f"\nSaved {len(targets)} probe targets to {out_file}")
    print(f"Targets: {list(targets.keys())}")

    return targets


if __name__ == "__main__":
    import sys

    nwb_path = sys.argv[1] if len(sys.argv) > 1 else \
        "data/000623/sub-CS48/sub-CS48_ses-P48CSR1_behavior+ecephys.nwb"
    preprocessed = sys.argv[2] if len(sys.argv) > 2 else \
        "preprocessed_data/sub-CS48.npz"
    annotations = sys.argv[3] if len(sys.argv) > 3 else \
        "bmovie-release-NWB-BIDS/assets/annotations"

    compute_all_probe_targets(nwb_path, preprocessed, annotations)
