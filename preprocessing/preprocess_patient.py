"""
preprocessing/preprocess_patient.py

Full preprocessing pipeline for one patient:
  1. Extract movie-epoch spikes from NWB
  2. Bin and smooth firing rates
  3. Create sliding windows (X=input, Y=output)
  4. Temporal train/val/test split
  5. Save to preprocessed_data/<patient_id>.npz
"""
import pynwb
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter1d

# ── Region mappings (from Phase 0 exploration) ──
INPUT_LABELS = ['Left amygdala', 'Right amygdala',
                'Left hippocampus', 'Right hippocampus']
OUTPUT_LABELS = ['Left ACC', 'Right ACC',
                 'Left preSMA', 'Right preSMA',
                 'Left vmPFC', 'Right vmPFC']

# ── Window parameters ──
BIN_MS = 20          # 20 ms bins → 50 Hz
WINDOW_MS = 2000     # 2-second windows
STRIDE_MS = 500      # 500 ms stride
SIGMA_BINS = 2.0     # 40 ms Gaussian smoothing kernel
MIN_FIRING_RATE = 0.5  # Hz — exclude units below this


def get_neuron_region(nwb, unit_idx):
    """Resolve a unit's brain region string via electrodes → electrode_groups."""
    electrode_row = nwb.units['electrodes'][unit_idx]
    if hasattr(electrode_row, 'iloc') and 'group_name' in electrode_row.columns:
        gname = electrode_row['group_name'].iloc[0]
        if gname in nwb.electrode_groups:
            return nwb.electrode_groups[gname].location
    if hasattr(electrode_row, 'iloc') and 'location' in electrode_row.columns:
        return electrode_row['location'].iloc[0]
    return 'unknown'


def preprocess_patient(nwb_path, output_dir='preprocessed_data'):
    """Full preprocessing for one patient's movie-watching session."""
    nwb_path = Path(nwb_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subject_id = nwb_path.parent.name  # e.g., sub-CS48

    print(f"\n{'='*70}")
    print(f"PREPROCESSING: {subject_id} ({nwb_path.name})")
    print(f"{'='*70}")

    with pynwb.NWBHDF5IO(str(nwb_path), 'r') as io:
        nwb = io.read()

        # ── 1. Identify movie epoch from trials table ──
        # Trial 0 is encoding (movie), stim_phase='encoding'
        movie_start = nwb.trials['start_time'][0]
        movie_stop = nwb.trials['stop_time'][0]
        movie_duration_s = movie_stop - movie_start
        print(f"Movie epoch: {movie_start:.2f} – {movie_stop:.2f} s "
              f"(duration={movie_duration_s:.1f} s)")

        # ── 2. Sort neurons into input / output ──
        n_units = len(nwb.units)
        input_ids, output_ids = [], []
        input_spikes, output_spikes = [], []
        input_regions, output_regions = [], []

        for i in range(n_units):
            region = get_neuron_region(nwb, i)
            all_spikes = nwb.units['spike_times'][i]
            movie_spikes = all_spikes[
                (all_spikes >= movie_start) & (all_spikes <= movie_stop)
            ] - movie_start  # zero-reference

            # Firing rate filter
            fr = len(movie_spikes) / movie_duration_s
            if fr < MIN_FIRING_RATE:
                continue

            if region in INPUT_LABELS:
                input_ids.append(i)
                input_spikes.append(movie_spikes)
                input_regions.append(region)
            elif region in OUTPUT_LABELS:
                output_ids.append(i)
                output_spikes.append(movie_spikes)
                output_regions.append(region)

        print(f"Input neurons (>={MIN_FIRING_RATE} Hz): {len(input_ids)}")
        print(f"Output neurons (>={MIN_FIRING_RATE} Hz): {len(output_ids)}")

        # ── 3. Bin spike trains ──
        duration_ms = movie_duration_s * 1000.0
        n_bins = int(np.ceil(duration_ms / BIN_MS))
        bin_edges = np.arange(0, duration_ms + BIN_MS, BIN_MS) / 1000.0

        def bin_spikes(spike_list):
            arr = np.zeros((len(spike_list), n_bins), dtype=np.float32)
            for i, st in enumerate(spike_list):
                if len(st) > 0:
                    counts, _ = np.histogram(st, bins=bin_edges)
                    arr[i, :len(counts)] = counts
            return arr

        input_binned = bin_spikes(input_spikes)
        output_binned = bin_spikes(output_spikes)

        # ── 4. Gaussian smoothing ──
        input_rates = gaussian_filter1d(input_binned.astype(np.float32),
                                        sigma=SIGMA_BINS, axis=1)
        output_rates = gaussian_filter1d(output_binned.astype(np.float32),
                                         sigma=SIGMA_BINS, axis=1)

        # ── 4b. Per-neuron z-score normalization ──
        # Critical for LSTM training: raw spike counts vary wildly across
        # neurons. Without normalization, the MSE loss is dominated by
        # high-firing neurons and the model ignores low-firing ones.
        def zscore_neurons(rates):
            mu = rates.mean(axis=1, keepdims=True)
            std = rates.std(axis=1, keepdims=True)
            std[std < 1e-6] = 1.0  # avoid division by zero for silent neurons
            return ((rates - mu) / std).astype(np.float32), mu, std

        input_rates, input_mu, input_std = zscore_neurons(input_rates)
        output_rates, output_mu, output_std = zscore_neurons(output_rates)
        print(f"Continuous rates shape: input={input_rates.shape}, output={output_rates.shape}")
        print(f"Z-scored: input range [{input_rates.min():.2f}, {input_rates.max():.2f}], "
              f"output range [{output_rates.min():.2f}, {output_rates.max():.2f}]")

    # ── 5. Sliding windows ──
    window_bins = int(WINDOW_MS / BIN_MS)   # 100
    stride_bins = int(STRIDE_MS / BIN_MS)   # 25
    T = input_rates.shape[1]
    n_windows = (T - window_bins) // stride_bins + 1

    X = np.zeros((n_windows, window_bins, len(input_ids)), dtype=np.float32)
    Y = np.zeros((n_windows, window_bins, len(output_ids)), dtype=np.float32)
    window_times = np.zeros(n_windows, dtype=np.float32)

    for w in range(n_windows):
        s = w * stride_bins
        e = s + window_bins
        X[w] = input_rates[:, s:e].T
        Y[w] = output_rates[:, s:e].T
        window_times[w] = s * BIN_MS / 1000.0

    print(f"Windows: {n_windows} (shape X={X.shape}, Y={Y.shape})")

    # ── 6. Temporal train / val / test split ──
    # 60% train / 20% val / 20% test  BY TIME (no leakage)
    train_end = int(n_windows * 0.60)
    val_end = int(n_windows * 0.80)

    X_train, Y_train = X[:train_end], Y[:train_end]
    X_val, Y_val = X[train_end:val_end], Y[train_end:val_end]
    X_test, Y_test = X[val_end:], Y[val_end:]

    print(f"Split: train={X_train.shape[0]}, val={X_val.shape[0]}, test={X_test.shape[0]}")

    # ── 7. Save ──
    out_file = output_dir / f"{subject_id}.npz"
    np.savez_compressed(
        out_file,
        X_train=X_train, Y_train=Y_train,
        X_val=X_val, Y_val=Y_val,
        X_test=X_test, Y_test=Y_test,
        input_rates=input_rates, output_rates=output_rates,
        window_times=window_times,
        input_ids=np.array(input_ids),
        output_ids=np.array(output_ids),
        input_regions=np.array(input_regions),
        output_regions=np.array(output_regions),
        movie_start=movie_start, movie_stop=movie_stop,
        bin_ms=BIN_MS, window_ms=WINDOW_MS, stride_ms=STRIDE_MS,
        n_input=len(input_ids), n_output=len(output_ids),
    )
    print(f"Saved: {out_file} ({out_file.stat().st_size / 1e6:.1f} MB)")
    return out_file


if __name__ == "__main__":
    import sys
    nwb_path = sys.argv[1] if len(sys.argv) > 1 else \
        "data/000623/sub-CS48/sub-CS48_ses-P48CSR1_behavior+ecephys.nwb"
    preprocess_patient(nwb_path)
