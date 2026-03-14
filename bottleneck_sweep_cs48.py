"""
bottleneck_sweep_cs48.py

Systematically vary hidden_size = [16, 32, 64, 128, 256] on the
focused (top-10 neuron) model for sub-CS48.

For each hidden size:
  - Train 5 seeds, compute mean CC
  - Ridge dR2 (continuous) / Logistic dAccuracy (categorical) for all probes
  - Resample ablation at k = [3, 5, 10] WITH random-dimension control
  - Report tables:
      hidden_size x probe_target -> delta_metric
      hidden_size x k x (random vs target) -> z_score

Hypothesis: smaller hidden sizes force modular representations,
rescuing the ablation methodology and revealing mandatory intermediates.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import json
import time

# ================================================================
# CONFIGURATION
# ================================================================
PATIENT_ID = 'sub-CS48'
PREPROCESSED_PATH = f'preprocessed_data/{PATIENT_ID}.npz'
PROBE_TARGETS_PATH = f'probe_targets/{PATIENT_ID}.npz'
PILOT_REPORT_PATH = f'results/{PATIENT_ID}/pilot_report.json'
OUTPUT_DIR = Path(f'results/{PATIENT_ID}')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Sweep parameters ──
HIDDEN_SIZES = [16, 32, 64, 128, 256]
N_LAYERS = 2
N_SEEDS = 5            # 5 per hidden size (not 10, to save time)
MAX_EPOCHS = 200
BATCH_SIZE = 32
LR = 1e-3
PATIENCE = 20
DROPOUT = 0.1

# ── Probing thresholds ──
DELTA_R2_THRESHOLD = 0.05
DELTA_ACC_THRESHOLD = 0.02
ABLATION_Z_THRESHOLD = -2.0
N_RESAMPLES = 100
K_VALUES = [3, 5, 10]           # smaller k — at h=16, k=10 = 63%
N_RANDOM_ABLATION_REPEATS = 5

TOP_N_NEURONS = 10
CC_THRESHOLD = 0.3

CATEGORICAL_TARGETS = {'emotion_category'}
SKIP_TARGETS = {'valence'}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")


# ================================================================
# MODEL
# ================================================================
class LimbicPrefrontalLSTM(nn.Module):

    def __init__(self, n_input, n_output, hidden_size=128,
                 n_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.input_proj = nn.Linear(n_input, hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        # Output projection: hidden_size -> hidden_size//2 -> n_output
        # At small hidden sizes (h=16), h//2 = 8, which is fine
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, max(hidden_size // 2, 4)),
            nn.ReLU(),
            nn.Linear(max(hidden_size // 2, 4), n_output)
        )

    def forward(self, x, return_hidden=False):
        projected = self.input_proj(x)
        lstm_out, _ = self.lstm(projected)
        y_pred = self.output_proj(lstm_out)
        if return_hidden:
            return y_pred, lstm_out
        return y_pred


# ================================================================
# TRAINING
# ================================================================
def train_one_seed(n_input, n_output, hidden_size, X_train, Y_train,
                   X_val, Y_val, seed, device=DEVICE):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = LimbicPrefrontalLSTM(
        n_input=n_input, n_output=n_output,
        hidden_size=hidden_size, n_layers=N_LAYERS, dropout=DROPOUT)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=PATIENCE // 2, factor=0.5)
    criterion = nn.MSELoss()

    X_tr = torch.FloatTensor(X_train).to(device)
    Y_tr = torch.FloatTensor(Y_train).to(device)
    X_v = torch.FloatTensor(X_val).to(device)
    Y_v = torch.FloatTensor(Y_val).to(device)

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    final_epoch = 0

    for ep in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(len(X_tr))
        n_batches = max(1, len(X_tr) // BATCH_SIZE)
        epoch_loss = 0.0

        for b in range(n_batches):
            idx = perm[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            optimizer.zero_grad()
            y_pred = model(X_tr[idx])
            loss = criterion(y_pred, Y_tr[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= n_batches
        model.train(False)
        with torch.no_grad():
            val_pred = model(X_v)
            val_loss = criterion(val_pred, Y_v).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                final_epoch = ep + 1
                break

        final_epoch = ep + 1

    model.load_state_dict(best_state)
    model.train(False)

    with torch.no_grad():
        val_pred = model(X_v).cpu().numpy()

    cc_values = []
    for j in range(Y_val.shape[2]):
        pred_flat = val_pred[:, :, j].flatten()
        true_flat = Y_val[:, :, j].flatten()
        if np.std(pred_flat) > 0 and np.std(true_flat) > 0:
            cc = np.corrcoef(pred_flat, true_flat)[0, 1]
        else:
            cc = 0.0
        cc_values.append(cc)

    return {
        'seed': seed,
        'best_val_loss': best_val_loss,
        'cc_per_neuron': cc_values,
        'mean_cc': float(np.nanmean(cc_values)),
        'epochs_trained': final_epoch,
        'model': model,
    }


# ================================================================
# PROBING FUNCTIONS
# ================================================================
def ridge_delta_r2(h_trained, h_untrained, target, n_folds=5):
    if target.ndim == 1:
        target = target.reshape(-1, 1)
    alphas = np.logspace(-3, 3, 20)
    kf = KFold(n_splits=n_folds, shuffle=False)

    r2_tr = [RidgeCV(alphas=alphas).fit(h_trained[tr], target[tr]).score(
        h_trained[te], target[te]) for tr, te in kf.split(h_trained)]
    r2_un = [RidgeCV(alphas=alphas).fit(h_untrained[tr], target[tr]).score(
        h_untrained[te], target[te]) for tr, te in kf.split(h_untrained)]

    return {
        'r2_trained': float(np.mean(r2_tr)),
        'r2_untrained': float(np.mean(r2_un)),
        'delta_r2': float(np.mean(r2_tr) - np.mean(r2_un)),
    }


def logistic_delta_accuracy(h_trained, h_untrained, labels, n_folds=5):
    unique_classes = np.unique(labels)
    class_counts = {int(c): int((labels == c).sum()) for c in unique_classes}
    majority_class = max(class_counts, key=class_counts.get)
    chance_accuracy = class_counts[majority_class] / len(labels)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=False)
    scaler_tr = StandardScaler()
    scaler_un = StandardScaler()

    acc_trained_folds, acc_untrained_folds = [], []

    for tr_idx, te_idx in skf.split(h_trained, labels):
        X_tr_s = scaler_tr.fit_transform(h_trained[tr_idx])
        X_te_s = scaler_tr.transform(h_trained[te_idx])
        clf = LogisticRegressionCV(
            Cs=10, cv=3, max_iter=1000,
            class_weight='balanced', random_state=42)
        clf.fit(X_tr_s, labels[tr_idx])
        preds_tr = clf.predict(X_te_s)
        acc_trained_folds.append(float(np.mean(preds_tr == labels[te_idx])))

        X_tr_u = scaler_un.fit_transform(h_untrained[tr_idx])
        X_te_u = scaler_un.transform(h_untrained[te_idx])
        clf_un = LogisticRegressionCV(
            Cs=10, cv=3, max_iter=1000,
            class_weight='balanced', random_state=42)
        clf_un.fit(X_tr_u, labels[tr_idx])
        preds_un = clf_un.predict(X_te_u)
        acc_untrained_folds.append(
            float(np.mean(preds_un == labels[te_idx])))

    return {
        'acc_trained': float(np.mean(acc_trained_folds)),
        'acc_untrained': float(np.mean(acc_untrained_folds)),
        'delta_accuracy': float(np.mean(acc_trained_folds) -
                                np.mean(acc_untrained_folds)),
        'chance_accuracy': chance_accuracy,
        'class_distribution': class_counts,
    }


# ================================================================
# HIDDEN STATE EXTRACTION & TARGET ALIGNMENT
# ================================================================
def extract_hidden_flat(model, X, device=DEVICE):
    model.train(False)
    with torch.no_grad():
        X_t = torch.FloatTensor(X).to(device)
        _, hidden = model(X_t, return_hidden=True)
    return hidden.cpu().numpy().reshape(-1, hidden.shape[-1])


def align_target_to_windows(target_continuous, window_start_indices,
                             window_bins, n_windows):
    T_cont = len(target_continuous)
    aligned = []
    for w in range(n_windows):
        s = window_start_indices[w]
        e = s + window_bins
        if e <= T_cont:
            aligned.append(target_continuous[s:e])
        else:
            chunk = target_continuous[s:min(e, T_cont)]
            if len(chunk) < window_bins:
                chunk = np.pad(chunk, (0, window_bins - len(chunk)),
                               mode='edge')
            aligned.append(chunk)
    return np.concatenate(aligned).astype(np.float32)


def clip_to_3sigma(target):
    mu, std = np.mean(target), np.std(target)
    if std < 1e-10:
        return target, 0
    lo, hi = mu - 3 * std, mu + 3 * std
    n_clipped = int(((target < lo) | (target > hi)).sum())
    return np.clip(target, lo, hi), n_clipped


# ================================================================
# ABLATION FUNCTIONS
# ================================================================
def resample_ablation(model, X_test, Y_test, dims_to_ablate,
                      n_resamples=N_RESAMPLES, device=DEVICE):
    model.train(False)
    X_t = torch.FloatTensor(X_test).to(device)

    with torch.no_grad():
        y_baseline, h_baseline = model(X_t, return_hidden=True)
    baseline_mse = float(np.mean(
        (y_baseline.cpu().numpy() - Y_test) ** 2))

    h_np = h_baseline.cpu().numpy()
    h_flat = h_np.reshape(-1, h_np.shape[-1])

    ablated_mses = []
    for _ in range(n_resamples):
        h_resampled = h_np.copy()
        for dim in dims_to_ablate:
            n_total = h_flat.shape[0]
            n_need = h_resampled.shape[0] * h_resampled.shape[1]
            repl_idx = np.random.choice(n_total, size=n_need)
            repl_vals = h_flat[repl_idx, dim]
            h_resampled[:, :, dim] = repl_vals.reshape(
                h_resampled.shape[0], h_resampled.shape[1])

        with torch.no_grad():
            h_t = torch.FloatTensor(h_resampled).to(device)
            y_ablated = model.output_proj(h_t)

        abl_mse = float(np.mean(
            (y_ablated.cpu().numpy() - Y_test) ** 2))
        ablated_mses.append(abl_mse)

    ablated_mses = np.array(ablated_mses)
    mean_abl = float(np.mean(ablated_mses))
    std_abl = float(np.std(ablated_mses))
    z_score = float((baseline_mse - mean_abl) / std_abl) \
        if std_abl > 0 else 0.0

    return {
        'baseline_mse': baseline_mse,
        'mean_ablated_mse': mean_abl,
        'z_score': z_score,
        'relative_degradation': float(
            (mean_abl - baseline_mse) / max(baseline_mse, 1e-10)),
        'n_dims_ablated': len(dims_to_ablate),
    }


def random_ablation_control(model, X_test, Y_test, k, hidden_size,
                             n_repeats=N_RANDOM_ABLATION_REPEATS,
                             n_resamples=N_RESAMPLES, device=DEVICE):
    z_scores = []
    for rep in range(n_repeats):
        random_dims = np.random.choice(hidden_size, size=min(k, hidden_size),
                                       replace=False).tolist()
        result = resample_ablation(model, X_test, Y_test, random_dims,
                                   n_resamples=n_resamples, device=device)
        z_scores.append(result['z_score'])

    return {
        'mean_z': float(np.mean(z_scores)),
        'std_z': float(np.std(z_scores)),
        'individual_z': [float(z) for z in z_scores],
        'k': k,
    }


def identify_top_dims_for_target(h_trained, target, k):
    if target.ndim == 1:
        target = target.reshape(-1, 1)
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 20))
    ridge.fit(h_trained, target)
    coef_mag = np.abs(ridge.coef_).sum(axis=0)
    return np.argsort(coef_mag)[-k:].tolist()


# ================================================================
# MAIN SWEEP
# ================================================================
def main():
    t0_total = time.time()

    print(f"\n{'#'*70}")
    print(f"# BOTTLENECK SWEEP: {PATIENT_ID} focused model")
    print(f"# hidden_size = {HIDDEN_SIZES}")
    print(f"# {N_SEEDS} seeds | k = {K_VALUES} | device = {DEVICE}")
    print(f"{'#'*70}")

    # ── Load pilot report to get top neuron indices ──
    with open(PILOT_REPORT_PATH) as f:
        pilot = json.load(f)
    cc_per_neuron = np.array(pilot['task4_training']['cc_per_neuron_best_seed'])
    sorted_idx = np.argsort(cc_per_neuron)[::-1]
    top_idx = sorted_idx[:TOP_N_NEURONS]
    top_ccs = cc_per_neuron[top_idx]

    print(f"\nFocused on top-{TOP_N_NEURONS} neurons:")
    for rank, (idx, cc) in enumerate(zip(top_idx, top_ccs)):
        print(f"  #{rank}: neuron {idx}, pilot CC={cc:.4f}")

    # ── Load preprocessed data ──
    data = np.load(PREPROCESSED_PATH)
    X_train, Y_train_full = data['X_train'], data['Y_train']
    X_val, Y_val_full = data['X_val'], data['Y_val']
    X_test, Y_test_full = data['X_test'], data['Y_test']
    n_input = int(data['n_input'])
    window_ms = int(data['window_ms'])
    stride_ms = int(data['stride_ms'])
    bin_ms = int(data['bin_ms'])

    # Subset to focused neurons
    Y_train = Y_train_full[:, :, top_idx]
    Y_val = Y_val_full[:, :, top_idx]
    Y_test = Y_test_full[:, :, top_idx]
    n_output = TOP_N_NEURONS

    print(f"Data shapes: X={X_train.shape}, Y_focused={Y_train.shape}")

    # ── Load probe targets ──
    probe_data = np.load(PROBE_TARGETS_PATH)
    probe_target_names = [n for n in probe_data.files
                          if n not in SKIP_TARGETS]
    print(f"Probe targets ({len(probe_target_names)}): {probe_target_names}")

    # Window alignment parameters
    window_bins = int(window_ms / bin_ms)
    stride_bins = int(stride_ms / bin_ms)
    train_end = X_train.shape[0]
    val_end = train_end + X_val.shape[0]
    test_start_window = val_end
    n_test = X_test.shape[0]
    test_window_start_bins = [(test_start_window + w) * stride_bins
                              for w in range(n_test)]

    # ── Pre-align and clip probe targets (once, shared across hidden sizes) ──
    aligned_targets = {}
    for name in probe_target_names:
        target_continuous = probe_data[name]
        target_aligned = align_target_to_windows(
            target_continuous, test_window_start_bins, window_bins, n_test)

        if name in CATEGORICAL_TARGETS:
            aligned_targets[name] = target_aligned.astype(int)
        else:
            clipped, _ = clip_to_3sigma(target_aligned)
            aligned_targets[name] = clipped

    # ══════════════════════════════════════════════════════════════
    # SWEEP LOOP
    # ══════════════════════════════════════════════════════════════
    sweep_results = {}

    for hidden_size in HIDDEN_SIZES:
        t0_h = time.time()
        print(f"\n{'='*70}")
        print(f"HIDDEN SIZE = {hidden_size}")
        print(f"{'='*70}")

        # ── TASK 4: Train ──
        seed_results = []
        for seed in range(N_SEEDS):
            result = train_one_seed(
                n_input, n_output, hidden_size,
                X_train, Y_train, X_val, Y_val, seed)
            seed_results.append(result)
            flag = " ***CC<0.3***" if result['mean_cc'] < CC_THRESHOLD else ""
            print(f"  seed={seed}: CC={result['mean_cc']:.4f}, "
                  f"epochs={result['epochs_trained']}, "
                  f"val_loss={result['best_val_loss']:.4f}{flag}")

        cc_all = [r['mean_cc'] for r in seed_results]
        best_seed_idx = int(np.argmax(cc_all))
        best_model = seed_results[best_seed_idx]['model']
        mean_cc = float(np.mean(cc_all))
        best_cc = float(cc_all[best_seed_idx])
        cc_pass = mean_cc >= CC_THRESHOLD

        print(f"\n  Mean CC = {mean_cc:.4f} +/- {np.std(cc_all):.4f}")
        print(f"  Best seed = {best_seed_idx} (CC={best_cc:.4f})")
        print(f"  CC threshold: {'PASS' if cc_pass else 'FAIL'}")

        # ── TASK 5: Probing ──
        # Create untrained baseline model with same hidden_size
        torch.manual_seed(9999)
        untrained_model = LimbicPrefrontalLSTM(
            n_input=n_input, n_output=n_output,
            hidden_size=hidden_size, n_layers=N_LAYERS, dropout=DROPOUT)
        untrained_model.to(DEVICE)
        untrained_model.train(False)

        h_trained = extract_hidden_flat(best_model, X_test)
        h_untrained = extract_hidden_flat(untrained_model, X_test)

        probe_results = {}
        for name in probe_target_names:
            t_use = aligned_targets[name]
            min_len = min(len(t_use), len(h_trained))
            t_slice = t_use[:min_len]
            h_tr_slice = h_trained[:min_len]
            h_un_slice = h_untrained[:min_len]

            if np.std(t_slice.astype(float)) < 1e-8:
                probe_results[name] = {'delta_metric': 0.0, 'status': 'constant'}
                continue

            if name in CATEGORICAL_TARGETS:
                labels = t_slice.astype(int)
                res = logistic_delta_accuracy(h_tr_slice, h_un_slice, labels)
                delta = res['delta_accuracy']
                if delta > 0.10:
                    status = 'STRONG'
                elif delta > DELTA_ACC_THRESHOLD:
                    status = 'CANDIDATE'
                else:
                    status = 'zombie'
                probe_results[name] = {
                    'method': 'logistic',
                    'delta_accuracy': delta,
                    'delta_metric': delta,
                    'acc_trained': res['acc_trained'],
                    'acc_untrained': res['acc_untrained'],
                    'chance': res['chance_accuracy'],
                    'status': status,
                }
                print(f"  {name:25s} dAcc={delta:+.4f} [{status}]")
            else:
                res = ridge_delta_r2(h_tr_slice, h_un_slice, t_slice)
                dr2 = res['delta_r2']
                if dr2 > 0.15:
                    status = 'STRONG'
                elif dr2 > DELTA_R2_THRESHOLD:
                    status = 'CANDIDATE'
                else:
                    status = 'zombie'
                probe_results[name] = {
                    'method': 'ridge',
                    'delta_r2': dr2,
                    'delta_metric': dr2,
                    'r2_trained': res['r2_trained'],
                    'r2_untrained': res['r2_untrained'],
                    'status': status,
                }
                print(f"  {name:25s} dR2={dr2:+.4f} [{status}]")

        # ── TASK 6: Ablation with random control ──
        # Only ablate targets that passed probing threshold
        candidates = {}
        for name, res in probe_results.items():
            dm = res.get('delta_metric', 0)
            if name in CATEGORICAL_TARGETS:
                if dm > DELTA_ACC_THRESHOLD:
                    candidates[name] = res
            else:
                if dm > DELTA_R2_THRESHOLD:
                    candidates[name] = res

        ablation_results = {}
        random_control_results = {}

        # Determine valid k values for this hidden_size
        # Skip k values that ablate > 80% of dims
        valid_k = [k for k in K_VALUES if k <= int(hidden_size * 0.8)]
        if not valid_k:
            valid_k = [min(K_VALUES)]  # always test at least smallest k
        print(f"\n  Ablation k values (valid for h={hidden_size}): {valid_k}")

        if candidates:
            # Run random controls
            for k in valid_k:
                ctrl = random_ablation_control(
                    best_model, X_test, Y_test, k, hidden_size)
                random_control_results[f'k={k}'] = ctrl
                print(f"  Random control k={k}: z={ctrl['mean_z']:.2f} "
                      f"+/- {ctrl['std_z']:.2f}")

            # Target-aligned ablation
            for name in candidates:
                target_for_dims = aligned_targets[name][:min(
                    len(aligned_targets[name]), len(h_trained))].astype(float)
                target_abl = {}

                for k in valid_k:
                    # Clamp k to hidden_size (safety)
                    k_eff = min(k, hidden_size)
                    top_dims = identify_top_dims_for_target(
                        h_trained[:len(target_for_dims)], target_for_dims,
                        k_eff)
                    abl = resample_ablation(
                        best_model, X_test, Y_test, top_dims)
                    random_z = random_control_results[f'k={k}']['mean_z']
                    is_mandatory = (abl['z_score'] < ABLATION_Z_THRESHOLD
                                    and abl['z_score'] < random_z - 2)
                    status = 'MANDATORY' if is_mandatory else 'not mandatory'
                    print(f"    {name} k={k}: target z={abl['z_score']:.2f} "
                          f"vs random z={random_z:.2f} [{status}]")
                    target_abl[f'k={k}'] = {
                        'z_score': abl['z_score'],
                        'random_z': random_z,
                        'relative_degradation': abl['relative_degradation'],
                        'mandatory': is_mandatory,
                    }
                ablation_results[name] = target_abl
        else:
            print("  No probe candidates above threshold — skipping ablation.")

        elapsed_h = time.time() - t0_h
        print(f"\n  Time for h={hidden_size}: {elapsed_h:.1f}s")

        sweep_results[hidden_size] = {
            'mean_cc': mean_cc,
            'best_cc': best_cc,
            'std_cc': float(np.std(cc_all)),
            'cc_per_seed': [float(c) for c in cc_all],
            'cc_pass': cc_pass,
            'epochs_per_seed': [r['epochs_trained'] for r in seed_results],
            'probing': {name: {
                'delta_metric': res['delta_metric'],
                'status': res['status'],
                'method': res.get('method', 'unknown'),
            } for name, res in probe_results.items()},
            'probing_full': probe_results,
            'ablation': ablation_results,
            'random_control': random_control_results,
            'elapsed_s': elapsed_h,
        }

    # ══════════════════════════════════════════════════════════════
    # SUMMARY TABLES
    # ══════════════════════════════════════════════════════════════
    elapsed_total = time.time() - t0_total

    print(f"\n{'#'*70}")
    print(f"# BOTTLENECK SWEEP COMPLETE")
    print(f"# Total time: {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
    print(f"{'#'*70}")

    # ── Table 1: hidden_size x probe_target -> delta_metric ──
    print(f"\n{'='*70}")
    print("TABLE 1: hidden_size x probe_target -> delta_metric")
    print(f"{'='*70}")

    # Header
    header = f"{'target':25s}"
    for h in HIDDEN_SIZES:
        header += f" | h={h:>4d}"
    print(header)
    print("-" * len(header))

    for name in probe_target_names:
        row = f"{name:25s}"
        for h in HIDDEN_SIZES:
            dm = sweep_results[h]['probing'][name]['delta_metric']
            st = sweep_results[h]['probing'][name]['status']
            marker = '*' if st == 'STRONG' else ('+' if st == 'CANDIDATE' else ' ')
            row += f" | {dm:+.4f}{marker}"
        print(row)

    print("\n  * = STRONG, + = CANDIDATE")

    # CC row
    cc_row = f"{'MEAN CC':25s}"
    for h in HIDDEN_SIZES:
        cc = sweep_results[h]['mean_cc']
        p = 'PASS' if sweep_results[h]['cc_pass'] else 'FAIL'
        cc_row += f" |  {cc:.4f} "
    print(f"\n{cc_row}")

    # ── Table 2: hidden_size x k x (random vs target) -> z-score ──
    print(f"\n{'='*70}")
    print("TABLE 2: Ablation z-scores (target-aligned vs random)")
    print(f"{'='*70}")

    for k in K_VALUES:
        kstr = f"k={k}"
        print(f"\n--- {kstr} ---")
        header2 = f"{'':25s}"
        for h in HIDDEN_SIZES:
            header2 += f" |   h={h:>4d}   "
        print(header2)
        print("-" * len(header2))

        # Random control row
        rand_row = f"{'RANDOM':25s}"
        for h in HIDDEN_SIZES:
            rc = sweep_results[h]['random_control'].get(f'k={k}', {})
            z = rc.get('mean_z', float('nan'))
            if np.isnan(z):
                rand_row += " |    n/a    "
            else:
                rand_row += f" | {z:+8.1f}  "
        print(rand_row)

        # Per-target rows
        all_abl_targets = set()
        for h in HIDDEN_SIZES:
            all_abl_targets.update(sweep_results[h]['ablation'].keys())

        for name in sorted(all_abl_targets):
            targ_row = f"{name:25s}"
            for h in HIDDEN_SIZES:
                abl_data = sweep_results[h]['ablation'].get(name, {})
                k_data = abl_data.get(f'k={k}', {})
                z = k_data.get('z_score', float('nan'))
                mand = k_data.get('mandatory', False)
                if np.isnan(z):
                    targ_row += " |    ---    "
                else:
                    m = ' M' if mand else '  '
                    targ_row += f" | {z:+8.1f}{m}"
            print(targ_row)

        print("  M = MANDATORY (target z < -2 AND target z < random z - 2)")

    # ── Save report ──
    report = {
        'patient_id': PATIENT_ID,
        'run_type': 'bottleneck_sweep',
        'hidden_sizes': HIDDEN_SIZES,
        'n_seeds': N_SEEDS,
        'k_values': K_VALUES,
        'n_output_focused': n_output,
        'selected_neuron_indices': top_idx.tolist(),
        'probe_target_names': probe_target_names,
        'results': {str(h): {
            'mean_cc': sweep_results[h]['mean_cc'],
            'best_cc': sweep_results[h]['best_cc'],
            'std_cc': sweep_results[h]['std_cc'],
            'cc_per_seed': sweep_results[h]['cc_per_seed'],
            'cc_pass': sweep_results[h]['cc_pass'],
            'epochs_per_seed': sweep_results[h]['epochs_per_seed'],
            'probing': sweep_results[h]['probing'],
            'ablation': sweep_results[h]['ablation'],
            'random_control': sweep_results[h]['random_control'],
            'elapsed_s': sweep_results[h]['elapsed_s'],
        } for h in HIDDEN_SIZES},
        'elapsed_total_s': elapsed_total,
    }

    report_path = OUTPUT_DIR / 'bottleneck_sweep_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved: {report_path}")
    return report


if __name__ == '__main__':
    main()
