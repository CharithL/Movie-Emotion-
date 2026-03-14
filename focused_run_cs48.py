"""
focused_run_cs48.py  (v2 — with methodological fixes)

Re-run Tasks 4-6 predicting ONLY the top-10 output neurons
(CC > 0.5 from the pilot run).

v2 fixes:
  1. emotion_category uses logistic regression dAccuracy, not Ridge R2
  2. valence dropped (= -1*arousal, zero independent info);
     continuous targets clipped to +/-3 std before probing
  3. Random-dimension ablation control at every k value
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

HIDDEN_SIZE = 128
N_LAYERS = 2
N_SEEDS = 10
MAX_EPOCHS = 200
BATCH_SIZE = 32
LR = 1e-3
PATIENCE = 20
DROPOUT = 0.1

DELTA_R2_THRESHOLD = 0.05
DELTA_ACC_THRESHOLD = 0.02   # for categorical targets
ABLATION_Z_THRESHOLD = -2.0
N_RESAMPLES = 100
K_VALUES = [10, 25, 50]
N_RANDOM_ABLATION_REPEATS = 5  # repeat random control for stability

TOP_N_NEURONS = 10
CC_THRESHOLD = 0.5

# Targets that are categorical (use logistic regression)
CATEGORICAL_TARGETS = {'emotion_category'}
# Targets to skip entirely (redundant / methodologically invalid)
SKIP_TARGETS = {'valence'}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")


# ================================================================
# MODEL (identical to pilot)
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
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, n_output)
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
def train_one_seed(model, X_train, Y_train, X_val, Y_val, seed,
                   device=DEVICE):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=PATIENCE//2, factor=0.5)
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
            idx = perm[b*BATCH_SIZE:(b+1)*BATCH_SIZE]
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
        'model_state': best_state,
    }


# ================================================================
# PROBING — continuous targets (Ridge R2)
# ================================================================
def ridge_delta_r2(h_trained, h_untrained, target, n_folds=5):
    """Standard Ridge dR2 for continuous targets."""
    if target.ndim == 1:
        target = target.reshape(-1, 1)
    alphas = np.logspace(-3, 3, 20)
    kf = KFold(n_splits=n_folds, shuffle=False)

    r2_tr = [RidgeCV(alphas=alphas).fit(h_trained[tr], target[tr]).score(
        h_trained[te], target[te]) for tr, te in kf.split(h_trained)]
    r2_un = [RidgeCV(alphas=alphas).fit(h_untrained[tr], target[tr]).score(
        h_untrained[te], target[te]) for tr, te in kf.split(h_untrained)]

    r2_trained = float(np.mean(r2_tr))
    r2_untrained = float(np.mean(r2_un))
    return {
        'r2_trained': r2_trained,
        'r2_untrained': r2_untrained,
        'delta_r2': float(r2_trained - r2_untrained),
    }


# ================================================================
# PROBING — categorical targets (Logistic Regression accuracy)
# ================================================================
def logistic_delta_accuracy(h_trained, h_untrained, labels, n_folds=5):
    """Logistic regression dAccuracy for categorical targets.

    Uses StratifiedKFold to preserve class proportions in each fold.
    Reports accuracy for trained vs untrained, majority-class baseline,
    and per-class accuracy breakdown.
    """
    unique_classes = np.unique(labels)
    n_classes = len(unique_classes)

    # Majority-class baseline
    class_counts = {int(c): int((labels == c).sum()) for c in unique_classes}
    majority_class = max(class_counts, key=class_counts.get)
    chance_accuracy = class_counts[majority_class] / len(labels)

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=False)

    scaler_tr = StandardScaler()
    scaler_un = StandardScaler()

    acc_trained_folds = []
    acc_untrained_folds = []
    per_class_acc_trained = {int(c): [] for c in unique_classes}

    for tr_idx, te_idx in skf.split(h_trained, labels):
        # Trained model hidden states
        X_tr_s = scaler_tr.fit_transform(h_trained[tr_idx])
        X_te_s = scaler_tr.transform(h_trained[te_idx])
        clf = LogisticRegressionCV(
            Cs=10, cv=3, max_iter=1000,
            class_weight='balanced', random_state=42)
        clf.fit(X_tr_s, labels[tr_idx])
        preds_tr = clf.predict(X_te_s)
        acc_trained_folds.append(float(np.mean(preds_tr == labels[te_idx])))

        # Per-class accuracy
        for c in unique_classes:
            mask_c = labels[te_idx] == c
            if mask_c.sum() > 0:
                per_class_acc_trained[int(c)].append(
                    float(np.mean(preds_tr[mask_c] == c)))

        # Untrained model hidden states
        X_tr_u = scaler_un.fit_transform(h_untrained[tr_idx])
        X_te_u = scaler_un.transform(h_untrained[te_idx])
        clf_un = LogisticRegressionCV(
            Cs=10, cv=3, max_iter=1000,
            class_weight='balanced', random_state=42)
        clf_un.fit(X_tr_u, labels[tr_idx])
        preds_un = clf_un.predict(X_te_u)
        acc_untrained_folds.append(
            float(np.mean(preds_un == labels[te_idx])))

    acc_trained = float(np.mean(acc_trained_folds))
    acc_untrained = float(np.mean(acc_untrained_folds))
    delta_acc = acc_trained - acc_untrained

    per_class_mean = {c: float(np.mean(v)) if v else 0.0
                      for c, v in per_class_acc_trained.items()}

    return {
        'acc_trained': acc_trained,
        'acc_untrained': acc_untrained,
        'delta_accuracy': delta_acc,
        'chance_accuracy': chance_accuracy,
        'class_distribution': class_counts,
        'per_class_accuracy_trained': per_class_mean,
        'n_classes': n_classes,
    }


# ================================================================
# HIDDEN STATE EXTRACTION
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
    """Clip continuous target to +/-3 standard deviations."""
    mu = np.mean(target)
    std = np.std(target)
    if std < 1e-10:
        return target
    lo = mu - 3 * std
    hi = mu + 3 * std
    n_clipped = int(((target < lo) | (target > hi)).sum())
    clipped = np.clip(target, lo, hi)
    return clipped, n_clipped


# ================================================================
# ABLATION (with random control)
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
    z_score = (float((baseline_mse - mean_abl) / std_abl)
               if std_abl > 0 else 0.0)

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
    """Ablate k RANDOM dimensions (not aligned to any target).

    Repeats n_repeats times with different random dim selections,
    returns mean z-score and individual z-scores.
    """
    z_scores = []
    for rep in range(n_repeats):
        random_dims = np.random.choice(hidden_size, size=k,
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
# MAIN
# ================================================================
def main():
    t0 = time.time()

    print(f"\n{'='*70}")
    print(f"FOCUSED RUN v2: {PATIENT_ID} (top-{TOP_N_NEURONS} neurons)")
    print(f"  Fixes: logistic for categorical, valence dropped,")
    print(f"         outlier clipping, random ablation control")
    print(f"{'='*70}")

    # ── Load pilot report to get per-neuron CCs ──
    with open(PILOT_REPORT_PATH) as f:
        pilot = json.load(f)
    cc_per_neuron = np.array(pilot['task4_training']['cc_per_neuron_best_seed'])

    # Select top neurons by CC
    sorted_idx = np.argsort(cc_per_neuron)[::-1]
    top_idx = sorted_idx[:TOP_N_NEURONS]
    top_ccs = cc_per_neuron[top_idx]

    print(f"\nSelected {TOP_N_NEURONS} output neurons with highest CC:")
    for rank, (idx, cc) in enumerate(zip(top_idx, top_ccs)):
        above = "above" if cc >= CC_THRESHOLD else "BELOW"
        print(f"  #{rank}: neuron {idx}, pilot CC = {cc:.4f} ({above} {CC_THRESHOLD})")

    # ── Load and subset preprocessed data ──
    data = np.load(PREPROCESSED_PATH)
    X_train, Y_train_full = data['X_train'], data['Y_train']
    X_val, Y_val_full = data['X_val'], data['Y_val']
    X_test, Y_test_full = data['X_test'], data['Y_test']
    n_input = int(data['n_input'])
    window_ms = int(data['window_ms'])
    stride_ms = int(data['stride_ms'])
    bin_ms = int(data['bin_ms'])

    Y_train = Y_train_full[:, :, top_idx]
    Y_val = Y_val_full[:, :, top_idx]
    Y_test = Y_test_full[:, :, top_idx]
    n_output = TOP_N_NEURONS

    print(f"\nData: X={X_train.shape}, Y_focused={Y_train.shape}")
    print(f"Neurons: {n_input} input -> {n_output} output (focused)")

    # ── Load probe targets ──
    probe_data = np.load(PROBE_TARGETS_PATH)
    all_target_names = list(probe_data.files)

    # Filter out skipped targets
    probe_target_names = [n for n in all_target_names
                          if n not in SKIP_TARGETS]
    print(f"Probe targets ({len(probe_target_names)}): {probe_target_names}")
    if SKIP_TARGETS:
        print(f"  DROPPED: {SKIP_TARGETS} (redundant with arousal)")

    window_bins = int(window_ms / bin_ms)
    stride_bins = int(stride_ms / bin_ms)
    train_end = X_train.shape[0]
    val_end = train_end + X_val.shape[0]
    test_start_window = val_end
    n_test = X_test.shape[0]
    test_window_start_bins = [(test_start_window + w) * stride_bins
                              for w in range(n_test)]

    # ── Print probe target distribution stats ──
    print(f"\n--- Probe Target Distributions ---")
    for name in probe_target_names:
        d = probe_data[name].astype(float)
        mu, std = d.mean(), d.std()
        pcts = np.percentile(d, [1, 25, 50, 75, 99])
        if name in CATEGORICAL_TARGETS:
            cats = np.unique(d)
            dist = {int(c): int((d == c).sum()) for c in cats}
            pcts_str = ", ".join(f"{c}:{100*n/len(d):.1f}%"
                                 for c, n in dist.items())
            print(f"  {name:25s} CATEGORICAL  classes={pcts_str}")
        else:
            n_outliers = int(((d - mu) / max(std, 1e-10) > 3).sum() +
                             ((d - mu) / max(std, 1e-10) < -3).sum())
            print(f"  {name:25s} mean={mu:+.4f} std={std:.4f} "
                  f"[{d.min():+.4f}, {d.max():+.4f}] "
                  f"p1/p99=[{pcts[0]:+.4f},{pcts[4]:+.4f}] "
                  f"outliers(>3s)={n_outliers}")

    # ══════════════════════════════════════════════════════════════
    # TASK 4: Train LSTM on focused output
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"TASK 4: Training LSTM ({n_input} -> {n_output} focused neurons)")
    print(f"{'='*70}")

    all_seed_results = []
    trained_models = []

    for seed in range(N_SEEDS):
        print(f"\n--- Seed {seed} ---")
        model = LimbicPrefrontalLSTM(
            n_input=n_input, n_output=n_output,
            hidden_size=HIDDEN_SIZE, n_layers=N_LAYERS, dropout=DROPOUT)
        result = train_one_seed(model, X_train, Y_train, X_val, Y_val, seed)
        all_seed_results.append(result)
        trained_models.append(model)

        flag = " *** CC < 0.3 ***" if result['mean_cc'] < 0.3 else " OK"
        print(f"  Val loss: {result['best_val_loss']:.6f}, "
              f"Mean CC: {result['mean_cc']:.4f}, "
              f"Epochs: {result['epochs_trained']}{flag}")

    cc_all = [r['mean_cc'] for r in all_seed_results]
    best_seed_idx = int(np.argmax(cc_all))
    best_model = trained_models[best_seed_idx]

    print(f"\n--- Focused Training Summary ---")
    print(f"  Mean CC: {np.mean(cc_all):.4f} +/- {np.std(cc_all):.4f}")
    print(f"  Best seed: {best_seed_idx} (CC={cc_all[best_seed_idx]:.4f})")
    low_cc = sum(1 for c in cc_all if c < 0.3)
    if low_cc > 0:
        print(f"  WARNING: {low_cc}/{N_SEEDS} seeds still have CC < 0.3")
    else:
        print(f"  ALL {N_SEEDS} seeds have CC >= 0.3")

    cc_neurons = all_seed_results[best_seed_idx]['cc_per_neuron']
    print(f"\n  Per-neuron CC (best seed):")
    for rank, (nidx, cc_pilot, cc_focused) in enumerate(
            zip(top_idx, top_ccs, cc_neurons)):
        print(f"    neuron {nidx}: pilot={cc_pilot:.4f} -> "
              f"focused={cc_focused:.4f}")

    # ══════════════════════════════════════════════════════════════
    # TASK 5: Probing (method-appropriate per target type)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("TASK 5: Probing (Ridge R2 for continuous, Logistic Acc for categorical)")
    print(f"{'='*70}")

    untrained_model = LimbicPrefrontalLSTM(
        n_input=n_input, n_output=n_output,
        hidden_size=HIDDEN_SIZE, n_layers=N_LAYERS, dropout=DROPOUT)
    torch.manual_seed(9999)
    untrained_model.to(DEVICE)
    untrained_model.train(False)

    h_trained = extract_hidden_flat(best_model, X_test)
    h_untrained = extract_hidden_flat(untrained_model, X_test)
    print(f"Hidden: trained={h_trained.shape}, untrained={h_untrained.shape}")

    probe_results = {}
    for name in probe_target_names:
        target_continuous = probe_data[name]
        target_aligned = align_target_to_windows(
            target_continuous, test_window_start_bins, window_bins, n_test)

        min_len = min(len(target_aligned), len(h_trained))
        t_use = target_aligned[:min_len]
        h_tr_use = h_trained[:min_len]
        h_un_use = h_untrained[:min_len]

        if np.std(t_use) < 1e-8:
            probe_results[name] = {
                'delta_metric': 0.0, 'status': 'constant',
                'method': 'skipped'}
            print(f"  {name}: SKIPPED (constant)")
            continue

        # ── Categorical target: logistic regression ──
        if name in CATEGORICAL_TARGETS:
            labels = t_use.astype(int)
            unique_classes, class_counts = np.unique(labels,
                                                      return_counts=True)
            class_dist = {int(c): int(n) for c, n
                          in zip(unique_classes, class_counts)}
            majority_pct = 100 * max(class_counts) / len(labels)

            print(f"\n  {name} (CATEGORICAL):")
            print(f"    Class distribution in test: {class_dist}")
            print(f"    Majority class: {majority_pct:.1f}%")

            result = logistic_delta_accuracy(h_tr_use, h_un_use, labels)
            delta = result['delta_accuracy']

            if delta > 0.10:
                status = "STRONG"
            elif delta > DELTA_ACC_THRESHOLD:
                status = "CANDIDATE"
            else:
                status = "zombie"

            print(f"    Acc(trained)={result['acc_trained']:.4f}, "
                  f"Acc(untrained)={result['acc_untrained']:.4f}, "
                  f"chance={result['chance_accuracy']:.4f}")
            print(f"    dAccuracy={delta:+.4f}  [{status}]")
            print(f"    Per-class acc (trained): "
                  f"{result['per_class_accuracy_trained']}")

            probe_results[name] = {
                'method': 'logistic_accuracy',
                'acc_trained': result['acc_trained'],
                'acc_untrained': result['acc_untrained'],
                'delta_accuracy': delta,
                'delta_metric': delta,  # unified key for thresholding
                'chance_accuracy': result['chance_accuracy'],
                'class_distribution': result['class_distribution'],
                'per_class_accuracy': result['per_class_accuracy_trained'],
                'status': status,
            }

        # ── Continuous target: Ridge R2 with outlier clipping ──
        else:
            # Clip to +/-3 std
            t_clipped, n_clipped = clip_to_3sigma(t_use)
            if n_clipped > 0:
                print(f"  {name}: clipped {n_clipped} outliers "
                      f"({100*n_clipped/len(t_use):.1f}%) to +/-3 std")
            t_use = t_clipped

            result = ridge_delta_r2(h_tr_use, h_un_use, t_use)
            dr2 = result['delta_r2']

            if dr2 > 0.15:
                status = "STRONG"
            elif dr2 > DELTA_R2_THRESHOLD:
                status = "CANDIDATE"
            else:
                status = "zombie"

            result['status'] = status
            result['method'] = 'ridge_r2'
            result['delta_metric'] = dr2  # unified key
            result['n_outliers_clipped'] = n_clipped
            probe_results[name] = result

            print(f"  {name:<25s} dR2={dr2:+.4f}  "
                  f"(tr={result['r2_trained']:.4f}, "
                  f"un={result['r2_untrained']:.4f})  [{status}]")

    # ══════════════════════════════════════════════════════════════
    # TASK 6: Ablation with RANDOM CONTROL
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("TASK 6: Resample Ablation (with random-dimension control)")
    print(f"{'='*70}")

    # Determine candidates using unified threshold
    candidates = {}
    for name, res in probe_results.items():
        if name in CATEGORICAL_TARGETS:
            if res.get('delta_accuracy', 0) > DELTA_ACC_THRESHOLD:
                candidates[name] = res
        else:
            if res.get('delta_r2', 0) > DELTA_R2_THRESHOLD:
                candidates[name] = res

    ablation_results = {}
    random_control_results = {}

    if not candidates:
        print("No candidates above threshold -- skipping ablation.")
    else:
        # ── First: run random control for each k ──
        print(f"\n--- Random Dimension Control ---")
        print(f"  (Ablating k random dims, {N_RANDOM_ABLATION_REPEATS} "
              f"repeats each, {N_RESAMPLES} resamples)")
        for k in K_VALUES:
            ctrl = random_ablation_control(
                best_model, X_test, Y_test, k, HIDDEN_SIZE,
                n_repeats=N_RANDOM_ABLATION_REPEATS,
                n_resamples=N_RESAMPLES)
            random_control_results[f'k={k}'] = ctrl
            print(f"  k={k}: random z = {ctrl['mean_z']:.2f} "
                  f"+/- {ctrl['std_z']:.2f}  "
                  f"(individual: {[f'{z:.1f}' for z in ctrl['individual_z']]})")

        # ── Then: target-aligned ablation for each candidate ──
        print(f"\nAblation candidates: {list(candidates.keys())}")
        for name, probe_res in candidates.items():
            metric_val = probe_res.get('delta_metric', 0)
            print(f"\n--- Ablating: {name} "
                  f"(delta={metric_val:+.4f}) ---")

            target_continuous = probe_data[name]
            target_aligned = align_target_to_windows(
                target_continuous, test_window_start_bins,
                window_bins, n_test)
            min_len = min(len(target_aligned), len(h_trained))

            # For categorical targets, use the continuous version
            # of the label for dim selection (Ridge on raw labels)
            target_for_dims = target_aligned[:min_len]

            target_results = {}
            for k in K_VALUES:
                top_dims = identify_top_dims_for_target(
                    h_trained[:min_len], target_for_dims, k)
                abl_result = resample_ablation(
                    best_model, X_test, Y_test, top_dims,
                    n_resamples=N_RESAMPLES)
                random_z = random_control_results[f'k={k}']['mean_z']
                is_mandatory = (abl_result['z_score'] < ABLATION_Z_THRESHOLD
                                and abl_result['z_score'] < random_z - 2)
                status = "MANDATORY" if is_mandatory else "not mandatory"
                print(f"  k={k}: target z={abl_result['z_score']:.2f} "
                      f"vs random z={random_z:.2f}  [{status}]")
                target_results[f'k={k}'] = {
                    'z_score': abl_result['z_score'],
                    'random_control_z': random_z,
                    'relative_degradation': abl_result['relative_degradation'],
                    'baseline_mse': abl_result['baseline_mse'],
                    'mean_ablated_mse': abl_result['mean_ablated_mse'],
                    'n_dims': k,
                    'mandatory': is_mandatory,
                }
            ablation_results[name] = target_results

    # ══════════════════════════════════════════════════════════════
    # REPORT
    # ══════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    mean_cc = float(np.mean(cc_all))
    best_cc = float(cc_all[best_seed_idx])

    report = {
        'patient_id': PATIENT_ID,
        'run_type': 'focused_v2',
        'version_notes': [
            'emotion_category uses logistic regression accuracy (not Ridge R2)',
            'valence dropped (= -1*arousal, redundant)',
            'continuous targets clipped to +/-3 std before probing',
            'random dimension ablation control at each k value',
        ],
        'n_output_focused': n_output,
        'selected_neuron_indices': top_idx.tolist(),
        'pilot_cc_of_selected': top_ccs.tolist(),
        'config': {
            'hidden_size': HIDDEN_SIZE,
            'n_layers': N_LAYERS,
            'n_seeds': N_SEEDS,
            'max_epochs': MAX_EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LR,
            'patience': PATIENCE,
            'device': DEVICE,
        },
        'data': {
            'n_input': n_input,
            'n_output': n_output,
            'n_train': int(X_train.shape[0]),
            'n_val': int(X_val.shape[0]),
            'n_test': int(X_test.shape[0]),
        },
        'task4_training': {
            'cc_per_seed': [float(c) for c in cc_all],
            'mean_cc': mean_cc,
            'std_cc': float(np.std(cc_all)),
            'best_seed': best_seed_idx,
            'best_cc': best_cc,
            'cc_below_0_3': low_cc,
            'epochs_per_seed': [r['epochs_trained']
                                for r in all_seed_results],
            'val_loss_per_seed': [float(r['best_val_loss'])
                                  for r in all_seed_results],
            'cc_per_neuron_best_seed': [float(c) for c in cc_neurons],
        },
        'task5_probing': {name: res for name, res in probe_results.items()},
        'task6_ablation': ablation_results,
        'task6_random_control': random_control_results,
        'elapsed_seconds': elapsed,
        'cc_above_0_3': mean_cc >= 0.3,
    }

    report_path = OUTPUT_DIR / 'focused_v2_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    # ── Final summary ──
    print(f"\n{'='*70}")
    print(f"FOCUSED RUN v2 COMPLETE: {PATIENT_ID}")
    print(f"{'='*70}")
    print(f"  Mean CC: {mean_cc:.4f} (threshold: 0.3) -- "
          f"{'PASS' if mean_cc >= 0.3 else 'FAIL'}")
    print(f"\n  Probing Summary:")
    for name, res in probe_results.items():
        method = res.get('method', 'unknown')
        if method == 'logistic_accuracy':
            print(f"    {name:<25s} dAcc={res['delta_accuracy']:+.4f}  "
                  f"[{res['status']}]  (chance={res['chance_accuracy']:.3f})")
        elif method == 'ridge_r2':
            print(f"    {name:<25s} dR2={res['delta_r2']:+.4f}  "
                  f"[{res['status']}]")
        else:
            print(f"    {name:<25s} [{res['status']}]")

    if ablation_results:
        print(f"\n  Ablation Summary (target z vs random z):")
        for name, k_res in ablation_results.items():
            for k_label, r in k_res.items():
                mand = "MANDATORY" if r['mandatory'] else "NOT mandatory"
                print(f"    {name} {k_label}: "
                      f"target z={r['z_score']:.1f}, "
                      f"random z={r['random_control_z']:.1f}  [{mand}]")

    print(f"\n  Time: {elapsed:.1f}s")
    print(f"  Report: {report_path}")

    return report


if __name__ == '__main__':
    main()
