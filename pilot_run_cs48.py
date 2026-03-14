"""
pilot_run_cs48.py

Complete DESCARTES pilot run for sub-CS48:
  Task 4: Train LSTM surrogate (hidden=128, 10 seeds, early stopping)
  Task 5: Ridge dR2 probing for all probe targets (trained vs untrained)
  Task 6: Resample ablation for targets with dR2 > 0.05
  Report: model quality (CC per neuron), dR2 for each probe target,
          ablation z-scores, flag any model with CC < 0.3
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from pathlib import Path
import json
import time

# ================================================================
# CONFIGURATION
# ================================================================
PATIENT_ID = 'sub-CS48'
PREPROCESSED_PATH = f'preprocessed_data/{PATIENT_ID}.npz'
PROBE_TARGETS_PATH = f'probe_targets/{PATIENT_ID}.npz'
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
ABLATION_Z_THRESHOLD = -2.0
N_RESAMPLES = 100
K_VALUES = [10, 25, 50]

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
# TASK 4: Training
# ================================================================
def train_one_seed(model, X_train, Y_train, X_val, Y_val, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=PATIENCE//2, factor=0.5)
    criterion = nn.MSELoss()

    X_tr = torch.FloatTensor(X_train).to(DEVICE)
    Y_tr = torch.FloatTensor(Y_train).to(DEVICE)
    X_v = torch.FloatTensor(X_val).to(DEVICE)
    Y_v = torch.FloatTensor(Y_val).to(DEVICE)

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
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
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

    mean_cc = np.nanmean(cc_values)

    return {
        'seed': seed,
        'best_val_loss': best_val_loss,
        'cc_per_neuron': cc_values,
        'mean_cc': mean_cc,
        'epochs_trained': final_epoch,
        'model_state': best_state,
    }


# ================================================================
# TASK 5: Ridge dR2 Probing
# ================================================================
def ridge_delta_r2(h_trained, h_untrained, target, n_folds=5):
    if target.ndim == 1:
        target = target.reshape(-1, 1)

    alphas = np.logspace(-3, 3, 20)
    kf = KFold(n_splits=n_folds, shuffle=False)

    r2_tr_folds = []
    for tr_idx, te_idx in kf.split(h_trained):
        ridge = RidgeCV(alphas=alphas)
        ridge.fit(h_trained[tr_idx], target[tr_idx])
        r2_tr_folds.append(ridge.score(h_trained[te_idx], target[te_idx]))

    r2_un_folds = []
    for tr_idx, te_idx in kf.split(h_untrained):
        ridge = RidgeCV(alphas=alphas)
        ridge.fit(h_untrained[tr_idx], target[tr_idx])
        r2_un_folds.append(ridge.score(h_untrained[te_idx], target[te_idx]))

    r2_trained = np.mean(r2_tr_folds)
    r2_untrained = np.mean(r2_un_folds)

    return {
        'r2_trained': float(r2_trained),
        'r2_untrained': float(r2_untrained),
        'delta_r2': float(r2_trained - r2_untrained),
    }


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
                chunk = np.pad(chunk, (0, window_bins - len(chunk)), mode='edge')
            aligned.append(chunk)
    return np.concatenate(aligned).astype(np.float32)


# ================================================================
# TASK 6: Resample Ablation
# ================================================================
def resample_ablation(model, X_test, Y_test, dims_to_ablate,
                      n_resamples=N_RESAMPLES, device=DEVICE):
    model.train(False)
    X_t = torch.FloatTensor(X_test).to(device)

    with torch.no_grad():
        y_baseline, h_baseline = model(X_t, return_hidden=True)
    baseline_mse = float(np.mean((y_baseline.cpu().numpy() - Y_test) ** 2))

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

        abl_mse = float(np.mean((y_ablated.cpu().numpy() - Y_test) ** 2))
        ablated_mses.append(abl_mse)

    ablated_mses = np.array(ablated_mses)
    mean_abl = float(np.mean(ablated_mses))
    std_abl = float(np.std(ablated_mses))

    z_score = float((baseline_mse - mean_abl) / std_abl) if std_abl > 0 else 0.0
    rel_degrad = float((mean_abl - baseline_mse) / max(baseline_mse, 1e-10))

    return {
        'baseline_mse': baseline_mse,
        'mean_ablated_mse': mean_abl,
        'std_ablated_mse': std_abl,
        'z_score': z_score,
        'relative_degradation': rel_degrad,
        'n_dims_ablated': len(dims_to_ablate),
    }


def identify_top_dims_for_target(h_trained, target, k):
    if target.ndim == 1:
        target = target.reshape(-1, 1)
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 20))
    ridge.fit(h_trained, target)
    coef_mag = np.abs(ridge.coef_).sum(axis=0)
    top_dims = np.argsort(coef_mag)[-k:]
    return top_dims.tolist()


# ================================================================
# MAIN PIPELINE
# ================================================================
def main():
    t0 = time.time()

    print(f"\n{'='*70}")
    print(f"DESCARTES PILOT RUN: {PATIENT_ID}")
    print(f"{'='*70}")

    data = np.load(PREPROCESSED_PATH)
    X_train, Y_train = data['X_train'], data['Y_train']
    X_val, Y_val = data['X_val'], data['Y_val']
    X_test, Y_test = data['X_test'], data['Y_test']
    n_input = int(data['n_input'])
    n_output = int(data['n_output'])
    window_ms = int(data['window_ms'])
    stride_ms = int(data['stride_ms'])
    bin_ms = int(data['bin_ms'])

    print(f"Data: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")
    print(f"Neurons: {n_input} input, {n_output} output")

    probe_data = np.load(PROBE_TARGETS_PATH)
    probe_target_names = list(probe_data.files)
    print(f"Probe targets: {probe_target_names}")

    window_bins = int(window_ms / bin_ms)
    stride_bins = int(stride_ms / bin_ms)
    train_end = X_train.shape[0]
    val_end = train_end + X_val.shape[0]
    test_start_window = val_end
    n_test = X_test.shape[0]

    test_window_start_bins = [(test_start_window + w) * stride_bins
                              for w in range(n_test)]

    # ── TASK 4 ──
    print(f"\n{'='*70}")
    print("TASK 4: Training LSTM Surrogates (10 seeds)")
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

        flag = " *** CC < 0.3 WARNING ***" if result['mean_cc'] < 0.3 else ""
        print(f"  Val loss: {result['best_val_loss']:.6f}, "
              f"Mean CC: {result['mean_cc']:.4f}, "
              f"Epochs: {result['epochs_trained']}{flag}")

    cc_all = [r['mean_cc'] for r in all_seed_results]
    print(f"\n--- Training Summary ---")
    print(f"  Mean CC across seeds: {np.mean(cc_all):.4f} +/- {np.std(cc_all):.4f}")
    print(f"  Min CC: {np.min(cc_all):.4f}, Max CC: {np.max(cc_all):.4f}")
    low_cc = sum(1 for c in cc_all if c < 0.3)
    if low_cc > 0:
        print(f"  WARNING: {low_cc}/{N_SEEDS} seeds have CC < 0.3")

    best_seed_idx = int(np.argmax(cc_all))
    best_model = trained_models[best_seed_idx]
    print(f"  Best seed: {best_seed_idx} (CC={cc_all[best_seed_idx]:.4f})")

    # ── TASK 5 ──
    print(f"\n{'='*70}")
    print("TASK 5: Ridge dR2 Probing")
    print(f"{'='*70}")

    untrained_model = LimbicPrefrontalLSTM(
        n_input=n_input, n_output=n_output,
        hidden_size=HIDDEN_SIZE, n_layers=N_LAYERS, dropout=DROPOUT)
    torch.manual_seed(9999)
    untrained_model.to(DEVICE)
    untrained_model.train(False)

    h_trained = extract_hidden_flat(best_model, X_test)
    h_untrained = extract_hidden_flat(untrained_model, X_test)
    print(f"Hidden states: trained={h_trained.shape}, untrained={h_untrained.shape}")

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
            print(f"  {name}: SKIPPED (constant target)")
            probe_results[name] = {
                'r2_trained': 0.0, 'r2_untrained': 0.0,
                'delta_r2': 0.0, 'status': 'constant'
            }
            continue

        result = ridge_delta_r2(h_tr_use, h_un_use, t_use)
        probe_results[name] = result

        if result['delta_r2'] > 0.15:
            status = "STRONG"
        elif result['delta_r2'] > DELTA_R2_THRESHOLD:
            status = "CANDIDATE"
        else:
            status = "zombie"

        print(f"  {name:<25s} dR2={result['delta_r2']:+.4f}  "
              f"(trained={result['r2_trained']:.4f}, "
              f"untrained={result['r2_untrained']:.4f})  [{status}]")

    # ── TASK 6 ──
    print(f"\n{'='*70}")
    print("TASK 6: Resample Ablation")
    print(f"{'='*70}")

    candidates = {name: res for name, res in probe_results.items()
                  if res['delta_r2'] > DELTA_R2_THRESHOLD}

    if not candidates:
        print("No candidates with dR2 > 0.05 -- skipping ablation.")
        ablation_results = {}
    else:
        print(f"Ablation candidates: {list(candidates.keys())}")
        ablation_results = {}

        for name, probe_res in candidates.items():
            print(f"\n--- Ablating for: {name} (dR2={probe_res['delta_r2']:.4f}) ---")
            target_continuous = probe_data[name]
            target_aligned = align_target_to_windows(
                target_continuous, test_window_start_bins, window_bins, n_test)
            min_len = min(len(target_aligned), len(h_trained))

            target_results = {}
            for k in K_VALUES:
                top_dims = identify_top_dims_for_target(
                    h_trained[:min_len], target_aligned[:min_len], k)

                abl_result = resample_ablation(
                    best_model, X_test, Y_test, top_dims,
                    n_resamples=N_RESAMPLES)

                is_mandatory = abl_result['z_score'] < ABLATION_Z_THRESHOLD
                status = "MANDATORY" if is_mandatory else "not mandatory"
                print(f"  k={k}: z={abl_result['z_score']:.2f}, "
                      f"degrad={abl_result['relative_degradation']:.4f}  [{status}]")

                target_results[f'k={k}'] = {
                    'z_score': abl_result['z_score'],
                    'relative_degradation': abl_result['relative_degradation'],
                    'baseline_mse': abl_result['baseline_mse'],
                    'mean_ablated_mse': abl_result['mean_ablated_mse'],
                    'n_dims': k,
                    'mandatory': is_mandatory,
                }

            ablation_results[name] = target_results

    # ── REPORT ──
    print(f"\n{'='*70}")
    print(f"PILOT RUN REPORT: {PATIENT_ID}")
    print(f"{'='*70}")

    report = {
        'patient_id': PATIENT_ID,
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
            'window_ms': window_ms,
            'stride_ms': stride_ms,
            'bin_ms': bin_ms,
        },
        'task4_training': {
            'cc_per_seed': [float(c) for c in cc_all],
            'mean_cc': float(np.mean(cc_all)),
            'std_cc': float(np.std(cc_all)),
            'best_seed': best_seed_idx,
            'best_cc': float(cc_all[best_seed_idx]),
            'cc_below_0_3': low_cc,
            'epochs_per_seed': [r['epochs_trained'] for r in all_seed_results],
            'val_loss_per_seed': [float(r['best_val_loss']) for r in all_seed_results],
            'cc_per_neuron_best_seed': [float(c) for c in
                                         all_seed_results[best_seed_idx]['cc_per_neuron']],
        },
        'task5_probing': {},
        'task6_ablation': {},
    }

    print("\n--- Model Quality (Task 4) ---")
    print(f"  Mean CC: {np.mean(cc_all):.4f} +/- {np.std(cc_all):.4f}")
    print(f"  CC per seed: {[f'{c:.3f}' for c in cc_all]}")
    cc_neurons = all_seed_results[best_seed_idx]['cc_per_neuron']
    print(f"  CC per neuron (best seed): min={min(cc_neurons):.3f}, "
          f"max={max(cc_neurons):.3f}, "
          f"median={np.median(cc_neurons):.3f}")
    n_low = sum(1 for c in cc_neurons if c < 0.3)
    print(f"  Neurons with CC < 0.3: {n_low}/{n_output}")

    print("\n--- Probing Results (Task 5) ---")
    print(f"  {'Target':<25s} {'dR2':>8s}  {'R2_tr':>8s}  {'R2_un':>8s}  {'Status'}")
    print(f"  {'-'*70}")
    for name in sorted(probe_results.keys()):
        res = probe_results[name]
        dr2 = res['delta_r2']
        if dr2 > 0.15:
            status = "STRONG"
        elif dr2 > DELTA_R2_THRESHOLD:
            status = "CANDIDATE"
        else:
            status = "zombie"
        print(f"  {name:<25s} {dr2:+8.4f}  {res['r2_trained']:8.4f}  "
              f"{res['r2_untrained']:8.4f}  {status}")

        report['task5_probing'][name] = {
            'delta_r2': float(dr2),
            'r2_trained': float(res['r2_trained']),
            'r2_untrained': float(res['r2_untrained']),
            'status': status,
        }

    print("\n--- Ablation Results (Task 6) ---")
    if ablation_results:
        for name, k_results in ablation_results.items():
            print(f"  {name}:")
            for k_label, k_res in k_results.items():
                mand = "MANDATORY" if k_res['mandatory'] else "not mandatory"
                print(f"    {k_label}: z={k_res['z_score']:.2f}, "
                      f"degrad={k_res['relative_degradation']:.4f}  [{mand}]")
            report['task6_ablation'][name] = k_results
    else:
        print("  No targets exceeded dR2 threshold for ablation.")

    print("\n--- Quality Flags ---")
    if low_cc > 0:
        print(f"  FLAG: {low_cc} seeds have mean CC < 0.3 (model may be unreliable)")
    else:
        print(f"  OK: All {N_SEEDS} seeds have CC >= 0.3")

    if any(c < 0.3 for c in cc_neurons):
        print(f"  FLAG: {n_low}/{n_output} output neurons have CC < 0.3 in best model")

    elapsed = time.time() - t0
    report['elapsed_seconds'] = elapsed
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    report_path = OUTPUT_DIR / 'pilot_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {report_path}")

    return report


if __name__ == '__main__':
    main()
