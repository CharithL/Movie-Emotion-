"""
phases2_to_6.py — DESCARTES Statistical Hardening Phases 2-6

Phase 2: LSTM Gate Probing + Temporal Generalization Matrix
Phase 3: Causal Inference (DAS + Conditional MI)
Phase 4: Hierarchical FDR Multiple Comparison Correction
Phase 5: Cross-Domain Comparison (CC-normalized delta-R-squared, Transfer Probing)
Phase 6: Geometric Comparison (Persistent Homology)

Builds on phase1_foundation.py infrastructure.
Run from the "movie emotion" directory.
"""
import numpy as np
import torch
import torch.nn as nn
import json
import time
import sys
from pathlib import Path
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.special import digamma
from scipy.stats import combine_pvalues
from sklearn.neighbors import NearestNeighbors

# Import Phase 1 infrastructure
from phase1_foundation import (
    LimbicPrefrontalLSTM, set_inference_mode, extract_hidden_flat,
    train_model, align_target_to_windows, to_window_level,
    shuffled_cv_r2, temporal_cv_r2,
    BASE_DIR, PREPROCESSED_DIR, PROBE_TARGETS_DIR, RESULTS_DIR,
    OUTPUT_DIR, HIDDEN_SIZE, N_LAYERS, DROPOUT, N_SEEDS,
    MAX_EPOCHS, BATCH_SIZE, LR, PATIENCE,
    BIN_MS, WINDOW_MS, STRIDE_MS, DEVICE,
    SKIP_TARGETS, CATEGORICAL_TARGETS,
)

# ================================================================
# PHASE 2: LSTM-Specific Probing
# ================================================================

class LSTMWithGateAccess(nn.Module):
    """
    Manual LSTM forward pass that exposes all gate activations.

    Standard PyTorch LSTM doesn't expose gates. We manually decompose
    using the weight matrices to compute i_t, f_t, g_t, o_t, c_t, h_t.

    For a 2-layer LSTM, we extract gates from the LAST layer only
    (that's the layer whose hidden state feeds into output_proj).
    """

    def __init__(self, original_model):
        super().__init__()
        self.model = original_model

    def forward_with_gates(self, x):
        """
        Manual forward pass recording all gate activations.

        Returns:
            components: dict of (batch, time, hidden_size) tensors
        """
        batch, time_steps, n_input = x.shape
        hidden_size = self.model.hidden_size
        n_layers = self.model.lstm.num_layers
        device = x.device

        # Project input
        projected = self.model.input_proj(x)  # (batch, time, hidden)

        # Extract weight matrices for all layers
        weights = {}
        for layer in range(n_layers):
            weights[layer] = {
                'w_ih': getattr(self.model.lstm, f'weight_ih_l{layer}'),
                'w_hh': getattr(self.model.lstm, f'weight_hh_l{layer}'),
                'b_ih': getattr(self.model.lstm, f'bias_ih_l{layer}'),
                'b_hh': getattr(self.model.lstm, f'bias_hh_l{layer}'),
            }

        # Initialize hidden and cell states per layer
        h = [torch.zeros(batch, hidden_size, device=device) for _ in range(n_layers)]
        c = [torch.zeros(batch, hidden_size, device=device) for _ in range(n_layers)]

        # Storage for last-layer gate activations
        all_h, all_c, all_i, all_f, all_g, all_o = [], [], [], [], [], []

        for t in range(time_steps):
            layer_input = projected[:, t, :]  # (batch, hidden)

            for layer in range(n_layers):
                w = weights[layer]
                # Compute gates manually
                gates = (layer_input @ w['w_ih'].t() + w['b_ih'] +
                         h[layer] @ w['w_hh'].t() + w['b_hh'])

                i_gate = torch.sigmoid(gates[:, :hidden_size])
                f_gate = torch.sigmoid(gates[:, hidden_size:2*hidden_size])
                g_gate = torch.tanh(gates[:, 2*hidden_size:3*hidden_size])
                o_gate = torch.sigmoid(gates[:, 3*hidden_size:])

                c[layer] = f_gate * c[layer] + i_gate * g_gate
                h[layer] = o_gate * torch.tanh(c[layer])

                # Input to next layer is this layer's hidden output
                layer_input = h[layer]

                # Record last layer gates
                if layer == n_layers - 1:
                    all_h.append(h[layer].clone())
                    all_c.append(c[layer].clone())
                    all_i.append(i_gate.clone())
                    all_f.append(f_gate.clone())
                    all_g.append(g_gate.clone())
                    all_o.append(o_gate.clone())

        components = {
            'hidden':      torch.stack(all_h, dim=1),
            'cell':        torch.stack(all_c, dim=1),
            'input_gate':  torch.stack(all_i, dim=1),
            'forget_gate': torch.stack(all_f, dim=1),
            'candidate':   torch.stack(all_g, dim=1),
            'output_gate': torch.stack(all_o, dim=1),
        }

        return components

    def extract_all_components(self, X_test):
        """Extract all gate activations, flattened for probing."""
        self.model.to(DEVICE)
        self.model.train(False)  # AppControl DLL workaround
        with torch.no_grad():
            X_t = torch.FloatTensor(X_test).to(DEVICE)
            components = self.forward_with_gates(X_t)

        result = {}
        for name, tensor in components.items():
            result[name] = tensor.cpu().numpy().reshape(-1, tensor.shape[-1])

        return result


def probe_all_components(trained_components, untrained_components,
                          target, n_windows, bins_per_window):
    """
    Probe each LSTM component (h, c, i, f, g, o) at window level.

    Key interpretation:
    - High delta-R-squared in cell but low in hidden = zombie mechanism
    - High delta-R-squared in forget_gate = active maintenance
    - High delta-R-squared in output_gate = selective gating
    """
    results = {}

    for name in trained_components:
        h_tr = trained_components[name]
        h_un = untrained_components.get(name)
        if h_un is None:
            continue

        min_len = min(len(h_tr), len(h_un), len(target))
        h_tr_w, t_w = to_window_level(h_tr[:min_len], target[:min_len],
                                        n_windows, bins_per_window)
        h_un_w, _ = to_window_level(h_un[:min_len], target[:min_len],
                                      n_windows, bins_per_window)

        # Z-score target
        t = t_w.astype(np.float64)
        t_std = np.std(t)
        if t_std < 1e-8:
            results[name] = {'delta_r2': 0.0, 'r2_trained': 0.0,
                              'r2_untrained': 0.0, 'valid': False}
            continue
        t = (t - np.mean(t)) / t_std
        t = np.clip(t, -3.0, 3.0)

        r2_tr, _ = shuffled_cv_r2(h_tr_w, t)
        r2_un, _ = shuffled_cv_r2(h_un_w, t)
        delta = r2_tr - r2_un

        results[name] = {
            'delta_r2': float(delta),
            'r2_trained': float(r2_tr),
            'r2_untrained': float(r2_un),
            'valid': True,
        }

    return results


def temporal_generalization_matrix(hidden_flat, target_flat,
                                     n_windows, bins_per_window,
                                     alpha=1.0):
    """
    King & Dehaene (2014) temporal generalization matrix.

    Train probe at time t within window, test at time t'.
    TxT matrix reveals temporal dynamics of representation.

    - Strong diagonal: transient, re-encoded at each step
    - Off-diagonal square: sustained attractor-like
    - Below diagonal: builds up over time
    """
    hidden_size = hidden_flat.shape[1]
    N = n_windows * bins_per_window

    H = hidden_flat[:N].reshape(n_windows, bins_per_window, hidden_size)
    T_raw = target_flat[:N].reshape(n_windows, bins_per_window)

    # Use window-mean target (same label for all timesteps within window)
    t_mean = T_raw.mean(axis=1)  # (n_windows,)
    t_std = np.std(t_mean)
    if t_std < 1e-8:
        return np.zeros((bins_per_window, bins_per_window)), []
    t_norm = (t_mean - np.mean(t_mean)) / t_std
    t_norm = np.clip(t_norm, -3.0, 3.0)

    # Broadcast to all timesteps: (n_windows, bins_per_window)
    T = np.tile(t_norm[:, None], (1, bins_per_window))

    # Temporal split of WINDOWS (not bins)
    n_train = int(0.7 * n_windows)
    H_train, H_test = H[:n_train], H[n_train:]
    T_train, T_test = T[:n_train], T[n_train:]

    # Subsample time bins for speed (100 bins -> every 5th = 20 bins)
    step = max(1, bins_per_window // 20)
    time_indices = list(range(0, bins_per_window, step))
    n_t = len(time_indices)

    gen_matrix = np.zeros((n_t, n_t))

    for i, t_train in enumerate(time_indices):
        X_tr = H_train[:, t_train, :]
        y_tr = T_train[:, t_train]

        ridge = Ridge(alpha=alpha)
        ridge.fit(X_tr, y_tr)

        for j, t_test in enumerate(time_indices):
            X_te = H_test[:, t_test, :]
            y_te = T_test[:, t_test]
            r2 = ridge.score(X_te, y_te)
            gen_matrix[i, j] = np.clip(r2, -1, 1)

    return gen_matrix, time_indices


def run_phase2(patient_id, X_test, n_input, n_focused,
               X_train, Y_train_f, X_val, Y_val_f,
               targets, n_windows, bins_per_window):
    """Phase 2: LSTM Gate Probing + Temporal Generalization."""
    print("\n" + "=" * 70)
    print("PHASE 2: LSTM-Specific Probing")
    print("=" * 70)
    t0 = time.time()

    best_seed = 0

    # Train best-seed model
    print(f"\n  Training seed {best_seed} model...")
    torch.manual_seed(best_seed)
    np.random.seed(best_seed)
    model_trained = LimbicPrefrontalLSTM(n_input, n_focused, HIDDEN_SIZE,
                                           N_LAYERS, DROPOUT)
    model_trained = train_model(model_trained, X_train, Y_train_f,
                                  X_val, Y_val_f, best_seed)

    # Create matched untrained
    torch.manual_seed(best_seed)
    np.random.seed(best_seed)
    model_untrained = LimbicPrefrontalLSTM(n_input, n_focused, HIDDEN_SIZE,
                                              N_LAYERS, DROPOUT)

    # Extract gate activations
    print("  Extracting gate activations (trained)...")
    gate_wrapper_trained = LSTMWithGateAccess(model_trained)
    trained_components = gate_wrapper_trained.extract_all_components(X_test)

    print("  Extracting gate activations (untrained)...")
    gate_wrapper_untrained = LSTMWithGateAccess(model_untrained)
    untrained_components = gate_wrapper_untrained.extract_all_components(X_test)

    for name, arr in trained_components.items():
        print(f"    {name}: {arr.shape}")

    # 2.1 Gate-specific probing for ALL targets
    print("\n" + "-" * 60)
    print("2.1 GATE-SPECIFIC PROBING")
    print("-" * 60)

    gate_results = {}
    for tname, target in targets.items():
        print(f"\n  Target: {tname}")
        res = probe_all_components(trained_components, untrained_components,
                                     target, n_windows, bins_per_window)
        gate_results[tname] = {}
        for comp_name, comp_res in res.items():
            gate_results[tname][comp_name] = comp_res
            marker = ""
            if comp_name == 'cell' and comp_res['delta_r2'] > 0.05:
                h_delta = res.get('hidden', {}).get('delta_r2', 0)
                if h_delta < 0.03:
                    marker = " <-- ZOMBIE (in cell, not hidden)"
            if comp_name == 'forget_gate' and comp_res['delta_r2'] > 0.05:
                marker = " <-- ACTIVE MAINTENANCE"
            print(f"    {comp_name:15s}: dR2 = {comp_res['delta_r2']:+.4f}{marker}")

    # 2.2 Temporal Generalization Matrix for all targets
    print("\n" + "-" * 60)
    print("2.2 TEMPORAL GENERALIZATION MATRIX")
    print("-" * 60)

    h_trained_flat = trained_components['hidden']

    tgm_results = {}
    for tname, target in targets.items():
        print(f"  Computing TGM for {tname}...")
        min_len = min(len(h_trained_flat), len(target))
        result = temporal_generalization_matrix(
            h_trained_flat[:min_len], target[:min_len],
            n_windows, bins_per_window)

        gen_matrix, time_indices = result

        # Characterize: diagonal vs off-diagonal strength
        diag = np.mean(np.diag(gen_matrix))
        full_mean = np.mean(gen_matrix)

        # Sustained = high off-diagonal relative to diagonal
        sustained_ratio = full_mean / max(abs(diag), 0.01)

        pattern = 'SUSTAINED' if sustained_ratio > 0.5 else 'TRANSIENT'

        tgm_results[tname] = {
            'diagonal_mean_r2': float(diag),
            'full_mean_r2': float(full_mean),
            'sustained_ratio': float(sustained_ratio),
            'pattern': pattern,
            'matrix_shape': list(gen_matrix.shape),
            'time_step_ms': float(BIN_MS * max(1, bins_per_window // 20)),
        }
        print(f"    Diagonal R2={diag:.4f}, Full mean R2={full_mean:.4f}, "
              f"Pattern: {pattern}")

    elapsed = time.time() - t0
    print(f"\n  Phase 2 complete ({elapsed:.0f}s)")

    return {
        'gate_probing': gate_results,
        'temporal_generalization': tgm_results,
        'elapsed_seconds': elapsed,
    }


# ================================================================
# PHASE 3: Causal Inference
# ================================================================

def run_das_for_target(model, X_test, target, target_name,
                        subspace_dim=8, n_epochs=100):
    """
    Distributed Alignment Search -- interchange interventions.

    Learns rotation R isolating target-aligned subspace of hidden state.
    Swaps subspace between high/low target windows.
    If output changes predictably -> CAUSALLY USED.

    >0.7 = CAUSAL, 0.4-0.7 = PARTIAL, <0.4 = NOT CAUSAL
    """
    set_inference_mode(model)

    with torch.no_grad():
        X_t = torch.FloatTensor(X_test).to(DEVICE)
        y_pred, hidden = model(X_t, return_hidden=True)

    # Window-averaged
    h_mean = hidden.mean(dim=1).cpu().numpy()  # (n_windows, hidden)
    y_mean = y_pred.mean(dim=1).cpu().numpy()  # (n_windows, n_output)

    # Sort by target
    sorted_idx = np.argsort(target)
    n = len(sorted_idx)
    low_idx = sorted_idx[:n // 3]
    high_idx = sorted_idx[-n // 3:]

    h_low = h_mean[low_idx]
    h_high = h_mean[high_idx]
    y_low = y_mean[low_idx]
    y_high = y_mean[high_idx]

    hidden_size = h_mean.shape[1]
    n_pairs = min(len(h_low), len(h_high))

    # Learn rotation matrix
    R = torch.eye(hidden_size, requires_grad=True, device=DEVICE,
                   dtype=torch.float32)
    optimizer = torch.optim.Adam([R], lr=0.01)

    h_low_t = torch.FloatTensor(h_low[:n_pairs]).to(DEVICE)
    h_high_t = torch.FloatTensor(h_high[:n_pairs]).to(DEVICE)
    y_low_t = torch.FloatTensor(y_low[:n_pairs]).to(DEVICE)
    y_high_t = torch.FloatTensor(y_high[:n_pairs]).to(DEVICE)

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        Q, _ = torch.linalg.qr(R)

        h_high_r = h_high_t @ Q.T
        h_low_r = h_low_t @ Q.T

        h_int = h_high_r.clone()
        h_int[:, :subspace_dim] = h_low_r[:, :subspace_dim]
        h_int_orig = h_int @ Q

        with torch.no_grad():
            y_int = model.output_proj(h_int_orig.unsqueeze(1)).squeeze(1)

        dist_to_low = ((y_int - y_low_t) ** 2).mean()
        dist_to_high = ((y_int - y_high_t) ** 2).mean()

        loss = dist_to_low - dist_to_high + 0.01 * (R - Q).pow(2).sum()
        loss.backward()
        optimizer.step()

    # Test interchange accuracy
    with torch.no_grad():
        Q, _ = torch.linalg.qr(R)
        correct = 0

        for i in range(n_pairs):
            h_high_r = (h_high_t[i] @ Q.T).unsqueeze(0)
            h_low_r = (h_low_t[i] @ Q.T).unsqueeze(0)

            h_int = h_high_r.clone()
            h_int[0, :subspace_dim] = h_low_r[0, :subspace_dim]
            h_int_orig = h_int @ Q

            y_int = model.output_proj(h_int_orig.unsqueeze(1)).squeeze()

            dist_to_low = ((y_int - y_low_t[i]) ** 2).sum()
            dist_to_high = ((y_int - y_high_t[i]) ** 2).sum()

            if dist_to_low < dist_to_high:
                correct += 1

    accuracy = correct / max(n_pairs, 1)
    status = 'CAUSAL' if accuracy > 0.7 else \
             'PARTIAL' if accuracy > 0.4 else 'NOT_CAUSAL'

    return {
        'target': target_name,
        'subspace_dim': subspace_dim,
        'interchange_accuracy': float(accuracy),
        'n_pairs': n_pairs,
        'status': status,
    }


def ksg_mi_estimate(X, Y, k=5):
    """
    Kraskov-Stoegbauer-Grassberger mutual information estimator.
    Non-parametric, works for continuous variables.
    """
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    N = len(X)
    XY = np.hstack([X, Y])

    nn_joint = NearestNeighbors(n_neighbors=k+1, metric='chebyshev')
    nn_joint.fit(XY)
    distances, _ = nn_joint.kneighbors(XY)
    epsilon = distances[:, -1]

    nn_x = NearestNeighbors(metric='chebyshev')
    nn_x.fit(X)
    nn_y = NearestNeighbors(metric='chebyshev')
    nn_y.fit(Y)

    n_x = np.zeros(N)
    n_y = np.zeros(N)

    for i in range(N):
        eps_i = max(epsilon[i], 1e-10)
        n_x[i] = len(nn_x.radius_neighbors(X[i:i+1], radius=eps_i,
                                              return_distance=False)[0]) - 1
        n_y[i] = len(nn_y.radius_neighbors(Y[i:i+1], radius=eps_i,
                                              return_distance=False)[0]) - 1

    # Avoid log(0)
    n_x = np.maximum(n_x, 1)
    n_y = np.maximum(n_y, 1)

    mi = digamma(k) + digamma(N) - np.mean(digamma(n_x) + digamma(n_y))
    return max(mi, 0.0)


def conditional_mi_test(hidden, target, output, n_pca_dims=10, k=5):
    """
    I(H;T) and I(H;T|O) to classify as mandatory or zombie.

    I(H;T|O) = I(H;T,O) - I(H;O)

    MANDATORY: I(H;T) > 0 and I(H;T|O) ~ 0 -> all info used by output
    ZOMBIE: I(H;T) > 0 and I(H;T|O) > 0 -> info stored but not transmitted
    """
    N = len(hidden)

    pca_h = PCA(n_components=min(n_pca_dims, hidden.shape[1], N - 1))
    H_reduced = pca_h.fit_transform(hidden)

    if output.ndim > 1 and output.shape[1] > 5:
        pca_out = PCA(n_components=5)
        O_reduced = pca_out.fit_transform(output)
    else:
        O_reduced = output if output.ndim == 2 else output.reshape(-1, 1)

    T = target.reshape(-1, 1)

    mi_h_t = ksg_mi_estimate(H_reduced, T, k=k)
    mi_h_o = ksg_mi_estimate(H_reduced, O_reduced, k=k)

    TO = np.hstack([T, O_reduced])
    mi_h_to = ksg_mi_estimate(H_reduced, TO, k=k)

    cmi = mi_h_to - mi_h_o

    mi_threshold = 0.05
    if mi_h_t < mi_threshold:
        status = 'NOT_ENCODED'
    elif cmi < mi_threshold:
        status = 'MANDATORY'
    else:
        status = 'ZOMBIE'

    return {
        'I_H_T': float(mi_h_t),
        'I_H_O': float(mi_h_o),
        'I_H_TO': float(mi_h_to),
        'I_H_T_given_O': float(cmi),
        'status': status,
    }


def run_phase3(patient_id, model_trained, X_test, targets,
               surviving_targets, n_windows, bins_per_window):
    """Phase 3: DAS + Conditional MI."""
    print("\n" + "=" * 70)
    print("PHASE 3: Causal Inference")
    print("=" * 70)
    t0 = time.time()

    # Get hidden states and model output for CMI
    set_inference_mode(model_trained)
    with torch.no_grad():
        X_t = torch.FloatTensor(X_test).to(DEVICE)
        y_pred, hidden = model_trained(X_t, return_hidden=True)

    h_flat = hidden.cpu().numpy().reshape(-1, hidden.shape[-1])
    y_flat = y_pred.cpu().numpy().reshape(-1, y_pred.shape[-1])

    # Window-level target for DAS
    window_targets = {}
    for tname, target in targets.items():
        min_len = min(len(h_flat), len(target))
        h_w, t_w = to_window_level(h_flat[:min_len], target[:min_len],
                                     n_windows, bins_per_window)
        t = t_w.astype(np.float64)
        t_std = np.std(t)
        if t_std > 1e-8:
            t = (t - np.mean(t)) / t_std
            t = np.clip(t, -3.0, 3.0)
        window_targets[tname] = t

    # 3.1 DAS -- only for Phase 1 survivors
    print("\n" + "-" * 60)
    print("3.1 DISTRIBUTED ALIGNMENT SEARCH (DAS)")
    print("-" * 60)

    das_results = {}
    for tname in surviving_targets:
        if tname not in window_targets:
            continue
        print(f"\n  DAS for {tname}...")
        for subspace_dim in [4, 8, 16]:
            res = run_das_for_target(model_trained, X_test,
                                      window_targets[tname], tname,
                                      subspace_dim=subspace_dim, n_epochs=100)
            key = f"{tname}_dim{subspace_dim}"
            das_results[key] = res
            print(f"    dim={subspace_dim}: accuracy={res['interchange_accuracy']:.3f} "
                  f"-> {res['status']}")

    # 3.2 CMI -- for all targets
    print("\n" + "-" * 60)
    print("3.2 CONDITIONAL MUTUAL INFORMATION")
    print("-" * 60)

    cmi_results = {}
    for tname, target in targets.items():
        min_len = min(len(h_flat), len(target), len(y_flat))
        # Window-level for tractability
        h_w, t_w = to_window_level(h_flat[:min_len], target[:min_len],
                                     n_windows, bins_per_window)
        N_w = n_windows * bins_per_window
        y_w = y_flat[:N_w].reshape(n_windows, bins_per_window, -1).mean(axis=1)

        t = t_w.astype(np.float64)
        t_std = np.std(t)
        if t_std < 1e-8:
            continue
        t = (t - np.mean(t)) / t_std
        t = np.clip(t, -3.0, 3.0)

        print(f"  CMI for {tname}...", end=" ")
        res = conditional_mi_test(h_w, t, y_w)
        cmi_results[tname] = res
        print(f"I(H;T)={res['I_H_T']:.4f}, I(H;T|O)={res['I_H_T_given_O']:.4f} "
              f"-> {res['status']}")

    elapsed = time.time() - t0
    print(f"\n  Phase 3 complete ({elapsed:.0f}s)")

    return {
        'das': das_results,
        'conditional_mi': cmi_results,
        'elapsed_seconds': elapsed,
    }


# ================================================================
# PHASE 4: Multiple Comparison Correction
# ================================================================

def hierarchical_fdr(p_values, alpha=0.05):
    """
    Hierarchical FDR: tree-structured hypothesis testing.

    Level 0: "Did LSTM learn ANY biological variable?" (Fisher's method)
    Level 1: "Oscillatory" family vs "Cognitive" family
    Level 2: Individual targets within surviving families
    """
    from statsmodels.stats.multitest import multipletests

    results = {}

    # Level 0: aggregate p-value per family
    family_pvals = {}
    for family_name, family_targets in p_values.items():
        pvals = list(family_targets.values())
        if len(pvals) > 0:
            _, combined_p = combine_pvalues(pvals, method='fisher')
            family_pvals[family_name] = combined_p

    # Level 1: test families
    family_names = list(family_pvals.keys())
    family_p_array = [family_pvals[f] for f in family_names]

    if len(family_p_array) > 1:
        rej_families, adj_p_families, _, _ = multipletests(
            family_p_array, alpha=alpha, method='fdr_bh')
    elif len(family_p_array) == 1:
        rej_families = [family_p_array[0] < alpha]
        adj_p_families = family_p_array
    else:
        return results

    # Level 2: test within rejected families
    for i, family_name in enumerate(family_names):
        family_result = {
            'family_p': float(family_pvals[family_name]),
            'family_adjusted_p': float(adj_p_families[i]),
            'family_rejected': bool(rej_families[i]),
            'targets': {}
        }

        if rej_families[i]:
            target_names = list(p_values[family_name].keys())
            target_pvals = [p_values[family_name][t] for t in target_names]

            if len(target_pvals) > 1:
                rej_targets, adj_p_targets, _, _ = multipletests(
                    target_pvals, alpha=alpha, method='fdr_bh')
            else:
                rej_targets = [target_pvals[0] < alpha]
                adj_p_targets = target_pvals

            for j, tname in enumerate(target_names):
                family_result['targets'][tname] = {
                    'raw_p': float(target_pvals[j]),
                    'adjusted_p': float(adj_p_targets[j]),
                    'rejected': bool(rej_targets[j]),
                }
        else:
            for tname in p_values[family_name]:
                family_result['targets'][tname] = {
                    'raw_p': float(p_values[family_name][tname]),
                    'adjusted_p': 1.0,
                    'rejected': False,
                    'note': 'parent family not rejected',
                }

        results[family_name] = family_result

    return results


def run_phase4(phase1_results):
    """Phase 4: Hierarchical FDR on Phase 1 p-values."""
    print("\n" + "=" * 70)
    print("PHASE 4: Multiple Comparison Correction (Hierarchical FDR)")
    print("=" * 70)

    # Organize p-values into families
    iaaft = phase1_results.get('iaaft_significance', {})

    oscillatory_targets = ['theta_power', 'gamma_power', 'population_synchrony']
    cognitive_targets = ['arousal', 'threat_pe', 'encoding_success',
                          'event_boundaries', 'temporal_stability']

    p_values = {'oscillatory': {}, 'cognitive': {}}
    for tname, res in iaaft.items():
        p = res.get('p_value', 1.0)
        if tname in oscillatory_targets:
            p_values['oscillatory'][tname] = p
        elif tname in cognitive_targets:
            p_values['cognitive'][tname] = p

    print(f"\n  Oscillatory family: {p_values['oscillatory']}")
    print(f"  Cognitive family:   {p_values['cognitive']}")

    hfdr_results = hierarchical_fdr(p_values, alpha=0.05)

    print("\n  +------------------------+----------+----------+----------+")
    print("  | Family / Target        | Raw p    | Adj. p   | Sig?     |")
    print("  +------------------------+----------+----------+----------+")

    for family_name, fres in hfdr_results.items():
        fam_sig = "YES" if fres['family_rejected'] else "NO"
        print(f"  | {family_name:22s} | {fres['family_p']:.6f} | "
              f"{fres['family_adjusted_p']:.6f} | {fam_sig:8s} |")

        for tname, tres in fres['targets'].items():
            t_sig = "YES" if tres['rejected'] else "NO"
            print(f"  |   {tname:20s} | {tres['raw_p']:.6f} | "
                  f"{tres['adjusted_p']:.6f} | {t_sig:8s} |")

    print("  +------------------------+----------+----------+----------+")

    return hfdr_results


# ================================================================
# PHASE 5: Cross-Domain Comparison
# ================================================================

def normalize_by_cc(delta_r2, cc):
    """dR2_conditional = dR2_raw / CC-squared -- normalized by model quality."""
    cc_sq = cc ** 2
    if cc_sq < 0.01:
        return float('nan')
    return delta_r2 / cc_sq


def run_phase5(patient_id, phase1_results):
    """
    Phase 5: CC-normalized delta-R-squared and transfer probing.

    Transfer probing requires multiple circuits. For the CS48 pilot,
    we compute CC-normalized values and note transfer is pending.
    """
    print("\n" + "=" * 70)
    print("PHASE 5: Cross-Domain Comparison")
    print("=" * 70)

    # 5.1 CC-Normalized delta-R-squared
    print("\n" + "-" * 60)
    print("5.1 CC-NORMALIZED delta-R-squared")
    print("-" * 60)

    # Load focused neuron CCs
    focused_path = RESULTS_DIR / patient_id / 'focused_neurons.json'
    with open(focused_path) as f:
        focused_info = json.load(f)

    focused_ccs = focused_info.get('focused_ccs', [])
    if focused_ccs:
        mean_cc = float(np.mean(focused_ccs))
    else:
        mean_cc = 0.5  # fallback

    print(f"  Mean CC of focused neurons: {mean_cc:.4f}")

    cc_norm_results = {}
    for tname, summary in phase1_results.get('summary', {}).items():
        delta = summary.get('delta_r2_matched', 0)
        normed = normalize_by_cc(delta, mean_cc)
        cc_norm_results[tname] = {
            'delta_r2_raw': float(delta),
            'cc': float(mean_cc),
            'delta_r2_normalized': float(normed) if not np.isnan(normed) else None,
        }
        normed_str = f"{normed:.4f}" if not np.isnan(normed) else "N/A"
        print(f"  {tname:25s}: dR2={delta:+.4f}, CC2={mean_cc**2:.4f}, "
              f"dR2_norm={normed_str}")

    # 5.2 Transfer Probing -- check if other circuits exist
    print("\n" + "-" * 60)
    print("5.2 TRANSFER PROBING")
    print("-" * 60)

    other_circuits = list(PREPROCESSED_DIR.glob('sub-*.npz'))
    other_patients = [p.stem for p in other_circuits if p.stem != patient_id]

    transfer_results = {}
    if not other_patients:
        print("  No other circuits available for transfer probing.")
        print("  Transfer probing will be available when multi-circuit data exists.")
    else:
        print(f"  Found {len(other_patients)} other patients: {other_patients[:5]}")

    return {
        'cc_normalized': cc_norm_results,
        'transfer_probing': transfer_results,
        'mean_cc': float(mean_cc),
    }


# ================================================================
# PHASE 6: Geometric Comparison
# ================================================================

def compute_persistence_basic(hidden_states, max_dim=1, n_points=300):
    """
    Persistent homology of hidden state point cloud.
    Falls back to PCA + basic topology if ripser not installed.
    """
    try:
        from ripser import ripser
        has_ripser = True
    except ImportError:
        has_ripser = False

    # Subsample
    if len(hidden_states) > n_points:
        idx = np.random.choice(len(hidden_states), n_points, replace=False)
        idx.sort()
        points = hidden_states[idx]
    else:
        points = hidden_states

    # PCA reduce
    n_comp = min(10, points.shape[1], points.shape[0] - 1)
    pca = PCA(n_components=n_comp)
    points_reduced = pca.fit_transform(points)
    var_explained = float(np.sum(pca.explained_variance_ratio_))

    if has_ripser:
        result = ripser(points_reduced, maxdim=max_dim)
        persistence_threshold = 0.1
        betti = {}
        for dim in range(max_dim + 1):
            dgm = result['dgms'][dim]
            persistence = dgm[:, 1] - dgm[:, 0]
            finite = persistence[np.isfinite(persistence)]
            betti[f'betti_{dim}'] = int(np.sum(finite > persistence_threshold))

        return {
            'betti_numbers': betti,
            'n_points_used': len(points),
            'pca_variance_explained': var_explained,
            'method': 'ripser',
        }
    else:
        # Fallback: basic geometric statistics
        from scipy.spatial.distance import pdist
        dists = pdist(points_reduced)

        return {
            'mean_pairwise_distance': float(np.mean(dists)),
            'std_pairwise_distance': float(np.std(dists)),
            'pca_variance_explained': var_explained,
            'pca_dim_95': int(np.searchsorted(
                np.cumsum(pca.explained_variance_ratio_), 0.95) + 1),
            'n_points_used': len(points),
            'method': 'pca_fallback (ripser not installed)',
        }


def run_phase6(patient_id, X_test, model_trained, targets,
               n_windows, bins_per_window):
    """Phase 6: Geometric comparison."""
    print("\n" + "=" * 70)
    print("PHASE 6: Geometric Comparison")
    print("=" * 70)
    t0 = time.time()

    # Extract hidden states
    set_inference_mode(model_trained)
    with torch.no_grad():
        X_t = torch.FloatTensor(X_test).to(DEVICE)
        _, hidden = model_trained(X_t, return_hidden=True)
    h_flat = hidden.cpu().numpy().reshape(-1, hidden.shape[-1])

    # 6.1 Contrastive Embedding
    print("\n" + "-" * 60)
    print("6.1 CONTRASTIVE EMBEDDING (CEBRA)")
    print("-" * 60)

    cebra_results = {}
    try:
        import cebra
        has_cebra = True
    except ImportError:
        has_cebra = False
        print("  CEBRA not installed -- skipping contrastive embedding.")
        print("  Install with: pip install cebra")

    if has_cebra:
        for tname in ['population_synchrony', 'gamma_power']:
            if tname not in targets:
                continue
            target = targets[tname]
            min_len = min(len(h_flat), len(target))
            h_w, t_w = to_window_level(h_flat[:min_len], target[:min_len],
                                         n_windows, bins_per_window)

            print(f"  Fitting CEBRA for {tname}...")
            cebra_model = cebra.CEBRA(
                model_architecture='offset10-model',
                batch_size=min(512, len(h_w)),
                output_dimension=3,
                max_iterations=3000,
                verbose=False)
            cebra_model.fit(h_w, t_w)
            embedding = cebra_model.transform(h_w)

            cebra_results[tname] = {
                'embedding_shape': list(embedding.shape),
                'embedding_variance': float(np.var(embedding)),
            }
            print(f"    Embedding shape: {embedding.shape}")

    # 6.2 Persistent Homology
    print("\n" + "-" * 60)
    print("6.2 PERSISTENT HOMOLOGY")
    print("-" * 60)

    # Overall topology
    h_w_all = h_flat[:n_windows * bins_per_window].reshape(
        n_windows, bins_per_window, -1).mean(axis=1)

    print("  Computing overall topology...")
    topo_overall = compute_persistence_basic(h_w_all)
    print(f"    Result: {topo_overall}")

    # Compare high vs low conditions for surviving targets
    topo_by_condition = {}
    for tname in ['population_synchrony', 'gamma_power']:
        if tname not in targets:
            continue
        target = targets[tname]
        min_len = min(len(h_flat), len(target))
        h_w, t_w = to_window_level(h_flat[:min_len], target[:min_len],
                                     n_windows, bins_per_window)

        median_val = np.median(t_w)
        high_mask = t_w >= median_val
        low_mask = t_w < median_val

        if np.sum(high_mask) > 20 and np.sum(low_mask) > 20:
            print(f"\n  Topology for {tname} (high vs low)...")
            topo_high = compute_persistence_basic(h_w[high_mask])
            topo_low = compute_persistence_basic(h_w[low_mask])

            topo_by_condition[tname] = {
                'high': topo_high,
                'low': topo_low,
            }
            print(f"    High: {topo_high}")
            print(f"    Low:  {topo_low}")

    elapsed = time.time() - t0
    print(f"\n  Phase 6 complete ({elapsed:.0f}s)")

    return {
        'cebra': cebra_results,
        'topology_overall': topo_overall,
        'topology_by_condition': topo_by_condition,
        'elapsed_seconds': elapsed,
    }


# ================================================================
# MAIN: RUN ALL PHASES 2-6
# ================================================================

def run_all_phases(patient_id='sub-CS48'):
    print("=" * 70)
    print(f"DESCARTES STATISTICAL HARDENING -- PHASES 2-6 -- {patient_id}")
    print("=" * 70)

    t0_total = time.time()

    # Load Phase 1 results
    phase1_path = OUTPUT_DIR / f'phase1_{patient_id}.json'
    with open(phase1_path) as f:
        phase1_results = json.load(f)

    # Identify Phase 1 survivors
    surviving_targets = [
        tname for tname, summary in phase1_results.get('summary', {}).items()
        if summary.get('survives_phase1', False)
    ]
    print(f"\nPhase 1 survivors: {surviving_targets}")

    # Load data
    print("\n[data] Loading preprocessed data and probe targets...")
    data = np.load(PREPROCESSED_DIR / f'{patient_id}.npz')
    X_train = data['X_train']
    X_val = data['X_val']
    X_test = data['X_test']
    Y_train = data['Y_train']
    Y_val = data['Y_val']
    n_input = int(data['n_input'])

    focused_path = RESULTS_DIR / patient_id / 'focused_neurons.json'
    with open(focused_path) as f:
        focused_info = json.load(f)
    focused_idx = focused_info['focused_indices']
    n_focused = len(focused_idx)

    Y_train_f = Y_train[:, :, focused_idx]
    Y_val_f = Y_val[:, :, focused_idx]

    probe_data = np.load(PROBE_TARGETS_DIR / f'{patient_id}.npz')
    target_names = [n for n in probe_data.files
                    if n not in SKIP_TARGETS and n not in CATEGORICAL_TARGETS]

    window_bins = WINDOW_MS // BIN_MS
    stride_bins = STRIDE_MS // BIN_MS
    test_start_window = X_train.shape[0] + X_val.shape[0]
    n_test = X_test.shape[0]
    bins_per_window = window_bins  # 100
    n_windows = n_test  # 191

    targets = {}
    for name in target_names:
        t_raw = probe_data[name]
        t_aligned = align_target_to_windows(
            t_raw, n_test, test_start_window, window_bins, stride_bins)
        if np.std(t_aligned) < 1e-6:
            continue
        targets[name] = t_aligned

    print(f"  Targets: {list(targets.keys())}")
    print(f"  n_windows={n_windows}, bins_per_window={bins_per_window}")

    # == Phase 2 ==
    phase2_results = run_phase2(
        patient_id, X_test, n_input, n_focused,
        X_train, Y_train_f, X_val, Y_val_f,
        targets, n_windows, bins_per_window)

    # Need trained model for Phase 3 and 6 -- retrain best seed
    best_seed = 0
    torch.manual_seed(best_seed)
    np.random.seed(best_seed)
    model_trained = LimbicPrefrontalLSTM(n_input, n_focused, HIDDEN_SIZE,
                                           N_LAYERS, DROPOUT)
    model_trained = train_model(model_trained, X_train, Y_train_f,
                                  X_val, Y_val_f, best_seed)

    # == Phase 3 ==
    phase3_results = run_phase3(
        patient_id, model_trained, X_test, targets,
        surviving_targets, n_windows, bins_per_window)

    # == Phase 4 ==
    phase4_results = run_phase4(phase1_results)

    # == Phase 5 ==
    phase5_results = run_phase5(patient_id, phase1_results)

    # == Phase 6 ==
    phase6_results = run_phase6(
        patient_id, X_test, model_trained, targets,
        n_windows, bins_per_window)

    # == Final Summary ==
    elapsed_total = time.time() - t0_total

    print("\n" + "=" * 70)
    print(f"ALL PHASES COMPLETE ({elapsed_total:.0f}s)")
    print("=" * 70)

    # Compile final report
    final_report = {
        'patient_id': patient_id,
        'phase1_survivors': surviving_targets,
        'phase2': phase2_results,
        'phase3': phase3_results,
        'phase4': phase4_results,
        'phase5': phase5_results,
        'phase6': phase6_results,
        'total_elapsed_seconds': elapsed_total,
    }

    # Convert numpy types for JSON serialization
    def sanitize(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [sanitize(x) for x in obj]
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    final_report = sanitize(final_report)

    out_path = OUTPUT_DIR / f'phases2to6_{patient_id}.json'
    with open(out_path, 'w') as f:
        json.dump(final_report, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Print final summary table
    print("\n" + "=" * 70)
    print("FINAL INTEGRATED SUMMARY")
    print("=" * 70)

    print("\n+-------------------------+-----------+----------+----------+----------+----------+")
    print("| Target                  | Phase1    | Gate dR2 | CMI      | hFDR     | DAS      |")
    print("|                         | Survives? | (cell)   | Status   | Sig?     | Status   |")
    print("+-------------------------+-----------+----------+----------+----------+----------+")

    for tname in targets:
        p1_surv = "YES" if tname in surviving_targets else "NO"

        # Gate probing cell delta-R-squared
        gate_cell = phase2_results.get('gate_probing', {}).get(tname, {}).get('cell', {})
        cell_dr2 = gate_cell.get('delta_r2', 0)

        # CMI status
        cmi = phase3_results.get('conditional_mi', {}).get(tname, {})
        cmi_status = cmi.get('status', 'N/A')[:8]

        # hFDR
        hfdr_sig = "N/A"
        for family in phase4_results.values():
            if tname in family.get('targets', {}):
                hfdr_sig = "YES" if family['targets'][tname].get('rejected', False) else "NO"

        # DAS (best subspace dim)
        das_status = "N/A"
        for key, res in phase3_results.get('das', {}).items():
            if key.startswith(tname):
                if res.get('interchange_accuracy', 0) > 0.5:
                    das_status = res.get('status', 'N/A')[:8]
                    break
                das_status = res.get('status', 'N/A')[:8]

        print(f"| {tname:23s} | {p1_surv:9s} | {cell_dr2:+.4f}  | "
              f"{cmi_status:8s} | {hfdr_sig:8s} | {das_status:8s} |")

    print("+-------------------------+-----------+----------+----------+----------+----------+")

    # Zombie check
    print("\n  ZOMBIE MECHANISM CHECK:")
    found_zombie = False
    for tname in targets:
        gate_data = phase2_results.get('gate_probing', {}).get(tname, {})
        cell_dr2 = gate_data.get('cell', {}).get('delta_r2', 0)
        hidden_dr2 = gate_data.get('hidden', {}).get('delta_r2', 0)
        if cell_dr2 > 0.05 and hidden_dr2 < 0.03:
            print(f"    {tname}: ZOMBIE -- dR2 cell={cell_dr2:+.4f}, "
                  f"hidden={hidden_dr2:+.4f}")
            found_zombie = True
    if not found_zombie:
        print("    No zombie mechanisms detected.")

    # TGM pattern
    print("\n  TEMPORAL DYNAMICS:")
    for tname, tgm in phase2_results.get('temporal_generalization', {}).items():
        print(f"    {tname}: {tgm['pattern']} "
              f"(diag R2={tgm['diagonal_mean_r2']:.4f})")

    return final_report


if __name__ == '__main__':
    run_all_phases('sub-CS48')
