"""
run_analysis.py — DESCARTES Circuit 5 Post-Pipeline Analysis
Tasks 1-7 from the analysis instructions.

Requires: patient reports in results/*/patient_report.json
Uses JSON serialization only (no unsafe formats).
"""
import numpy as np
import json
import sys
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / 'results'
PREPROCESSED_DIR = BASE_DIR / 'preprocessed_data'
PROBE_TARGETS_DIR = BASE_DIR / 'probe_targets'
FIGURES_DIR = BASE_DIR / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PATIENTS = ['sub-CS42', 'sub-CS43', 'sub-CS44', 'sub-CS47', 'sub-CS48',
            'sub-CS51', 'sub-CS53', 'sub-CS54', 'sub-CS55', 'sub-CS56',
            'sub-CS57', 'sub-CS58', 'sub-CS60', 'sub-CS62']

PROBE_TARGETS = ['arousal', 'threat_pe', 'emotion_category', 'event_boundaries',
                 'population_synchrony', 'temporal_stability', 'theta_power',
                 'gamma_power', 'encoding_success']

# ================================================================
# TASK 1: COMPILE FINAL RESULTS
# ================================================================
def task1_compile_results():
    print("\n" + "="*80)
    print("TASK 1: COMPILE FINAL RESULTS")
    print("="*80)

    reports = {}
    focused_info = {}
    for pid in PATIENTS:
        rpath = RESULTS_DIR / pid / 'patient_report.json'
        fpath = RESULTS_DIR / pid / 'focused_neurons.json'
        if rpath.exists():
            with open(rpath) as f:
                reports[pid] = json.load(f)
        if fpath.exists():
            with open(fpath) as f:
                focused_info[pid] = json.load(f)

    n_found = len(reports)
    print(f"\nFound {n_found}/{len(PATIENTS)} patient reports")
    if n_found == 0:
        print("ERROR: No reports found. Pipeline may still be running.")
        return None, None

    # --- Patient summary table ---
    lines = []
    lines.append("DESCARTES CIRCUIT 5 -- PATIENT SUMMARY TABLE")
    lines.append("="*100)
    header = f"{'Patient':<10} | {'Neurons(in/out)':<16} | {'Focused':<8} | {'Mean CC':<10} | {'Quality':<12} | {'Best Probe Result'}"
    lines.append(header)
    lines.append("-"*100)

    for pid in PATIENTS:
        if pid not in reports:
            lines.append(f"{pid:<10} | {'N/A':>16} | {'N/A':>8} | {'N/A':>10} | {'NOT RUN':<12} | N/A")
            continue

        r = reports[pid]
        n_in = r['n_input_neurons']
        fi = focused_info.get(pid, {})
        n_out_total = fi.get('n_output_total', '?')
        n_foc = r['n_focused_neurons']
        cc = r['mean_cc']
        qual = r['quality_flag']
        topk = fi.get('used_topk_fallback', False)
        if topk:
            qual += '*'

        # Find best probe result
        best_target = 'all zombie'
        best_delta = -999
        for tname, tres in r.get('probe_results', {}).items():
            if tres.get('status') in ('STRONG', 'CANDIDATE'):
                d = tres.get('delta', 0)
                if d > best_delta:
                    best_delta = d
                    best_target = f"{tname} {d:+.4f} [{tres['status']}]"

        lines.append(f"{pid:<10} | {n_in:>3}/{n_out_total:<12} | {n_foc:>8} | {cc:>10.4f} | {qual:<12} | {best_target}")

    lines.append("-"*100)
    lines.append("* = used top-K fallback (all neurons below CC threshold)")
    lines.append(f"\nTotal patients with reports: {n_found}/{len(PATIENTS)}")
    ok_patients = [pid for pid in reports if reports[pid]['quality_flag'] == 'OK']
    low_patients = [pid for pid in reports if reports[pid]['quality_flag'] == 'LOW_QUALITY']
    lines.append(f"OK quality (CC >= 0.3): {len(ok_patients)} -- {', '.join(ok_patients)}")
    lines.append(f"LOW quality (CC < 0.3): {len(low_patients)}")

    # --- Cross-patient probe target summary ---
    lines.append("\n\n" + "="*100)
    lines.append("CROSS-PATIENT PROBE TARGET SUMMARY")
    lines.append("="*100)

    t_header = f"{'Target':<25} | {'Valid pts':<10} | {'STRONG':<7} | {'CAND':<5} | {'zombie':<7} | {'INVALID':<7} | {'Mean dR2':<10} | {'Learned?'}"
    lines.append(t_header)
    lines.append("-"*100)

    for tname in PROBE_TARGETS:
        n_strong = 0; n_cand = 0; n_zombie = 0; n_invalid = 0; n_valid = 0
        deltas = []

        for pid in reports:
            pr = reports[pid].get('probe_results', {}).get(tname, {})
            status = pr.get('status', 'N/A')
            if status == 'STRONG':
                n_strong += 1; n_valid += 1
                deltas.append(pr.get('delta', 0))
            elif status == 'CANDIDATE':
                n_cand += 1; n_valid += 1
                deltas.append(pr.get('delta', 0))
            elif status == 'zombie':
                n_zombie += 1; n_valid += 1
                if pr.get('valid', False):
                    deltas.append(pr.get('delta', 0))
            elif status == 'INVALID':
                n_invalid += 1

        mean_d = f"{np.mean(deltas):+.4f}" if deltas else "N/A"
        pct = (n_strong + n_cand) / max(n_valid, 1) * 100
        learned = "YES" if pct >= 50 else "no"

        lines.append(f"{tname:<25} | {n_valid:>3}/{n_found:<6} | {n_strong:>7} | {n_cand:>5} | {n_zombie:>7} | {n_invalid:>7} | {mean_d:>10} | {learned}")

    lines.append("-"*100)

    table_text = "\n".join(lines)
    print(table_text)

    out_path = RESULTS_DIR / 'final_summary.txt'
    with open(out_path, 'w') as f:
        f.write(table_text)
    print(f"\nSaved: {out_path}")

    return reports, focused_info


# ================================================================
# TASK 2: DIAGNOSE THE CC PROBLEM
# ================================================================
def task2_cc_diagnostic():
    print("\n" + "="*80)
    print("TASK 2: CC DIAGNOSTIC -- Why do most patients have CC < 0.3?")
    print("="*80)

    lines = []
    lines.append("CC DIAGNOSTIC TABLE")
    lines.append("="*110)
    header = (f"{'Patient':<10} | {'n_in':<5} | {'n_out':<5} | "
              f"{'CC>0.3':<6} | {'CC>0.2':<6} | {'CC>0.1':<6} | "
              f"{'Best CC':<8} | {'Max xCorr':<9} | {'Top-K?':<6}")
    lines.append(header)
    lines.append("-"*110)

    diag_data = []

    for pid in PATIENTS:
        prep_path = PREPROCESSED_DIR / f"{pid}.npz"
        fn_path = RESULTS_DIR / pid / 'focused_neurons.json'

        if not prep_path.exists():
            lines.append(f"{pid:<10} | NO PREPROCESSED DATA")
            continue

        data = np.load(prep_path)
        n_in = int(data['n_input'])
        n_out = int(data['n_output'])

        best_cc = 0.0
        all_ccs = []
        topk = False
        if fn_path.exists():
            with open(fn_path) as f:
                fi = json.load(f)
            all_ccs = fi.get('all_ccs', [])
            topk = fi.get('used_topk_fallback', False)
            if all_ccs:
                best_cc = max(all_ccs)

        cc_above_03 = sum(1 for c in all_ccs if c >= 0.3)
        cc_above_02 = sum(1 for c in all_ccs if c >= 0.2)
        cc_above_01 = sum(1 for c in all_ccs if c >= 0.1)

        # Cross-region correlation
        X_train = data['X_train']
        Y_train = data['Y_train']

        X_flat = X_train.reshape(-1, X_train.shape[-1])
        Y_flat = Y_train.reshape(-1, Y_train.shape[-1])

        max_cross_corr = []
        for j in range(Y_flat.shape[1]):
            best_r = 0
            for i in range(X_flat.shape[1]):
                r = np.corrcoef(X_flat[:, i], Y_flat[:, j])[0, 1]
                if not np.isnan(r) and abs(r) > abs(best_r):
                    best_r = r
            max_cross_corr.append(abs(best_r))

        max_xc = max(max_cross_corr) if max_cross_corr else 0.0
        mean_xc = np.mean(max_cross_corr) if max_cross_corr else 0.0

        topk_str = "YES" if topk else "no"
        lines.append(f"{pid:<10} | {n_in:<5} | {n_out:<5} | "
                     f"{cc_above_03:<6} | {cc_above_02:<6} | {cc_above_01:<6} | "
                     f"{best_cc:<8.3f} | {max_xc:<9.3f} | {topk_str:<6}")

        diag_data.append({
            'patient': pid, 'n_in': n_in, 'n_out': n_out,
            'n_total': n_in + n_out,
            'cc_above_03': cc_above_03, 'cc_above_02': cc_above_02,
            'cc_above_01': cc_above_01, 'best_cc': best_cc,
            'max_cross_corr': max_xc, 'mean_cross_corr': mean_xc,
            'topk': topk,
        })

    lines.append("-"*110)

    lines.append("\nDIAGNOSIS:")
    lines.append("-"*40)

    if diag_data:
        from scipy.stats import pearsonr
        n_totals = [d['n_total'] for d in diag_data]
        best_ccs = [d['best_cc'] for d in diag_data]
        max_xcs = [d['max_cross_corr'] for d in diag_data]

        if len(set(n_totals)) > 2:
            r_nt_cc, p_nt_cc = pearsonr(n_totals, best_ccs)
            lines.append(f"  Pearson(total_neurons, best_CC) = {r_nt_cc:.3f} (p={p_nt_cc:.4f})")
            r_nt_xc, p_nt_xc = pearsonr(n_totals, max_xcs)
            lines.append(f"  Pearson(total_neurons, max_cross_corr) = {r_nt_xc:.3f} (p={p_nt_xc:.4f})")

        for threshold_n in [30, 50, 60, 80, 100]:
            above = [d for d in diag_data if d['n_total'] >= threshold_n]
            below = [d for d in diag_data if d['n_total'] < threshold_n]
            above_ok = sum(1 for d in above if d['best_cc'] >= 0.3)
            below_ok = sum(1 for d in below if d['best_cc'] >= 0.3)
            lines.append(f"  Neurons >= {threshold_n}: {len(above)} patients, {above_ok} with CC>0.3")
            lines.append(f"  Neurons <  {threshold_n}: {len(below)} patients, {below_ok} with CC>0.3")

    table_text = "\n".join(lines)
    print(table_text)

    out_path = RESULTS_DIR / 'cc_diagnostic.txt'
    with open(out_path, 'w') as f:
        f.write(table_text)
    print(f"\nSaved: {out_path}")

    with open(RESULTS_DIR / 'cc_diagnostic.json', 'w') as f:
        json.dump(diag_data, f, indent=2)

    return diag_data


# ================================================================
# TASK 3: WINDOW-LEVEL PROBING (CS48 only)
# ================================================================
def task3_window_level_probing():
    print("\n" + "="*80)
    print("TASK 3: WINDOW-LEVEL PROBING (sub-CS48)")
    print("="*80)

    import torch
    import torch.nn as nn
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    pid = 'sub-CS48'
    prep_path = PREPROCESSED_DIR / f"{pid}.npz"
    probe_path = PROBE_TARGETS_DIR / f"{pid}.npz"
    report_path = RESULTS_DIR / pid / 'patient_report.json'

    if not all(p.exists() for p in [prep_path, probe_path, report_path]):
        print("ERROR: Missing CS48 files.")
        return None

    data = np.load(prep_path)
    probes = np.load(probe_path)
    with open(report_path) as f:
        report = json.load(f)

    n_input = int(data['n_input'])
    n_focused = report['n_focused_neurons']
    focused_idx = report['focused_neuron_ids']

    HIDDEN_SIZE = 128; N_LAYERS = 2; DROPOUT = 0.1
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Import model and training from run_all_patients
    sys.path.insert(0, str(BASE_DIR))
    from run_all_patients import (LimbicPrefrontalLSTM, train_model,
                                   set_inference_mode)

    X_train = data['X_train']
    Y_train_f = data['Y_train'][:, :, focused_idx]
    X_val = data['X_val']
    Y_val_f = data['Y_val'][:, :, focused_idx]
    X_test = data['X_test']
    n_test = X_test.shape[0]

    BIN_MS = 20; WINDOW_MS = 2000; STRIDE_MS = 500
    window_bins = WINDOW_MS // BIN_MS
    stride_bins = STRIDE_MS // BIN_MS
    train_n = data['X_train'].shape[0]
    val_n = data['X_val'].shape[0]
    test_start_window = train_n + val_n

    # Compute window-level targets
    targets_to_probe = ['arousal', 'theta_power', 'threat_pe',
                        'event_boundaries', 'temporal_stability',
                        'gamma_power', 'population_synchrony',
                        'encoding_success']

    window_targets = {}
    for tname in targets_to_probe:
        target_raw = probes[tname]
        vals = []
        for w in range(n_test):
            s = (test_start_window + w) * stride_bins
            e = s + window_bins
            if e <= len(target_raw):
                vals.append(np.mean(target_raw[s:e]))
            else:
                chunk = target_raw[s:min(e, len(target_raw))]
                vals.append(np.mean(chunk) if len(chunk) > 0 else 0.0)
        window_targets[tname] = np.array(vals, dtype=np.float32)

    def sanitized_ridge_window(h_trained, target, h_untrained, n_folds=5):
        target_z = (target - np.mean(target)) / max(np.std(target), 1e-8)
        target_z = np.clip(target_z, -3, 3)
        kf = KFold(n_splits=n_folds, shuffle=False)

        def probe_r2(hidden):
            sc = StandardScaler()
            h_s = sc.fit_transform(hidden)
            fold_r2 = []
            for tr, te in kf.split(h_s):
                mdl = RidgeCV(alphas=np.logspace(-3, 3, 10))
                mdl.fit(h_s[tr], target_z[tr])
                pred = mdl.predict(h_s[te])
                ss_res = np.sum((target_z[te] - pred)**2)
                ss_tot = np.sum((target_z[te] - np.mean(target_z[te]))**2)
                r2 = 1 - ss_res / max(ss_tot, 1e-10)
                fold_r2.append(float(np.clip(r2, -1, 1)))
            return float(np.mean(fold_r2))

        r2_tr = probe_r2(h_trained)
        r2_un = probe_r2(h_untrained)
        return {'r2_trained': r2_tr, 'r2_untrained': r2_un,
                'delta_r2': r2_tr - r2_un, 'valid': r2_tr > 0}

    # Train 3 seeds fresh for window-level probing
    N_WINDOW_SEEDS = 3
    print(f"  Training {N_WINDOW_SEEDS} seeds for CS48 window-level probing...")

    trained_h_win = []
    untrained_h_win = []

    for s in range(N_WINDOW_SEEDS):
        model = LimbicPrefrontalLSTM(n_input, n_focused, HIDDEN_SIZE, N_LAYERS, DROPOUT)
        mean_cc, _, _, epochs = train_model(model, X_train, Y_train_f, X_val, Y_val_f, s)
        print(f"    Seed {s}: CC={mean_cc:.3f}, epochs={epochs}")

        set_inference_mode(model)
        with torch.no_grad():
            X_t = torch.FloatTensor(X_test).to(DEVICE)
            _, h = model(X_t, return_hidden=True)
            h_np = h.cpu().numpy()
        h_window = h_np.mean(axis=1)
        trained_h_win.append(h_window)

        torch.manual_seed(100 + s)
        model_un = LimbicPrefrontalLSTM(n_input, n_focused, HIDDEN_SIZE, N_LAYERS, DROPOUT).to(DEVICE)
        set_inference_mode(model_un)
        with torch.no_grad():
            _, h_un = model_un(X_t, return_hidden=True)
        untrained_h_win.append(h_un.cpu().numpy().mean(axis=1))

    # Probe each target at window level
    results_window = {}
    print(f"\n  Window-level probing results:")
    print(f"  {'Target':<25} | {'Window dR2':<12} | {'Bin dR2':<10} | {'Change':<10} | {'Bin Status'}")
    print(f"  {'-'*75}")

    for tname in targets_to_probe:
        wt = window_targets[tname]
        if np.std(wt) < 1e-6:
            results_window[tname] = {'status': 'INVALID', 'reason': 'near-constant'}
            continue

        seed_results = []
        for s in range(N_WINDOW_SEEDS):
            res = sanitized_ridge_window(trained_h_win[s], wt, untrained_h_win[s])
            seed_results.append(res)

        mean_delta = float(np.mean([r['delta_r2'] for r in seed_results]))
        std_delta = float(np.std([r['delta_r2'] for r in seed_results]))
        mean_r2_tr = float(np.mean([r['r2_trained'] for r in seed_results]))
        mean_r2_un = float(np.mean([r['r2_untrained'] for r in seed_results]))
        any_valid = any(r['valid'] for r in seed_results)

        bin_result = report.get('probe_results', {}).get(tname, {})
        bin_delta = bin_result.get('delta', 'N/A')
        bin_status = bin_result.get('status', 'N/A')

        improvement = float(mean_delta - bin_delta) if isinstance(bin_delta, (int, float)) else None

        results_window[tname] = {
            'window_delta_r2': mean_delta,
            'window_delta_std': std_delta,
            'window_r2_trained': mean_r2_tr,
            'window_r2_untrained': mean_r2_un,
            'window_valid': any_valid,
            'bin_delta_r2': bin_delta,
            'bin_status': bin_status,
            'improvement': improvement,
        }

        bd_s = f"{bin_delta:+.4f}" if isinstance(bin_delta, (int, float)) else str(bin_delta)
        imp_s = f"{improvement:+.4f}" if improvement is not None else "N/A"
        print(f"  {tname:<25} | {mean_delta:>+12.4f} | {bd_s:>10} | {imp_s:>10} | {bin_status}")

    out_path = RESULTS_DIR / pid / 'window_level_probing.json'
    with open(out_path, 'w') as f:
        json.dump(results_window, f, indent=2)
    print(f"\n  Saved: {out_path}")

    return results_window


# ================================================================
# TASK 4: NEURON COUNT vs QUALITY THRESHOLD
# ================================================================
def task4_neuron_threshold(diag_data):
    print("\n" + "="*80)
    print("TASK 4: NEURON COUNT vs MODEL QUALITY")
    print("="*80)

    if not diag_data:
        print("ERROR: No diagnostic data available.")
        return

    from scipy.stats import pearsonr

    lines = []
    lines.append("NEURON COUNT vs MODEL QUALITY ANALYSIS")
    lines.append("="*60)

    for d in diag_data:
        rpath = RESULTS_DIR / d['patient'] / 'patient_report.json'
        if rpath.exists():
            with open(rpath) as f:
                r = json.load(f)
            d['mean_cc'] = r['mean_cc']
            d['quality'] = r['quality_flag']
        else:
            d['mean_cc'] = None
            d['quality'] = None

    valid = [d for d in diag_data if d['mean_cc'] is not None]

    if len(valid) < 3:
        lines.append("Too few patients with reports for correlation analysis.")
        print("\n".join(lines))
        return

    n_totals = np.array([d['n_total'] for d in valid])
    n_ins = np.array([d['n_in'] for d in valid])
    mean_ccs = np.array([d['mean_cc'] for d in valid])
    best_pilot = np.array([d['best_cc'] for d in valid])

    r1, p1 = pearsonr(n_totals, mean_ccs)
    r2, p2 = pearsonr(n_ins, mean_ccs)
    r3, p3 = pearsonr(best_pilot, mean_ccs)

    lines.append(f"\nCorrelations (n={len(valid)} patients):")
    lines.append(f"  total_neurons vs mean_CC:   r={r1:.3f}, p={p1:.4f}")
    lines.append(f"  n_input vs mean_CC:         r={r2:.3f}, p={p2:.4f}")
    lines.append(f"  best_pilot_CC vs mean_CC:   r={r3:.3f}, p={p3:.4f}")

    lines.append(f"\nThreshold analysis:")
    for threshold in [30, 50, 60, 80, 100]:
        above = [d for d in valid if d['n_total'] >= threshold]
        below = [d for d in valid if d['n_total'] < threshold]
        above_ok = sum(1 for d in above if d['mean_cc'] >= 0.3)
        above_mean = np.mean([d['mean_cc'] for d in above]) if above else 0
        below_mean = np.mean([d['mean_cc'] for d in below]) if below else 0
        lines.append(f"  >= {threshold:>3} neurons: {len(above):>2} pts, "
                     f"{above_ok} OK (mean CC={above_mean:.3f})")

    ok_patients = [d for d in valid if d['mean_cc'] >= 0.3]
    if ok_patients:
        min_n = min(d['n_total'] for d in ok_patients)
        lines.append(f"\nMinimum neurons for CC>=0.3: {min_n}")
        lines.append(f"OK patients: {[d['patient'] for d in ok_patients]}")
    else:
        lines.append(f"\nNo patients achieved CC >= 0.3")

    table_text = "\n".join(lines)
    print(table_text)

    out_path = RESULTS_DIR / 'neuron_threshold_analysis.txt'
    with open(out_path, 'w') as f:
        f.write(table_text)
    print(f"\nSaved: {out_path}")


# ================================================================
# TASK 5: CS48 COMPREHENSIVE REPORT
# ================================================================
def task5_cs48_comprehensive(window_results):
    print("\n" + "="*80)
    print("TASK 5: CS48 COMPREHENSIVE REPORT")
    print("="*80)

    pid = 'sub-CS48'
    rpath = RESULTS_DIR / pid / 'patient_report.json'
    fpath = RESULTS_DIR / pid / 'focused_neurons.json'
    ppath = PREPROCESSED_DIR / f"{pid}.npz"

    if not rpath.exists():
        print("ERROR: CS48 report not found.")
        return

    with open(rpath) as f:
        report = json.load(f)
    with open(fpath) as f:
        focused = json.load(f)

    data = np.load(ppath, allow_pickle=True)
    n_in = int(data['n_input'])
    n_out = int(data['n_output'])
    regions_in = [str(r) for r in data['input_regions']]
    regions_out = [str(r) for r in data['output_regions']]

    from collections import Counter
    in_regions = Counter(regions_in)
    out_regions = Counter(regions_out)

    lines = []
    lines.append("="*80)
    lines.append("DESCARTES CIRCUIT 5 -- COMPREHENSIVE PATIENT REPORT")
    lines.append(f"Patient: {pid}")
    lines.append("="*80)

    lines.append("\n1. DATASET CHARACTERISTICS")
    lines.append("-"*40)
    lines.append(f"  Total neurons: {n_in + n_out} ({n_in} input, {n_out} output)")
    lines.append(f"  Focused output neurons: {report['n_focused_neurons']} "
                 f"(CC > {focused['threshold_used']}, natural selection)")
    lines.append(f"  Top-K fallback: {'YES' if focused['used_topk_fallback'] else 'NO'}")
    lines.append(f"\n  Input regions (limbic):")
    for r, c in sorted(in_regions.items()):
        lines.append(f"    {r}: {c} neurons")
    lines.append(f"\n  Output regions (prefrontal):")
    for r, c in sorted(out_regions.items()):
        lines.append(f"    {r}: {c} neurons")

    n_windows = data['X_train'].shape[0] + data['X_val'].shape[0] + data['X_test'].shape[0]
    movie_dur = float(data['movie_stop']) - float(data['movie_start'])
    lines.append(f"\n  Total windows: {n_windows} "
                 f"({data['X_train'].shape[0]} train / "
                 f"{data['X_val'].shape[0]} val / "
                 f"{data['X_test'].shape[0]} test)")
    lines.append(f"  Movie duration: {movie_dur:.1f} seconds")

    lines.append("\n\n2. MODEL QUALITY")
    lines.append("-"*40)
    lines.append(f"  Mean CC across 10 seeds: {report['mean_cc']:.4f} +/- {report['cc_std']:.4f}")
    lines.append(f"  Quality flag: {report['quality_flag']}")

    all_ccs = focused['all_ccs']
    focused_ccs = focused['focused_ccs']
    lines.append(f"\n  Per-neuron CC distribution (all {n_out} output neurons):")
    lines.append(f"    Min:    {min(all_ccs):.3f}")
    lines.append(f"    25th:   {np.percentile(all_ccs, 25):.3f}")
    lines.append(f"    Median: {np.median(all_ccs):.3f}")
    lines.append(f"    75th:   {np.percentile(all_ccs, 75):.3f}")
    lines.append(f"    Max:    {max(all_ccs):.3f}")
    lines.append(f"\n  Focused neuron CCs: {[f'{c:.3f}' for c in focused_ccs]}")

    lines.append("\n\n3. PROBING RESULTS (9 targets)")
    lines.append("-"*40)
    lines.append(f"{'Target':<25} | {'dR2/dAcc':<10} | {'R2_tr':<8} | {'R2_un':<8} | "
                 f"{'Consist':<8} | {'Valid':<6} | {'Status'}")
    lines.append("-"*90)

    for tname in PROBE_TARGETS:
        pr = report.get('probe_results', {}).get(tname, {})
        if not pr or pr.get('status') == 'INVALID':
            lines.append(f"{tname:<25} | {'N/A':>10} | {'N/A':>8} | {'N/A':>8} | "
                        f"{'N/A':>8} | {'N/A':>6} | INVALID")
            continue

        delta = pr.get('delta', 0)
        m_tr = pr.get('metric_trained', 0)
        m_un = pr.get('metric_untrained', 0)
        cons = pr.get('seed_consistency', '?')
        valid_flag = pr.get('valid', False)
        status = pr.get('status', '?')

        lines.append(f"{tname:<25} | {delta:>+10.4f} | {m_tr:>8.4f} | {m_un:>8.4f} | "
                     f"{cons:>8} | {'YES' if valid_flag else 'no':>6} | {status}")
    lines.append("-"*90)

    if window_results:
        lines.append("\n\n4. WINDOW-LEVEL PROBING COMPARISON")
        lines.append("-"*40)
        lines.append(f"{'Target':<25} | {'Bin dR2':<10} | {'Window dR2':<10} | {'Change':<10}")
        lines.append("-"*70)
        for tname in ['arousal', 'theta_power', 'threat_pe', 'event_boundaries',
                      'temporal_stability', 'gamma_power', 'population_synchrony',
                      'encoding_success']:
            if tname in window_results and 'window_delta_r2' in window_results[tname]:
                wr = window_results[tname]
                bd = wr.get('bin_delta_r2', 'N/A')
                wd = wr['window_delta_r2']
                imp = wr.get('improvement', 'N/A')
                bd_s = f"{bd:+.4f}" if isinstance(bd, (int, float)) else str(bd)
                imp_s = f"{imp:+.4f}" if isinstance(imp, (int, float)) else str(imp)
                lines.append(f"{tname:<25} | {bd_s:>10} | {wd:>+10.4f} | {imp_s:>10}")
        lines.append("-"*70)

    lines.append("\n\n5. INTERPRETATION")
    lines.append("-"*40)

    candidate_targets = []
    for tname in PROBE_TARGETS:
        pr = report.get('probe_results', {}).get(tname, {})
        if pr.get('status') in ('STRONG', 'CANDIDATE'):
            candidate_targets.append((tname, pr))

    if candidate_targets:
        lines.append(f"  CANDIDATE+ targets ({len(candidate_targets)}):")
        for tname, pr in candidate_targets:
            lines.append(f"    - {tname}: dR2={pr['delta']:+.4f} [{pr['status']}]")
            if tname == 'temporal_stability':
                lines.append(f"      Plausibility: HIGH. Temporal stability of population firing "
                            f"patterns is a known signature of sustained cognitive processing. "
                            f"Replicated from Circuits 3-4.")
            elif tname == 'theta_power':
                lines.append(f"      Plausibility: HIGH. Theta oscillations linked to memory "
                            f"encoding and emotional processing.")
    else:
        lines.append(f"  No targets achieved CANDIDATE+ status at bin level.")

    lines.append(f"\n  Cross-circuit comparison:")
    lines.append(f"    C2 (HC navigation): theta_power MANDATORY")
    lines.append(f"    C3 (mouse WM): temporal_stability MANDATORY")
    lines.append(f"    C4 (human WM): temporal_stability Variable")
    ts_status = report.get('probe_results', {}).get('temporal_stability', {}).get('status', 'zombie')
    lines.append(f"    C5 (emotion): temporal_stability = {ts_status}")

    table_text = "\n".join(lines)
    print(table_text)

    out_txt = RESULTS_DIR / pid / 'comprehensive_report.txt'
    with open(out_txt, 'w') as f:
        f.write(table_text)
    print(f"\nSaved: {out_txt}")

    comp_json = {
        'patient_id': pid,
        'dataset': {
            'n_input': n_in, 'n_output': n_out,
            'n_focused': report['n_focused_neurons'],
            'input_regions': dict(in_regions),
            'output_regions': dict(out_regions),
            'n_windows_total': int(n_windows),
            'movie_duration_s': float(movie_dur),
        },
        'model_quality': {
            'mean_cc': report['mean_cc'],
            'cc_std': report['cc_std'],
            'quality_flag': report['quality_flag'],
        },
        'probing_results': report['probe_results'],
        'window_level_probing': window_results if window_results else {},
        'candidate_targets': [t[0] for t in candidate_targets],
    }
    out_json = RESULTS_DIR / pid / 'comprehensive_report.json'
    with open(out_json, 'w') as f:
        json.dump(comp_json, f, indent=2)
    print(f"Saved: {out_json}")


# ================================================================
# TASK 6: CIRCUIT 5 FINAL SUMMARY
# ================================================================
def task6_final_summary(reports, diag_data):
    print("\n" + "="*80)
    print("TASK 6: CIRCUIT 5 FINAL SUMMARY")
    print("="*80)

    if not reports:
        print("ERROR: No reports available.")
        return

    n_reports = len(reports)
    ok_patients = {pid: r for pid, r in reports.items() if r['quality_flag'] == 'OK'}
    marginal = {pid: r for pid, r in reports.items()
                if r['mean_cc'] >= 0.2 and r['mean_cc'] < 0.3}

    min_neurons_ok = None
    if diag_data:
        ok_diag = [d for d in diag_data if d.get('mean_cc', 0) and d.get('mean_cc', 0) >= 0.3]
        if ok_diag:
            min_neurons_ok = min(d['n_total'] for d in ok_diag)

    lines = []
    lines.append("="*80)
    lines.append("DESCARTES CIRCUIT 5 RESULTS")
    lines.append("Limbic -> Prefrontal Emotion Transformation")
    lines.append("DANDI 000623, 14 patients, 8-minute movie watching")
    lines.append("="*80)

    lines.append(f"\nPATIENT QUALITY:")
    lines.append(f"  Patients with CC > 0.3 (reliable):    {len(ok_patients)}/{n_reports}")
    lines.append(f"  Patients with CC 0.2-0.3 (marginal):  {len(marginal)}/{n_reports}")
    lines.append(f"  Patients with CC < 0.2 (unreliable):  "
                 f"{n_reports - len(ok_patients) - len(marginal)}/{n_reports}")
    if min_neurons_ok:
        lines.append(f"  Minimum total neurons for CC > 0.3:   ~{min_neurons_ok} neurons")

    lines.append(f"\nPROBE TARGET RESULTS (reliable patients only, n={len(ok_patients)}):")
    if ok_patients:
        lines.append(f"{'Target':<25} | {'STRONG':<7} | {'CAND':<5} | {'zombie':<7} | {'Mean dR2':<10}")
        lines.append("-"*60)
        for tname in PROBE_TARGETS:
            ns = nc = nz = 0
            deltas = []
            for pid, r in ok_patients.items():
                pr = r.get('probe_results', {}).get(tname, {})
                s = pr.get('status', 'N/A')
                if s == 'STRONG': ns += 1
                elif s == 'CANDIDATE': nc += 1
                elif s == 'zombie': nz += 1
                d = pr.get('delta')
                if d is not None and isinstance(d, (int, float)):
                    deltas.append(d)
            md = f"{np.mean(deltas):+.4f}" if deltas else "N/A"
            lines.append(f"{tname:<25} | {ns:>7} | {nc:>5} | {nz:>7} | {md:>10}")

    # Circuit comparison
    c5_status = {}
    ref = ok_patients if ok_patients else reports
    for tname in PROBE_TARGETS:
        statuses = []
        for pid, r in ref.items():
            pr = r.get('probe_results', {}).get(tname, {})
            statuses.append(pr.get('status', 'N/A'))
        n_cp = sum(1 for s in statuses if s in ('STRONG', 'CANDIDATE'))
        if n_cp > len(ref) * 0.5:
            c5_status[tname] = 'Learned'
        elif n_cp > 0:
            c5_status[tname] = 'Variable'
        else:
            c5_status[tname] = 'zombie'

    lines.append(f"\nCOMPARISON TO PREVIOUS CIRCUITS:")
    lines.append(f"{'Variable':<25} | {'C2:HC':<8} | {'C3:Mouse':<8} | {'C4:Human':<8} | {'C5:Emotion'}")
    lines.append("-"*70)

    circuit_data = [
        ('theta_power',         'MAND',   'MAND',   'Variable'),
        ('gamma_power',         'MAND',   'N/A',    'N/A'),
        ('temporal_stability',  'N/A',    'MAND',   'Variable'),
        ('population_synchrony','N/A',    'N/A',    'Variable'),
        ('arousal',             'N/A',    'N/A',    'N/A'),
        ('emotion_category',    'N/A',    'N/A',    'N/A'),
        ('encoding_success',    'N/A',    'N/A',    'N/A'),
        ('event_boundaries',    'N/A',    'N/A',    'N/A'),
        ('threat_pe',           'N/A',    'N/A',    'N/A'),
    ]

    for tname, c2, c3, c4 in circuit_data:
        c5 = c5_status.get(tname, 'zombie')
        lines.append(f"{tname:<25} | {c2:<8} | {c3:<8} | {c4:<8} | {c5}")

    lines.append("-"*70)
    lines.append("MAND = mandatory (passed ablation in that circuit)")
    lines.append("NOTE: Circuit 5 cannot test mandatoriness (ablation failed).")

    # Conclusions
    lines.append(f"\n\nCONCLUSIONS:")
    lines.append("-"*40)

    ts_status_c5 = c5_status.get('temporal_stability', 'zombie')
    any_learned = any(s in ('Learned', 'Variable') for s in c5_status.values())

    if not any_learned and len(ok_patients) > 0:
        lines.append(
            "The limbic->prefrontal transformation during passive movie watching "
            "produces sparse learned representations. Unlike active working memory "
            "tasks (Circuits 3-4) where theta and choice signals were mandatory, "
            "passive emotional processing does not force the surrogate to develop "
            "rich biological intermediates."
        )
    elif ts_status_c5 in ('Learned', 'Variable'):
        lines.append(
            "Temporal stability is the most consistently learned intermediate "
            "variable in the limbic->prefrontal emotion transformation, "
            "replicating its appearance in Circuits 3 and 4."
        )

    if len(ok_patients) <= 1:
        lines.append(
            f"\nNOTE: With only {len(ok_patients)} reliable patient(s), these are "
            f"N=1 exemplar findings, not cross-patient conclusions."
        )

    table_text = "\n".join(lines)
    print(table_text)

    out_path = RESULTS_DIR / 'circuit5_final_summary.txt'
    with open(out_path, 'w') as f:
        f.write(table_text)
    print(f"\nSaved: {out_path}")


# ================================================================
# TASK 7: FIGURES
# ================================================================
def task7_figures(reports, diag_data):
    print("\n" + "="*80)
    print("TASK 7: GENERATE FIGURES")
    print("="*80)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    try:
        import seaborn as sns
        sns.set_style('whitegrid')
        sns.set_context('paper', font_scale=1.2)
    except ImportError:
        pass

    if not reports:
        print("ERROR: No reports.")
        return

    # --- Figure 1: Patient Quality Overview ---
    print("  Figure 1: Patient Quality Overview...")
    pids_sorted = sorted(reports.keys(), key=lambda p: reports[p]['mean_cc'])
    ccs = [reports[p]['mean_cc'] for p in pids_sorted]
    colors = ['#2ecc71' if reports[p]['quality_flag'] == 'OK'
              else ('#f39c12' if reports[p]['mean_cc'] >= 0.2 else '#e74c3c')
              for p in pids_sorted]

    n_neurons = []
    for p in pids_sorted:
        fn_path = RESULTS_DIR / p / 'focused_neurons.json'
        if fn_path.exists():
            with open(fn_path) as f:
                fi = json.load(f)
            n_neurons.append(reports[p]['n_input_neurons'] + fi.get('n_output_total', 0))
        else:
            n_neurons.append(reports[p]['n_input_neurons'])

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(pids_sorted)), ccs, color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0.3, color='black', linestyle='--', linewidth=1.5, label='CC=0.3')
    ax.set_xticks(range(len(pids_sorted)))
    ax.set_xticklabels([p.replace('sub-', '') for p in pids_sorted], rotation=45, ha='right')
    ax.set_ylabel('Mean Correlation Coefficient (CC)')
    ax.set_title(f'DESCARTES Circuit 5 -- Patient Model Quality (n={len(reports)})')
    for i, (bar, nn) in enumerate(zip(bars, n_neurons)):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'n={nn}', ha='center', va='bottom', fontsize=8)
    ok_patch = mpatches.Patch(color='#2ecc71', label='OK (CC>=0.3)')
    marg_patch = mpatches.Patch(color='#f39c12', label='Marginal (0.2-0.3)')
    low_patch = mpatches.Patch(color='#e74c3c', label='Low (CC<0.2)')
    ax.legend(handles=[ok_patch, marg_patch, low_patch], loc='upper left')
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig1_patient_quality.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'fig1_patient_quality.pdf', bbox_inches='tight')
    plt.close(fig)
    print("    Saved fig1")

    # --- Figure 2: Neuron Count vs CC ---
    print("  Figure 2: Neuron Count vs CC...")
    if diag_data:
        valid_diag = [d for d in diag_data if d.get('mean_cc') is not None]
        if len(valid_diag) >= 3:
            from scipy.stats import pearsonr
            nts = [d['n_total'] for d in valid_diag]
            mccs = [d['mean_cc'] for d in valid_diag]
            cols = ['#2ecc71' if d.get('quality') == 'OK'
                    else ('#f39c12' if (d.get('mean_cc') or 0) >= 0.2 else '#e74c3c')
                    for d in valid_diag]

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(nts, mccs, c=cols, s=100, edgecolors='black', linewidth=0.5, zorder=5)
            for d in valid_diag:
                ax.annotate(d['patient'].replace('sub-', ''),
                           (d['n_total'], d['mean_cc']),
                           textcoords="offset points", xytext=(5, 5), fontsize=7)
            ax.axhline(y=0.3, color='black', linestyle='--', linewidth=1, alpha=0.7)
            ax.set_xlabel('Total Neurons (input + output)')
            ax.set_ylabel('Mean CC')
            ax.set_title('Neuron Count vs Model Quality')
            r, p = pearsonr(nts, mccs)
            ax.text(0.05, 0.95, f'r = {r:.3f}, p = {p:.4f}',
                   transform=ax.transAxes, va='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            plt.tight_layout()
            fig.savefig(FIGURES_DIR / 'fig2_neurons_vs_cc.png', dpi=300, bbox_inches='tight')
            fig.savefig(FIGURES_DIR / 'fig2_neurons_vs_cc.pdf', bbox_inches='tight')
            plt.close(fig)
            print("    Saved fig2")

    # --- Figure 3: Probing Heatmap ---
    print("  Figure 3: Probing Heatmap...")
    patient_ids = sorted(reports.keys())
    delta_matrix = np.full((len(PROBE_TARGETS), len(patient_ids)), np.nan)
    valid_matrix = np.ones((len(PROBE_TARGETS), len(patient_ids)), dtype=bool)

    for j, pid in enumerate(patient_ids):
        for i, tname in enumerate(PROBE_TARGETS):
            pr = reports[pid].get('probe_results', {}).get(tname, {})
            d = pr.get('delta')
            if d is not None and isinstance(d, (int, float)):
                delta_matrix[i, j] = d
            valid_matrix[i, j] = pr.get('valid', False)

    fig, ax = plt.subplots(figsize=(max(12, len(patient_ids)*1.2), 8))
    finite_vals = delta_matrix[np.isfinite(delta_matrix)]
    vmax = max(0.15, np.nanmax(np.abs(finite_vals))) if len(finite_vals) > 0 else 0.15
    im = ax.imshow(delta_matrix, cmap='RdBu_r', aspect='auto',
                   vmin=-vmax, vmax=vmax, interpolation='nearest')

    for i in range(len(PROBE_TARGETS)):
        for j in range(len(patient_ids)):
            val = delta_matrix[i, j]
            if np.isfinite(val):
                color = 'white' if abs(val) > vmax * 0.6 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                       fontsize=7, color=color)
                if not valid_matrix[i, j]:
                    ax.text(j, i - 0.3, 'X', ha='center', va='center',
                           fontsize=10, color='red', fontweight='bold')

    ax.set_xticks(range(len(patient_ids)))
    labels = []
    for p in patient_ids:
        q = 'OK' if reports[p]['quality_flag'] == 'OK' else 'LOW'
        labels.append(f"{p.replace('sub-', '')}\n({q})")
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks(range(len(PROBE_TARGETS)))
    ax.set_yticklabels(PROBE_TARGETS, fontsize=9)
    ax.set_title('Probing Results: dR2 / dAccuracy per Patient x Target\n(X = invalid)')
    plt.colorbar(im, ax=ax, label='dR2 (or dAccuracy)')
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig3_probing_heatmap.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'fig3_probing_heatmap.pdf', bbox_inches='tight')
    plt.close(fig)
    print("    Saved fig3")

    # --- Figure 4: Circuit Comparison ---
    print("  Figure 4: Circuit Comparison...")
    from matplotlib.colors import ListedColormap, BoundaryNorm

    status_map = {'MAND': 3, 'Learned': 2, 'Variable': 1, 'zombie': 0, 'N/A': -1}
    color_map = {3: '#27ae60', 2: '#2980b9', 1: '#f39c12', 0: '#95a5a6', -1: '#ecf0f1'}

    c5_status = {}
    ref = {pid: r for pid, r in reports.items() if r['quality_flag'] == 'OK'}
    if not ref:
        ref = reports
    for tname in PROBE_TARGETS:
        statuses = []
        for pid, r in ref.items():
            pr = r.get('probe_results', {}).get(tname, {})
            statuses.append(pr.get('status', 'N/A'))
        n_cp = sum(1 for s in statuses if s in ('STRONG', 'CANDIDATE'))
        if n_cp > len(ref) * 0.5:
            c5_status[tname] = 'Learned'
        elif n_cp > 0:
            c5_status[tname] = 'Variable'
        else:
            c5_status[tname] = 'zombie'

    targets_fig = ['theta_power', 'gamma_power', 'temporal_stability',
                   'population_synchrony', 'arousal', 'emotion_category',
                   'encoding_success', 'event_boundaries', 'threat_pe']
    circuits = ['C2:HC', 'C3:Mouse', 'C4:Human', 'C5:Emotion']

    circuit_lookup = {
        'theta_power':         ['MAND',    'MAND',    'Variable'],
        'gamma_power':         ['MAND',    'N/A',     'N/A'],
        'temporal_stability':  ['N/A',     'MAND',    'Variable'],
        'population_synchrony':['N/A',     'N/A',     'Variable'],
        'arousal':             ['N/A',     'N/A',     'N/A'],
        'emotion_category':    ['N/A',     'N/A',     'N/A'],
        'encoding_success':    ['N/A',     'N/A',     'N/A'],
        'event_boundaries':    ['N/A',     'N/A',     'N/A'],
        'threat_pe':           ['N/A',     'N/A',     'N/A'],
    }

    mat = np.zeros((len(targets_fig), len(circuits)))
    labels_mat = []
    for i, tname in enumerate(targets_fig):
        row_labels = circuit_lookup[tname] + [c5_status.get(tname, 'zombie')]
        labels_mat.append(row_labels)
        for j, status in enumerate(row_labels):
            mat[i, j] = status_map.get(status, -1)

    cmap_d = ListedColormap([color_map[-1], color_map[0], color_map[1],
                              color_map[2], color_map[3]])
    bounds = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
    norm = BoundaryNorm(bounds, cmap_d.N)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(mat, cmap=cmap_d, norm=norm, aspect='auto')
    for i in range(len(targets_fig)):
        for j in range(len(circuits)):
            label = labels_mat[i][j]
            ax.text(j, i, label, ha='center', va='center', fontsize=9,
                   fontweight='bold' if label in ('MAND', 'Learned') else 'normal')

    ax.set_xticks(range(len(circuits)))
    ax.set_xticklabels(circuits, fontsize=10)
    ax.set_yticks(range(len(targets_fig)))
    ax.set_yticklabels(targets_fig, fontsize=9)
    ax.set_title(f'DESCARTES Circuits 2-5: Intermediate Variable Status\n'
                 f'(C5: n={len(ref)} reliable patients)')

    patches = [mpatches.Patch(color=color_map[3], label='Mandatory'),
               mpatches.Patch(color=color_map[2], label='Learned'),
               mpatches.Patch(color=color_map[1], label='Variable'),
               mpatches.Patch(color=color_map[0], label='Zombie'),
               mpatches.Patch(color=color_map[-1], label='N/A')]
    ax.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, -0.05),
              ncol=5, fontsize=9)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig4_circuit_comparison.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'fig4_circuit_comparison.pdf', bbox_inches='tight')
    plt.close(fig)
    print("    Saved fig4")

    print(f"\nAll figures saved to: {FIGURES_DIR}")


# ================================================================
# MAIN
# ================================================================
def main():
    print("="*80)
    print("DESCARTES CIRCUIT 5 -- POST-PIPELINE ANALYSIS")
    print("="*80)

    n_reports = sum(1 for pid in PATIENTS
                    if (RESULTS_DIR / pid / 'patient_report.json').exists())
    print(f"\nPatient reports found: {n_reports}/{len(PATIENTS)}")

    if n_reports < len(PATIENTS):
        print(f"\nWARNING: Only {n_reports}/{len(PATIENTS)} reports found.")
        print(f"Proceeding with available data.\n")

    reports, focused_info = task1_compile_results()
    diag_data = task2_cc_diagnostic()

    window_results = None
    if (RESULTS_DIR / 'sub-CS48' / 'patient_report.json').exists():
        try:
            window_results = task3_window_level_probing()
        except Exception as e:
            print(f"  Task 3 error: {e}")
            import traceback; traceback.print_exc()

    task4_neuron_threshold(diag_data)
    task5_cs48_comprehensive(window_results)
    task6_final_summary(reports, diag_data)

    try:
        task7_figures(reports, diag_data)
    except Exception as e:
        print(f"  Task 7 error: {e}")
        import traceback; traceback.print_exc()

    print("\n" + "="*80)
    print("ALL ANALYSIS TASKS COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
