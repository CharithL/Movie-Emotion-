# DESCARTES Generalization Testing — Claude Code Guide

## Purpose

Test whether trained LSTM surrogates can predict neural activity for
completely novel inputs they never saw during training. This determines
whether the surrogates learned the biological transformation itself
or merely memorized the training distribution.

Five levels of generalization, ordered from easiest to hardest.
Each level that passes increases confidence that the surrogate
is a real prosthetic candidate, not a lookup table.

## Key Metric

```
Generalization Ratio = CC_novel / CC_trained

> 0.8  = STRONG generalization (prosthetic-grade)
0.5-0.8 = PARTIAL generalization (needs improvement)
< 0.5  = FAILURE (memorized training distribution)
```

Report this ratio for every test. Raw CC on novel data is meaningless
without the trained-data reference.

---

## Level 1: Cross-Condition Generalization

### Circuit 4 (Human WM — Sternberg Task)

The Sternberg task has different memory loads (1, 2, 3 items).
Train on two loads, test on the third. The neural dynamics during
load-3 are qualitatively different from load-1 (sustained firing,
different delay activity, stronger theta). If the surrogate predicts
load-3 output from load-3 input after only training on loads 1-2,
it learned the transformation, not the specific activity pattern.

```python
"""
generalization/level1_cross_condition.py

Cross-condition generalization for Circuit 4 Sternberg WM.
"""
import numpy as np
import torch
import json
from pathlib import Path


def extract_trials_by_load(nwb_filepath, region_column='electrode_group'):
    """
    Extract trials grouped by memory load from Sternberg task NWB.
    
    Returns:
        dict: {load_value: {'X': (n_trials, T, n_input), 
                             'Y': (n_trials, T, n_output)}}
    """
    import pynwb
    
    with pynwb.NWBHDF5IO(str(nwb_filepath), 'r') as io:
        nwb = io.read()
        
        # Get load column from trials table
        # Column name varies — check: 'loads', 'set_size', 'n_items', 'load'
        load_col = None
        for candidate in ['loads', 'set_size', 'n_items', 'load', 'num_items']:
            if candidate in nwb.trials.colnames:
                load_col = candidate
                break
        
        if load_col is None:
            print(f"Available trial columns: {nwb.trials.colnames}")
            raise ValueError("Cannot find memory load column in trials table")
        
        # Group trials by load
        trials_by_load = {}
        for i in range(len(nwb.trials)):
            load = int(nwb.trials[load_col][i])
            if load not in trials_by_load:
                trials_by_load[load] = []
            trials_by_load[load].append(i)
        
        print(f"Load distribution: { {k: len(v) for k, v in trials_by_load.items()} }")
    
    return trials_by_load


def cross_condition_test(model_class, model_kwargs, 
                          preprocessed_data, trial_indices_by_load,
                          train_loads, test_load,
                          n_seeds=5, n_epochs=200, patience=20,
                          device='cpu'):
    """
    Train on trials from train_loads, test on trials from test_load.
    
    Args:
        model_class: LSTM surrogate class
        model_kwargs: constructor args
        preprocessed_data: dict with 'X', 'Y' full arrays
        trial_indices_by_load: {load: [trial_indices]}
        train_loads: list of loads to train on (e.g., [1, 2])
        test_load: load to test on (e.g., 3)
        n_seeds: number of random seeds
    
    Returns:
        dict with CC on training conditions and CC on novel condition
    """
    X_full = preprocessed_data['X']  # (n_trials, T, n_input)
    Y_full = preprocessed_data['Y']  # (n_trials, T, n_output)
    
    # Split trials
    train_idx = []
    for load in train_loads:
        train_idx.extend(trial_indices_by_load[load])
    test_idx = trial_indices_by_load[test_load]
    
    X_train = X_full[train_idx]
    Y_train = Y_full[train_idx]
    X_test_novel = X_full[test_idx]
    Y_test_novel = Y_full[test_idx]
    
    # Also hold out some training-condition trials for reference CC
    np.random.seed(42)
    n_train = len(train_idx)
    perm = np.random.permutation(n_train)
    split = int(0.8 * n_train)
    
    X_tr = X_train[perm[:split]]
    Y_tr = Y_train[perm[:split]]
    X_val_same = X_train[perm[split:]]
    Y_val_same = Y_train[perm[split:]]
    
    results_per_seed = []
    
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = model_class(**model_kwargs).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()
        
        X_tr_t = torch.FloatTensor(X_tr).to(device)
        Y_tr_t = torch.FloatTensor(Y_tr).to(device)
        
        # Training loop
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(n_epochs):
            model.train()
            optimizer.zero_grad()
            y_pred = model(X_tr_t)
            loss = criterion(y_pred, Y_tr_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        model.load_state_dict(best_state)
        model.eval()
        
        with torch.no_grad():
            # CC on same-condition held-out trials
            y_same = model(torch.FloatTensor(X_val_same).to(device)).cpu().numpy()
            cc_same = compute_cc(y_same, Y_val_same)
            
            # CC on novel condition
            y_novel = model(torch.FloatTensor(X_test_novel).to(device)).cpu().numpy()
            cc_novel = compute_cc(y_novel, Y_test_novel)
        
        gen_ratio = cc_novel / max(cc_same, 0.01)
        
        results_per_seed.append({
            'seed': seed,
            'cc_same_condition': float(cc_same),
            'cc_novel_condition': float(cc_novel),
            'generalization_ratio': float(gen_ratio),
        })
        
        print(f"  Seed {seed}: CC_same={cc_same:.3f}, "
              f"CC_novel={cc_novel:.3f}, ratio={gen_ratio:.3f}")
    
    # Aggregate
    cc_same_all = [r['cc_same_condition'] for r in results_per_seed]
    cc_novel_all = [r['cc_novel_condition'] for r in results_per_seed]
    ratios = [r['generalization_ratio'] for r in results_per_seed]
    
    mean_ratio = float(np.mean(ratios))
    status = ('STRONG' if mean_ratio > 0.8 else 
              'PARTIAL' if mean_ratio > 0.5 else 'FAILURE')
    
    return {
        'train_loads': train_loads,
        'test_load': test_load,
        'n_train_trials': len(train_idx),
        'n_test_trials': len(test_idx),
        'cc_same_mean': float(np.mean(cc_same_all)),
        'cc_same_std': float(np.std(cc_same_all)),
        'cc_novel_mean': float(np.mean(cc_novel_all)),
        'cc_novel_std': float(np.std(cc_novel_all)),
        'generalization_ratio_mean': mean_ratio,
        'generalization_ratio_std': float(np.std(ratios)),
        'status': status,
        'per_seed': results_per_seed,
    }


def compute_cc(y_pred, y_true):
    """Mean per-neuron Pearson correlation across time and trials."""
    # Flatten: (n_trials, T, n_output) → (N, n_output)
    pred_flat = y_pred.reshape(-1, y_pred.shape[-1])
    true_flat = y_true.reshape(-1, y_true.shape[-1])
    
    ccs = []
    for j in range(pred_flat.shape[1]):
        r = np.corrcoef(pred_flat[:, j], true_flat[:, j])[0, 1]
        if not np.isnan(r):
            ccs.append(r)
    
    return float(np.mean(ccs)) if ccs else 0.0


def run_all_condition_splits(model_class, model_kwargs,
                              preprocessed_data, trial_indices_by_load,
                              device='cpu'):
    """
    Test all leave-one-condition-out splits.
    
    For loads [1, 2, 3]:
      Train on [1,2], test on 3
      Train on [1,3], test on 2
      Train on [2,3], test on 1
    """
    loads = sorted(trial_indices_by_load.keys())
    print(f"Available loads: {loads}")
    
    all_results = []
    
    for test_load in loads:
        train_loads = [l for l in loads if l != test_load]
        print(f"\n{'='*60}")
        print(f"Train on loads {train_loads}, test on load {test_load}")
        print(f"{'='*60}")
        
        result = cross_condition_test(
            model_class, model_kwargs,
            preprocessed_data, trial_indices_by_load,
            train_loads, test_load,
            device=device
        )
        all_results.append(result)
    
    # Summary
    ratios = [r['generalization_ratio_mean'] for r in all_results]
    print(f"\n{'='*60}")
    print(f"CROSS-CONDITION SUMMARY")
    print(f"{'='*60}")
    for r in all_results:
        print(f"  Test load {r['test_load']}: "
              f"ratio={r['generalization_ratio_mean']:.3f} "
              f"({r['status']})")
    print(f"  Mean ratio: {np.mean(ratios):.3f}")
    
    return all_results
```

### Circuit 5 (Movie Emotion — Temporal Split)

The movie has distinct emotional phases. Train on first 5 minutes
(building tension), test on last 3 minutes (climax and resolution).

```python
def cross_phase_movie_test(model_class, model_kwargs,
                            preprocessed_data,
                            train_fraction=0.6,
                            device='cpu'):
    """
    Train on early movie, test on late movie.
    
    The emotional content changes dramatically — early scenes establish
    characters, later scenes have peak suspense and resolution.
    This tests whether the surrogate learned the limbic→prefrontal
    transformation or just the specific activity during calm scenes.
    """
    X = preprocessed_data['X']  # (n_windows, T, n_input)
    Y = preprocessed_data['Y']
    
    n_total = len(X)
    n_train = int(train_fraction * n_total)
    
    # Strict temporal split — no overlap
    X_train = X[:n_train]
    Y_train = Y[:n_train]
    
    # Hold out a validation set from training period
    n_val = int(0.2 * n_train)
    X_tr = X_train[:n_train - n_val]
    Y_tr = Y_train[:n_train - n_val]
    X_val_same = X_train[n_train - n_val:]
    Y_val_same = Y_train[n_train - n_val:]
    
    # Novel test: the unseen movie phase
    X_test_novel = X[n_train:]
    Y_test_novel = Y[n_train:]
    
    # Train and evaluate (same structure as Level 1)
    # ... (reuse training loop from cross_condition_test)
    
    return {
        'train_windows': n_train - n_val,
        'val_same_windows': n_val,
        'test_novel_windows': len(X_test_novel),
        'train_phase': f'minutes 0-{train_fraction * 8:.1f}',
        'test_phase': f'minutes {train_fraction * 8:.1f}-8',
        # ... CC and ratio results
    }
```

---

## Level 2: Cross-Session Generalization

### For Patients with Multiple Sessions

Some Circuit 4 patients have two recording sessions. The neural
population may drift between sessions — baseline rates change,
some neurons appear or disappear. This tests whether the surrogate
learned a transformation that's stable across population changes.

```python
"""
generalization/level2_cross_session.py

Train on session 1, test on session 2 for the same patient.
"""

def cross_session_test(model_class, model_kwargs,
                        session1_data, session2_data,
                        patient_id, device='cpu'):
    """
    Train surrogate on session 1, evaluate on session 2.
    
    CRITICAL ISSUE: The neuron populations may differ between sessions.
    Neurons are identified by electrode/channel, not by identity.
    Between sessions, a neuron on electrode 5 might be a different cell.
    
    Two approaches:
    A) Matched neurons: only use neurons present in both sessions
       (identified by electrode + similar waveform). This is conservative
       but may lose many neurons.
    B) Population-level: treat each session's population as a bag of
       neurons. The surrogate must generalize across population changes.
       This requires a model that takes variable-size input/output.
    
    For approach A (simpler, recommended first):
    """
    # Find matched electrodes between sessions
    electrodes_s1 = set(session1_data['input_electrode_ids'])
    electrodes_s2 = set(session2_data['input_electrode_ids'])
    shared_input = sorted(electrodes_s1 & electrodes_s2)
    
    electrodes_s1_out = set(session1_data['output_electrode_ids'])
    electrodes_s2_out = set(session2_data['output_electrode_ids'])
    shared_output = sorted(electrodes_s1_out & electrodes_s2_out)
    
    print(f"Patient {patient_id}:")
    print(f"  Session 1: {len(electrodes_s1)} input, {len(electrodes_s1_out)} output")
    print(f"  Session 2: {len(electrodes_s2)} input, {len(electrodes_s2_out)} output")
    print(f"  Shared: {len(shared_input)} input, {len(shared_output)} output")
    
    if len(shared_input) < 3 or len(shared_output) < 3:
        return {
            'patient_id': patient_id,
            'status': 'INSUFFICIENT_OVERLAP',
            'shared_input': len(shared_input),
            'shared_output': len(shared_output),
        }
    
    # Extract shared neurons from both sessions
    # Map electrode IDs to column indices
    s1_input_idx = [session1_data['input_electrode_ids'].index(e) 
                    for e in shared_input]
    s2_input_idx = [session2_data['input_electrode_ids'].index(e) 
                    for e in shared_input]
    s1_output_idx = [session1_data['output_electrode_ids'].index(e) 
                     for e in shared_output]
    s2_output_idx = [session2_data['output_electrode_ids'].index(e) 
                     for e in shared_output]
    
    X_train = session1_data['X'][:, :, s1_input_idx]
    Y_train = session1_data['Y'][:, :, s1_output_idx]
    X_test = session2_data['X'][:, :, s2_input_idx]
    Y_test = session2_data['Y'][:, :, s2_output_idx]
    
    # Update model kwargs for reduced neuron count
    kwargs = model_kwargs.copy()
    kwargs['n_input'] = len(shared_input)
    kwargs['n_output'] = len(shared_output)
    
    # Train on session 1, evaluate on both
    # ... (reuse training loop)
    # Report CC_session1 (held-out), CC_session2, ratio
    
    return {
        'patient_id': patient_id,
        'n_shared_input': len(shared_input),
        'n_shared_output': len(shared_output),
        # 'cc_session1': ...,
        # 'cc_session2': ...,
        # 'generalization_ratio': ...,
        # 'status': ...,
    }
```

---

## Level 3: Cross-Patient Generalization

### Leave-One-Patient-Out

Train on 12 patients, test on the 13th. This is the clinically
relevant test — can we deploy a pre-trained prosthetic?

```python
"""
generalization/level3_cross_patient.py

Leave-one-patient-out cross-validation.

MAJOR CHALLENGE: Different patients have different numbers of neurons.
Patient A has 20 input neurons, patient B has 50. The LSTM expects
a fixed input dimension.

SOLUTION: Population embedding layer.
Instead of feeding raw neuron × time matrices, embed each neuron's
activity into a fixed-dimensional space, then average across neurons.
This creates a patient-agnostic representation.
"""
import torch
import torch.nn as nn
import numpy as np


class PatientAgnosticSurrogate(nn.Module):
    """
    Surrogate that handles variable numbers of input/output neurons
    across patients via population embedding.
    
    Each neuron's time series is projected to a shared embedding space,
    then averaged across neurons to create a fixed-size population vector.
    """
    
    def __init__(self, embed_dim=32, hidden_size=128, n_layers=2):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        
        # Each neuron's single time bin → embed_dim
        self.neuron_encoder = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
        # Population vector → LSTM
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
        )
        
        # LSTM → population output embedding
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, embed_dim),
            nn.ReLU(),
        )
        
        # Output embedding → per-neuron prediction
        self.neuron_decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )
    
    def forward(self, x, n_output_neurons=None):
        """
        Args:
            x: (batch, time, n_input_neurons) — n_input varies per patient
            n_output_neurons: int — varies per patient
        
        Returns:
            y: (batch, time, n_output_neurons)
        """
        batch, time, n_input = x.shape
        
        # Embed each neuron independently
        # x_flat: (batch * time * n_input, 1)
        x_flat = x.reshape(-1, 1)
        embedded = self.neuron_encoder(x_flat)  # (B*T*N, embed_dim)
        embedded = embedded.reshape(batch, time, n_input, self.embed_dim)
        
        # Average across neurons → population vector
        pop_vector = embedded.mean(dim=2)  # (batch, time, embed_dim)
        
        # LSTM
        lstm_out, _ = self.lstm(pop_vector)  # (batch, time, hidden)
        
        # Project to output embedding
        out_embed = self.output_proj(lstm_out)  # (batch, time, embed_dim)
        
        # Decode to each output neuron
        # Expand embed to (batch, time, n_output, embed_dim)
        if n_output_neurons is None:
            n_output_neurons = n_input  # default: same as input
        
        out_expand = out_embed.unsqueeze(2).expand(-1, -1, n_output_neurons, -1)
        y = self.neuron_decoder(out_expand).squeeze(-1)  # (batch, time, n_output)
        
        return y


def leave_one_patient_out(patient_data_dict, embed_dim=32, 
                           hidden_size=128, n_epochs=100,
                           device='cpu'):
    """
    Train on all patients except one, test on held-out patient.
    
    Args:
        patient_data_dict: {patient_id: {'X': array, 'Y': array}}
    
    Returns:
        list of per-patient generalization results
    """
    patient_ids = sorted(patient_data_dict.keys())
    results = []
    
    for test_patient in patient_ids:
        print(f"\n{'='*60}")
        print(f"Test patient: {test_patient}")
        print(f"{'='*60}")
        
        train_patients = [p for p in patient_ids if p != test_patient]
        
        # Create model
        model = PatientAgnosticSurrogate(
            embed_dim=embed_dim,
            hidden_size=hidden_size,
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        # Training: iterate over patients
        best_loss = float('inf')
        
        for epoch in range(n_epochs):
            model.train()
            epoch_loss = 0
            
            for patient in train_patients:
                X = torch.FloatTensor(patient_data_dict[patient]['X']).to(device)
                Y = torch.FloatTensor(patient_data_dict[patient]['Y']).to(device)
                n_out = Y.shape[-1]
                
                optimizer.zero_grad()
                y_pred = model(X, n_output_neurons=n_out)
                loss = criterion(y_pred, Y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
            
            epoch_loss /= len(train_patients)
            
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
        
        model.load_state_dict(best_state)
        model.eval()
        
        # Evaluate on held-out patient
        with torch.no_grad():
            X_test = torch.FloatTensor(patient_data_dict[test_patient]['X']).to(device)
            Y_test = patient_data_dict[test_patient]['Y']
            n_out = Y_test.shape[-1]
            
            y_pred = model(X_test, n_output_neurons=n_out).cpu().numpy()
            cc_novel = compute_cc(y_pred, Y_test)
        
        # Reference: CC when this patient's data IS in training
        # (use the within-patient CC from the original pipeline)
        cc_trained = patient_data_dict[test_patient].get('within_patient_cc', None)
        
        gen_ratio = cc_novel / max(cc_trained, 0.01) if cc_trained else None
        
        result = {
            'test_patient': test_patient,
            'n_train_patients': len(train_patients),
            'cc_novel': float(cc_novel),
            'cc_trained_reference': cc_trained,
            'generalization_ratio': float(gen_ratio) if gen_ratio else None,
        }
        results.append(result)
        
        print(f"  CC_novel = {cc_novel:.3f}"
              + (f", ratio = {gen_ratio:.3f}" if gen_ratio else ""))
    
    return results
```

---

## Level 4: Cross-Task Generalization

### Train on WM (Circuit 4), Test on Emotion (Circuit 5)

Both circuits involve the limbic→prefrontal transformation.
If the surrogate generalizes across tasks, it learned the
anatomical transformation, not the task-specific computation.

```python
"""
generalization/level4_cross_task.py

Train on working memory data, test on movie watching data.
Same brain regions, different cognitive demands.

REQUIREMENT: Find patients recorded in BOTH tasks, or use the
population embedding approach to handle different neuron sets.

The Rutishauser lab recorded the same patients across multiple
paradigms. Check if any patients appear in both DANDI 000576 
(Sternberg WM) and DANDI 000623 (movie watching).
"""

def find_shared_patients(wm_inventory, movie_inventory):
    """
    Find patients who appear in both WM and movie datasets.
    Match by subject ID pattern.
    """
    wm_ids = set(wm_inventory.keys())
    movie_ids = set(movie_inventory.keys())
    
    shared = wm_ids & movie_ids
    print(f"WM patients: {len(wm_ids)}")
    print(f"Movie patients: {len(movie_ids)}")
    print(f"Shared: {len(shared)}")
    
    if not shared:
        print("No shared patients. Using population embedding approach.")
    
    return shared


def cross_task_test(model_class, model_kwargs,
                     wm_data, movie_data,
                     patient_id, device='cpu'):
    """
    Train surrogate on WM task data, test on movie task data.
    
    Both datasets must be preprocessed to the same format:
    same bin width, same neuron ordering for shared neurons.
    
    If same patient is in both datasets:
      Use matched neurons (same electrode channels).
    
    If different patients:
      Use PatientAgnosticSurrogate with population embedding.
    """
    # Find shared neurons (same electrodes in both tasks)
    shared_input, shared_output = find_shared_neurons(
        wm_data['input_electrode_ids'],
        wm_data['output_electrode_ids'],
        movie_data['input_electrode_ids'],
        movie_data['output_electrode_ids']
    )
    
    if len(shared_input) < 3 or len(shared_output) < 3:
        return {'status': 'INSUFFICIENT_OVERLAP'}
    
    # Train on WM with shared neurons only
    # Test on movie with shared neurons only
    # Report CC and generalization ratio
    
    pass  # Implementation follows same pattern as Level 2


def find_shared_neurons(wm_input_ids, wm_output_ids,
                         movie_input_ids, movie_output_ids):
    """Find electrode IDs present in both tasks."""
    shared_in = sorted(set(wm_input_ids) & set(movie_input_ids))
    shared_out = sorted(set(wm_output_ids) & set(movie_output_ids))
    return shared_in, shared_out
```

---

## Level 5: Perturbation Robustness

### Input Corruption Tests

Even without novel task conditions, test whether the surrogate
degrades gracefully when the input is corrupted in ways that
mimic real-world prosthetic failure modes.

```python
"""
generalization/level5_perturbation.py

Test robustness to input corruptions that a real prosthetic would face:
- Neuron dropout (electrode failure)
- Noise injection (electrical interference)
- Gain drift (signal amplitude changes over time)
- Temporal jitter (timing misalignment)
"""
import numpy as np
import torch


def neuron_dropout_test(model, X_test, Y_test, 
                         dropout_fractions=[0.1, 0.2, 0.3, 0.5],
                         n_repeats=20, device='cpu'):
    """
    Randomly zero out a fraction of input neurons and measure CC.
    
    This simulates electrode failure — a real prosthetic must handle
    losing some input channels without catastrophic failure.
    
    If CC drops <20% with 30% neuron dropout, the model is robust.
    """
    model.eval()
    n_input = X_test.shape[-1]
    
    # Baseline CC (no dropout)
    with torch.no_grad():
        y_base = model(torch.FloatTensor(X_test).to(device)).cpu().numpy()
    cc_baseline = compute_cc(y_base, Y_test)
    
    results = {'baseline_cc': float(cc_baseline), 'dropout_tests': []}
    
    for frac in dropout_fractions:
        n_drop = int(frac * n_input)
        ccs = []
        
        for rep in range(n_repeats):
            X_corrupt = X_test.copy()
            drop_idx = np.random.choice(n_input, n_drop, replace=False)
            X_corrupt[:, :, drop_idx] = 0.0
            
            with torch.no_grad():
                y_pred = model(torch.FloatTensor(X_corrupt).to(device)).cpu().numpy()
            cc = compute_cc(y_pred, Y_test)
            ccs.append(cc)
        
        retention = np.mean(ccs) / max(cc_baseline, 0.01)
        
        results['dropout_tests'].append({
            'dropout_fraction': frac,
            'n_neurons_dropped': n_drop,
            'cc_mean': float(np.mean(ccs)),
            'cc_std': float(np.std(ccs)),
            'retention': float(retention),
            'robust': retention > 0.8,
        })
        
        print(f"  Dropout {frac:.0%}: CC={np.mean(ccs):.3f} "
              f"(retention={retention:.3f})")
    
    return results


def noise_injection_test(model, X_test, Y_test,
                          snr_levels=[20, 10, 5, 2, 1],
                          n_repeats=20, device='cpu'):
    """
    Add Gaussian noise at various SNR levels.
    
    Real electrodes pick up electrical noise. The surrogate must
    produce reasonable output even with noisy input.
    
    SNR=10 is typical for good recordings. SNR=2 is very noisy.
    """
    model.eval()
    signal_power = np.mean(X_test ** 2)
    
    # Baseline
    with torch.no_grad():
        y_base = model(torch.FloatTensor(X_test).to(device)).cpu().numpy()
    cc_baseline = compute_cc(y_base, Y_test)
    
    results = {'baseline_cc': float(cc_baseline), 'noise_tests': []}
    
    for snr in snr_levels:
        noise_power = signal_power / snr
        ccs = []
        
        for rep in range(n_repeats):
            noise = np.random.randn(*X_test.shape) * np.sqrt(noise_power)
            X_noisy = X_test + noise.astype(np.float32)
            
            with torch.no_grad():
                y_pred = model(torch.FloatTensor(X_noisy).to(device)).cpu().numpy()
            cc = compute_cc(y_pred, Y_test)
            ccs.append(cc)
        
        retention = np.mean(ccs) / max(cc_baseline, 0.01)
        
        results['noise_tests'].append({
            'snr': snr,
            'cc_mean': float(np.mean(ccs)),
            'cc_std': float(np.std(ccs)),
            'retention': float(retention),
        })
        
        print(f"  SNR={snr}: CC={np.mean(ccs):.3f} "
              f"(retention={retention:.3f})")
    
    return results


def gain_drift_test(model, X_test, Y_test,
                     drift_magnitudes=[0.1, 0.2, 0.5, 1.0],
                     device='cpu'):
    """
    Apply slow gain drift across the recording.
    
    Over hours, electrode impedance changes, causing slow multiplicative
    drift in signal amplitude. The surrogate must handle this.
    
    Apply a linear ramp to each neuron's gain: gain(t) = 1 + magnitude * t/T
    """
    model.eval()
    
    with torch.no_grad():
        y_base = model(torch.FloatTensor(X_test).to(device)).cpu().numpy()
    cc_baseline = compute_cc(y_base, Y_test)
    
    results = {'baseline_cc': float(cc_baseline), 'drift_tests': []}
    
    n_windows, T, n_input = X_test.shape
    
    for magnitude in drift_magnitudes:
        X_drifted = X_test.copy()
        
        # Linear gain ramp across windows (not within windows)
        for w in range(n_windows):
            gain = 1.0 + magnitude * (w / n_windows)
            # Each neuron gets a slightly different drift rate
            per_neuron_gain = gain * (1.0 + 0.1 * np.random.randn(n_input))
            X_drifted[w] *= per_neuron_gain.astype(np.float32)
        
        with torch.no_grad():
            y_pred = model(torch.FloatTensor(X_drifted).to(device)).cpu().numpy()
        cc = compute_cc(y_pred, Y_test)
        retention = cc / max(cc_baseline, 0.01)
        
        results['drift_tests'].append({
            'magnitude': magnitude,
            'cc': float(cc),
            'retention': float(retention),
        })
        
        print(f"  Drift {magnitude}: CC={cc:.3f} "
              f"(retention={retention:.3f})")
    
    return results
```

---

## Execution Plan

### Run Order (total ~2-3 hours)

```
LEVEL 1: Cross-Condition (~45 min)
  For Circuit 4:
    For each of the 7 patients with CC > 0.3:
      Extract trials by memory load
      Run leave-one-load-out (3 splits per patient)
      Report generalization ratio per patient per split
  
  For Circuit 5 CS48:
      Train on first 60% of movie, test on last 40%
      Report ratio
  
  → CHECKPOINT: If mean ratio > 0.8, proceed. If < 0.5, stop.

LEVEL 2: Cross-Session (~30 min)
  For Circuit 4 patients with 2 sessions:
    Find shared neurons between sessions
    Train on session 1, test on session 2
    Report ratio
  
  → CHECKPOINT: Does the transformation survive population drift?

LEVEL 3: Cross-Patient (~45 min)
  Leave-one-patient-out across Circuit 4
  Using PatientAgnosticSurrogate with population embedding
  Report per-patient CC_novel and ratio
  
  → CHECKPOINT: Can a generic prosthetic work without calibration?

LEVEL 4: Cross-Task (~30 min)
  Check for shared patients between DANDI 000576 and 000623
  If found: train on WM, test on movie (same patient)
  If not: use population embedding across patients
  
  → CHECKPOINT: Does the anatomical transformation generalize across tasks?

LEVEL 5: Perturbation Robustness (~15 min)
  For Circuit 4 best patient (sub-10):
    Neuron dropout at 10%, 20%, 30%, 50%
    Noise injection at SNR = 20, 10, 5, 2, 1
    Gain drift at 10%, 20%, 50%, 100%
  
  → CHECKPOINT: Is the surrogate robust enough for real-world deployment?
```

### Final Report

```
Save results/generalization_report.json with:

{
  "level1_cross_condition": {
    "circuit4": {per-patient results},
    "circuit5": {temporal split results},
    "mean_ratio": float,
    "status": "STRONG/PARTIAL/FAILURE"
  },
  "level2_cross_session": {
    per-patient results for patients with 2 sessions
  },
  "level3_cross_patient": {
    leave-one-out results, mean CC_novel
  },
  "level4_cross_task": {
    WM→movie results if shared patients found
  },
  "level5_perturbation": {
    dropout, noise, drift robustness curves
  },
  "overall_assessment": {
    "prosthetic_viable": bool,
    "strongest_generalization_level": int,
    "weakest_generalization_level": int,
    "recommendation": str
  }
}
```

---

## Critical Reminders

1. The generalization RATIO is the metric, not raw CC.
   CC_novel = 0.3 is terrible if CC_trained = 0.9 (ratio 0.33)
   but acceptable if CC_trained = 0.4 (ratio 0.75).

2. Cross-patient (Level 3) requires the PatientAgnosticSurrogate
   because neuron counts differ across patients. Do not try to
   train a fixed-dimension LSTM across patients with different
   neuron counts.

3. Cross-task (Level 4) is the most informative test for prosthetics.
   If a surrogate trained on working memory can predict emotion
   processing output, the device works across cognitive contexts
   without retraining. This is what a real prosthetic needs.

4. Level 5 perturbation tests are engineering requirements, not
   scientific questions. A prosthetic that fails at 30% neuron
   dropout is clinically unusable regardless of its generalization.

5. If Level 1 fails (ratio < 0.5), do not proceed to higher levels.
   The model is a lookup table, not a transformation learner.
   Go back to architecture design before testing harder generalization.

6. Apply the corrected statistical pipeline (iAAFT, matched baselines)
   to any probing done on generalized models. The autocorrelation
   problem affects generalization testing too.
