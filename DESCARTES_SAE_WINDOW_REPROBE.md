# DESCARTES — SAE Retroactive Analysis + Window-Level Re-Probing

## Task 1: SAE Retroactive Analysis

### What
Apply Sparse Autoencoders to decompose LSTM hidden states from existing
trained models. Re-probe the decomposed (monosemantic) features for 
biological variables that Ridge probing on raw hidden states missed.

### Why
Large models (h=128, h=256) look zombie under linear Ridge probing. 
But biological variables may be encoded in polysemantic superposition —
multiple variables sharing the same hidden dimensions. SAE decomposes
hidden states into overcomplete sparse features that may reveal 
biological representations invisible to linear probes.

### Which Models
Use existing checkpoints — NO retraining needed:
- Circuit 2 (CA3→CA1): h=64, h=128, h=256 if available
- Circuit 3 (ALM→Thalamus): h=128 primary
- Circuit 4 (Human WM): h=128 primary, best-CC patients
- Circuit 5 (Movie Emotion): h=128, sub-CS48 focused model

### SAE Architecture

```python
import torch
import torch.nn as nn

class SparseAutoencoder(nn.Module):
    """
    Overcomplete sparse autoencoder for hidden state decomposition.
    
    Maps hidden_size → expansion_factor * hidden_size sparse features,
    then reconstructs. The sparse features are the monosemantic 
    decomposition to probe against biological variables.
    """
    def __init__(self, hidden_size, expansion_factor=4, sparsity_coeff=1e-3):
        super().__init__()
        self.n_features = hidden_size * expansion_factor
        self.sparsity_coeff = sparsity_coeff
        
        self.encoder = nn.Linear(hidden_size, self.n_features)
        self.decoder = nn.Linear(self.n_features, hidden_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x: (batch, hidden_size)
        features = self.relu(self.encoder(x))  # sparse activations
        reconstruction = self.decoder(features)
        return reconstruction, features
    
    def loss(self, x):
        recon, features = self.forward(x)
        recon_loss = nn.functional.mse_loss(recon, x)
        sparsity_loss = features.abs().mean()  # L1 on activations
        return recon_loss + self.sparsity_coeff * sparsity_loss, recon_loss, sparsity_loss
```

### Pipeline Per Circuit

```
1. Load trained LSTM checkpoint
2. Extract hidden states on FULL dataset (not just test):
   model.eval() → forward with return_hidden=True
   Flatten to (N_total, hidden_size)

3. Train SAE:
   - expansion_factor = 4 (128 → 512 features) and 8 (128 → 1024)
   - sparsity_coeff sweep: [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
   - Train for 50 epochs, batch_size=256, lr=1e-3, Adam
   - Track: reconstruction loss, sparsity (mean L0 = avg nonzero features per sample)
   - Target: reconstruction R² > 0.95, mean L0 < 50 active features per sample

4. Extract SAE features on TEST data:
   _, features = sae(hidden_states_test)  # (N_test, n_sae_features)

5. Re-probe with Ridge ΔR²:
   For each biological probe target:
     - sanitized_ridge_delta_r2(features_trained, target, features_untrained)
     - Compare ΔR² from SAE features vs ΔR² from raw hidden states
   
   CRITICAL: Also train SAE on UNTRAINED model hidden states with same
   hyperparameters. Use untrained SAE features as the baseline.

6. Report:
   For each probe target:
     - ΔR² (raw hidden states) — already computed
     - ΔR² (SAE features) — new
     - Change: did SAE reveal a hidden representation?
   
   Key question: any target that was zombie (ΔR² < 0.05) under raw 
   probing but becomes CANDIDATE+ (ΔR² > 0.05) under SAE probing?
   That target was encoded in superposition.
```

### What To Look For

```
IF SAE reveals theta/gamma in large models where Ridge showed zombie:
  → "Biology is always encoded; representational format varies with capacity"
  → Paper narrative changes fundamentally
  → Re-run SAE on ALL circuits to map the full picture

IF SAE finds nothing new:
  → Zombie finding is real — large models genuinely don't encode biology
  → Current paper narrative is correct
  → SAE analysis becomes a supplementary control (negative result)

IF SAE reveals DIFFERENT variables than Ridge found:
  → The two methods detect different representational formats
  → Report both as complementary analyses
```

### Deliverables
- results/sae/{circuit}_{hidden_size}/sae_probing.json
- results/sae/sae_vs_ridge_comparison.txt
- Figure: heatmap of ΔR² (Ridge vs SAE) for each circuit × target

---

## Task 2: Window-Level Re-Probing of Circuits 3-4

### What
Re-probe existing trained models from Circuits 3 and 4 at window-level 
granularity (averaging hidden states within each trial/window) instead 
of bin-level. Circuit 5 showed that slow cognitive variables are 
invisible at 20ms but strong at 2-second windows.

### Which Targets to Re-Probe
Only slow-varying targets benefit from window-level probing:
- theta_modulation (envelope changes over 100s of ms)
- choice_signal (stable within delay period)  
- delay_stability (by definition a slow variable)
- population_synchrony (computed in sliding windows)

Fast-varying targets (spike rates, gamma) stay at bin level.

### Pipeline

```
For Circuit 3 (ALM→Thalamus) and Circuit 4 (Human WM):

1. Load existing hidden states from saved checkpoints
   (or re-extract if not saved — load model + test data, forward pass)

2. Reshape: (n_trials, T_per_trial, hidden_size)
   Window-level: average across T_per_trial dimension
   → (n_trials, hidden_size)

3. Compute window-level targets:
   For each slow probe target, average within each trial/window
   → (n_trials,)

4. Run sanitized_ridge_delta_r2 on window-level data
   Compare to existing bin-level ΔR²

5. Report table:
   Target           | Bin ΔR² | Window ΔR² | Change
   -----------------|---------|-----------|-------
   theta_modulation | +0.05   | ???       | ???
   choice_signal    | +0.12   | ???       | ???
   ...
```

### For Circuit 4 Specifically
Run this on ALL patients that had OK quality (CC > 0.3).
Check whether the oscillatory vs rate-based patient dichotomy 
changes at window level — some patients classified as "no theta" 
at bin level might show theta at window level.

### Deliverables
- results/window_reprobe/circuit3_comparison.json
- results/window_reprobe/circuit4_comparison.json
- results/window_reprobe/window_vs_bin_summary.txt

---

## Execution Order

1. SAE analysis on Circuit 5 sub-CS48 first (fastest — model already loaded)
2. SAE analysis on Circuit 2 (gamma_amp is the gold standard — 
   if SAE can't recover it, something is wrong with the SAE)
3. SAE analysis on Circuits 3-4
4. Window-level re-probing of Circuits 3-4
5. Summary comparison table across all methods and circuits

## Critical Reminders

- Use sanitized_ridge_delta_r2 for ALL probing (clamp R² to [-1,+1])
- SAE must be trained on BOTH trained and untrained hidden states separately
- The SAE features from the untrained model are the baseline for ΔR²
- If SAE reconstruction R² < 0.90, increase capacity or reduce sparsity
  before trusting the probing results
- Circuit 2 gamma_amp is the positive control: if SAE cannot recover it
  at any hidden size, the SAE hyperparameters need tuning
