# Sparse Autoencoder (SAE) Analysis Results

## Overview
Overcomplete sparse autoencoders applied to LSTM hidden states to discover whether mechanistically interpretable features emerge that are invisible to linear Ridge probes.

## Method
- Train SAE on LSTM hidden state trajectories (overcomplete: hidden_dim × 4)
- Extract SAE features and probe for biological targets at two timescales:
  - **Bin-level**: Per-time-bin R2 (fine temporal resolution)
  - **Window-level**: Per-trial-window R2 (coarser, more stable)
- Compare SAE features vs raw hidden states for each target

## Results by Circuit

### Circuit 5 (Movie Emotion)
- **File**: `circuit5_summary.json`, `circuit5_sub-CS48/`, `circuit5_sub-CS42/`
- Patients: sub-CS48, sub-CS42
- SAE features do NOT outperform raw hidden states for any biological target
- Conclusion: No hidden structure revealed by SAE beyond what Ridge already captures

### Circuit 4 (Human Working Memory)
- **File**: `circuit4_summary.json`, `circuit4_sub-*/`
- 6 patient-sessions analyzed (sub-10, sub-12, sub-1 across ses-1 and ses-2)
- Window-level raw probing sometimes outperforms SAE (e.g., sub-10_ses-1: firing_rate R2=0.57 raw vs 0.41 SAE)
- SAE occasionally captures trial_variance better at window level
- Overall: Mixed results, no consistent SAE advantage

### Circuit 2 (CA3 → CA1) and Circuit 3 (Thalamocortical)
- **Files**: `circuit2_h128/`, `circuit2_h256/`, `circuit3_sub-*/`
- Circuit 2: Two hidden sizes tested (128, 256)
- Circuit 3: Two subjects (sub-440956, sub-440959)

## Key Finding
SAE analysis does not reveal biologically interpretable features beyond what linear probing captures. The "mechanistic interpretability" approach does not add explanatory power to the LSTM surrogates in this neuroscience context.

## Resources
- SAE method based on Cunningham et al. (2023) overcomplete dictionary learning
- Probing targets: mean_firing_rate, population_rate, trial_variance, temporal_stability, population_synchrony
