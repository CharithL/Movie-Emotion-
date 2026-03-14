# DESCARTES Cogito: Neural Surrogate Validation Pipeline

**Can LSTM neural surrogates learn genuine neural computation, or are they just exploiting temporal autocorrelation?**

This project implements a rigorous multi-circuit validation pipeline for LSTM-based neural surrogates, testing across 4 neural circuits from simulated hippocampal models to real human intracranial recordings.

## Project Structure

```
movie emotion/
├── README.md                          # This file
├── phase1_foundation.py               # Circuit 5 LSTM (LimbicPrefrontalLSTM)
├── phase1_retroactive.py              # Statistical hardening (Phases 1-6) across all circuits
├── phases2_to_6.py                    # Extended statistical tests
├── run_sae_circuit2.py                # SAE analysis for CA3→CA1
├── run_sae_circuit3.py                # SAE analysis for thalamocortical
├── run_sae_circuit4.py                # SAE analysis for human WM
├── run_sae_analysis.py                # SAE analysis for movie emotion
│
├── generalization/                    # 5-level generalization testing
│   ├── run_generalization.py          # All 5 levels (temporal, session, patient, perturbation)
│   └── README.md
│
├── results/
│   ├── statistical_hardening/         # Phase 1-6 JSON results
│   │   ├── phase1_circuit2.json       # CA3→CA1 (1000 surrogates)
│   │   ├── phase1_circuit3.json       # Thalamocortical (real bio targets)
│   │   ├── phase1_circuit4.json       # Human WM (13 patients)
│   │   ├── phase1_sub-CS48.json       # Movie emotion
│   │   ├── phases2to6_sub-CS48.json   # Extended analysis
│   │   └── README.md
│   ├── sae/                           # Sparse autoencoder results
│   │   ├── circuit4_summary.json
│   │   ├── circuit5_summary.json
│   │   └── README.md
│   └── generalization/                # Generalization test output
│       └── generalization_report.json
│
├── DESCARTES_STATISTICAL_HARDENING_GUIDE.md
├── DESCARTES_GENERALIZATION_GUIDE.md
├── DESCARTES_SAE_WINDOW_REPROBE.md
├── DESCARTES_MOVIE_EMOTION_GUIDE.md
└── DESCARTES_CIRCUIT5_SCALING_GUIDE.md
```

## Circuits Tested

| Circuit | Description | Species | Brain Region | Data Source |
|---------|------------|---------|-------------|-------------|
| **Circuit 2** | CA3 → CA1 hippocampal | Simulated | Hippocampus | Biophysical model |
| **Circuit 3** | Thalamus → Visual Cortex | Mouse | Thalamus/V1 | DANDI 000363 |
| **Circuit 4** | Working Memory | Human | MTL + Frontal | DANDI 000469 (41 patients) |
| **Circuit 5** | Movie Emotion | Human | Limbic → Prefrontal | DANDI 000623 (20 patients) |

## Final Results Summary

### Statistical Hardening (iAAFT Significance Testing)

| Circuit | Target | p-value | Survives? |
|---------|--------|---------|-----------|
| C2 (CA3→CA1) | gamma_amp | 0.163 | NO |
| C3 (Thalamus) | theta_modulation | 0.055 | NO |
| C4 (Human WM) | trial_variance | <0.05 in 8/13 pts | PARTIAL (62%) |
| C4 (Human WM) | temporal_stability | <0.05 in 8/13 pts | PARTIAL (62%) |
| C5 (Movie) | temporal_stability | 0.000 (z=5.6) | **YES** |

**Key finding**: Only `temporal_stability` in Circuit 5 and `trial_variance/temporal_stability` in Circuit 4 survive rigorous iAAFT null testing. The gamma_amp finding in Circuit 2 — the project's original cornerstone — is **not significant** (p=0.163 with 1000 surrogates).

### SAE Analysis
- Sparse autoencoders do NOT reveal biologically interpretable features beyond linear Ridge probing
- No consistent advantage of SAE over raw hidden state probing across any circuit

### Generalization Testing (5 Levels)
| Level | Test | Status |
|-------|------|--------|
| 1 | Cross-Condition (Temporal Split) | Pending |
| 2 | Cross-Session (ses-1 → ses-2) | Pending |
| 3 | Cross-Patient (Leave-One-Out) | Pending |
| 4 | Cross-Task | NOT TESTABLE (no shared patients) |
| 5 | Perturbation Robustness | Pending |

Run generalization tests:
```bash
cd "movie emotion"
python -X utf8 generalization/run_generalization.py
```

## Datasets and Resources

### DANDI Datasets (Open Access)
- **DANDI 000363**: Mouse thalamocortical recordings with optogenetics
  - URL: https://dandiarchive.org/dandiset/000363
  - Size: ~5 GB
- **DANDI 000469**: Human single-neuron working memory (Sternberg task)
  - URL: https://dandiarchive.org/dandiset/000469
  - 41 NWB files across 17 patients, sessions 1-2
  - Size: ~15 GB total
- **DANDI 000623**: Human single-neuron movie watching (emotion/cognition)
  - URL: https://dandiarchive.org/dandiset/000623
  - 20 patients, limbic + prefrontal cortex
  - Size: ~26 GB total

### Key Dependencies
```
torch>=2.0
numpy
scipy
h5py
dandi
remfile
pynwb
scikit-learn
```

### Model Architectures
- **LimbicPrefrontalLSTM** (Circuit 5): input_proj → LSTM(128) → 2-layer MLP output
- **HumanLSTMSurrogate** (Circuit 4): Linear → LSTM(128, 2 layers, dropout=0.1) → Linear
- **PatientAgnosticSurrogate** (Generalization L3): neuron_encoder → mean pool → LSTM → neuron_decoder

### Statistical Methods
- **iAAFT Surrogates**: Iterated Amplitude Adjusted Fourier Transform — preserves power spectrum and amplitude distribution while randomizing phase
- **Matched Baselines**: Compare trained vs untrained model probing (delta_R2)
- **Gap Cross-Validation**: Time-series CV with decorrelation gaps to prevent leakage
- **Family-wise Error Correction**: Simes test at family level, then Holm step-down

## Running on Vast.ai

See setup instructions below for cloud GPU execution.

```bash
# Clone and setup
git clone https://github.com/CharithL/Movie-Emotion-.git
cd Movie-Emotion-

# Install dependencies
pip install torch numpy scipy h5py dandi remfile pynwb scikit-learn

# Download NWB data from DANDI
python -c "
from dandi.dandiapi import DandiAPIClient
import urllib.request
from pathlib import Path

client = DandiAPIClient()

# Circuit 5 (Movie Emotion)
ds623 = client.get_dandiset('000623')
data_dir = Path('data/000623')
for asset in ds623.get_assets():
    if 'CS48' in asset.path:
        local = data_dir / asset.path
        local.parent.mkdir(parents=True, exist_ok=True)
        if not local.exists():
            url = asset.get_content_url(follow_redirects=1, strip_query=True)
            print(f'Downloading {asset.path}...')
            urllib.request.urlretrieve(url, str(local))
print('Done')
"

# Run generalization tests (GPU enabled)
ALLOW_GPU=1 python -X utf8 generalization/run_generalization.py
```

## License
Research use only. Data from DANDI Archive under their respective licenses.
