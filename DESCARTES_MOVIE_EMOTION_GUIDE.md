# Circuit 5: Limbic→Prefrontal Emotion Transformation — Claude Code Implementation Guide

## DANDI 000623: Movie Watching (Keles et al. 2024)

---

## Purpose

Apply the DESCARTES probing and ablation framework to the **limbic→prefrontal transformation during naturalistic emotion processing**. This is Circuit 5 in the DESCARTES universality program, extending from single-neuron biophysics (L5PC), simulated hippocampal memory (CA3→CA1), mouse thalamocortical working memory (ALM→Thalamus, DANDI 000363), and human declarative working memory (MTL→Frontal, DANDI 000576) into **emotion, social cognition, and narrative processing** — the four cognitive domains not yet tested.

The dataset is the Rutishauser/Adolphs multimodal movie watching dataset (DANDI 000623), containing single-neuron recordings from amygdala, hippocampus, ACC, preSMA, and vmPFC in 20 neurosurgical patients watching an 8-minute suspense film. The transformation to model is:

```
INPUT:   Amygdala + Hippocampus (emotional encoding, memory, threat detection)
OUTPUT:  Medial Frontal Cortex — ACC + preSMA + vmPFC (regulation, appraisal, action selection)
```

This is the limbic→prefrontal axis: the pathway through which raw emotional signals are transformed into regulated cognitive appraisals. If the DESCARTES framework reveals mandatory intermediate variables in this transformation, it identifies the computational primitives of emotional processing — what the brain *cannot skip* when transforming fear into action.

---

## What This Experiment Tests

### The Cross-Domain Universality Question

The DESCARTES program has established a gradient of mandatory variable density across circuits:

```
Circuit 1 — L5PC single neuron:       0% mandatory (universal zombie)
Circuit 2 — Hippocampal CA3→CA1:       ~4% mandatory (gamma_amp only)
Circuit 3 — Mouse ALM→Thalamus WM:    ~50% mandatory (theta, choice, stability)
Circuit 4 — Human MTL→Frontal WM:     57% of patients have ≥1, but individually variable
Circuit 5 — THIS EXPERIMENT:           ???
```

The critical question: **do the same mandatory variables appear across cognitive domains, or are they domain-specific?** Specifically:

**Theta modulation** has appeared as mandatory in Circuits 2, 3, and some Circuit 4 patients. If theta is mandatory here — in an emotion/narrative task with no explicit working memory demand — it would constitute strong evidence that theta oscillation is a **universal computational signature of the limbic-prefrontal axis**, not specific to memory encoding or working memory maintenance.

**Gamma amplitude** was the sole mandatory variable in Circuit 2 (hippocampal memory). If gamma also appears as mandatory in the hippocampal input during movie watching, it suggests gamma is a mandatory computation for hippocampal output regardless of the downstream task.

### The Continuous Cognition Challenge

Unlike Circuits 3 and 4 (discrete trial structure with clear encode/delay/probe phases), movie watching is **continuous**. There are no explicit trials during the 8-minute film. This creates both a methodological challenge (how to define analysis windows) and a scientific opportunity (naturalistic processing may recruit different mandatory variables than constrained laboratory tasks).

The recognition memory test after the movie provides discrete trials that can serve as a validation bridge to Circuit 4.

### What Success Looks Like

```
                          Theta      Gamma      Emotion-    Social     Memory
                          Mandatory  Mandatory  Specific    Vars       Encoding
                                                Vars        Mandatory  Success
──────────────────────────────────────────────────────────────────────────────
Cross-domain universal:   YES        YES        Maybe       No         Partial
Domain-specific:          No         No         YES         YES        No
Mixed (most likely):      YES        Partial    Some        Some       Variable
```

---

## Dataset Overview

### Paper and Data Source

**Paper:** Keles, U., Dubois, J., Le, K.J.M., Tyszka, J.M., Kahn, D.A., Reed, C.M., Chung, J.M., Mamelak, A.N., Adolphs, R. and Rutishauser, U. (2024). "Multimodal single-neuron, intracranial EEG, and fMRI brain responses during movie watching in human patients." *Scientific Data* 11:214.

**DANDI:** https://dandiarchive.org/dandiset/000623
**Code:** https://github.com/rutishauserlab/bmovie-release-NWB-BIDS
**Annotations:** Available in the GitHub repository under `assets/annotations/`

### Contents

The dataset contains recordings from **20 neurosurgical epilepsy patients** with hybrid depth electrodes:

**Neural data (per patient NWB file):**
- Spike times of all sorted single neurons (~1,450 total across all patients)
- LFP from microwires, downsampled to 1000 Hz (bandpass 0.1–500 Hz)
- iEEG from macroelectrodes, downsampled to 1000 Hz
- Spike sorting quality metrics
- Electrode locations (brain region assignments)

**Behavioral data:**
- Recognition memory test responses (old/new confidence judgments on movie frames)
- Eye tracking data (gaze position, pupil diameter)

**Brain regions targeted:**
- Amygdala (bilateral in most patients)
- Hippocampus (bilateral in most patients)
- Anterior cingulate cortex (ACC)
- Pre-supplementary motor area (preSMA)
- Ventromedial prefrontal cortex (vmPFC)

**Task structure:**
1. **Movie watching phase:** 8-minute excerpt from "Bang! You're Dead" (Alfred Hitchcock, 1961) — a suspense film where a child unknowingly carries a loaded gun
2. **Recognition memory test:** Individual frames from the movie presented along with novel frames; patients rate old/new confidence

**Additional (not in NWB):**
- 3T fMRI in 11 of 20 patients (BIDS format on OpenNeuro ds004798)
- Movie annotation files (scene cuts, face annotations) in GitHub repository

### The Transformation

```
┌─────────────────────────────────────────────────────────────────┐
│                    LIMBIC → PREFRONTAL                          │
│                                                                 │
│  INPUT REGIONS                    OUTPUT REGIONS                │
│  ─────────────                    ──────────────                │
│  Amygdala (bilateral)             ACC                           │
│    • Threat detection             • Conflict monitoring         │
│    • Emotional valence            • Error prediction            │
│    • Arousal signaling            • Autonomic regulation        │
│                                                                 │
│  Hippocampus (bilateral)          preSMA                        │
│    • Contextual encoding          • Action preparation          │
│    • Sequence memory              • Response selection          │
│    • Spatial/temporal binding     • Motor planning              │
│                                                                 │
│                                   vmPFC                         │
│                                   • Value computation           │
│                                   • Emotion regulation          │
│                                   • Self-referential processing │
│                                   • Social cognition            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 0: Environment Setup and Data Access

### 0.1 Install Dependencies

```bash
# Create project directory
mkdir -p ~/descartes_movie/{data,exploration,preprocessing,models,probing,ablation,analysis,figures}
cd ~/descartes_movie

# Python environment
python -m venv venv
source venv/bin/activate

# Core dependencies
pip install pynwb dandi numpy scipy pandas matplotlib seaborn h5py

# ML dependencies
pip install torch scikit-learn

# LFP analysis
pip install mne

# NWB streaming (for initial exploration without full download)
pip install fsspec requests aiohttp remfile

# Clone the release code for annotations and reference scripts
git clone https://github.com/rutishauserlab/bmovie-release-NWB-BIDS.git
```

### 0.2 Download Data from DANDI

```bash
# Install DANDI CLI
pip install dandi

# Download all NWB files (~large, plan storage accordingly)
cd ~/descartes_movie/data
dandi download https://dandiarchive.org/dandiset/000623

# Alternative: download specific subjects for initial exploration
# Use streaming for exploration, full download for training
```

### 0.3 Initial NWB Exploration Script

**CRITICAL:** NWB column names vary across datasets and even across subjects within the same dataset. This lesson from Circuit 4 is non-negotiable: ALWAYS run the exploration script FIRST before writing any analysis code. Do not assume column names match the paper or previous Rutishauser datasets.

```python
"""
exploration/explore_nwb_structure.py

TASK 0: Run this FIRST on every subject file.
Discover the actual NWB structure before writing any analysis code.
"""
import pynwb
import numpy as np
from pathlib import Path

def explore_nwb(filepath):
    """
    Comprehensive exploration of a single NWB file.
    Prints ALL available data containers, column names, and shapes.
    """
    print(f"\n{'='*80}")
    print(f"EXPLORING: {filepath}")
    print(f"{'='*80}")
    
    with pynwb.NWBHDF5IO(str(filepath), 'r') as io:
        nwb = io.read()
        
        # ── Basic metadata ──
        print(f"\nSession: {nwb.session_description}")
        print(f"Subject: {nwb.subject}")
        print(f"Session start: {nwb.session_start_time}")
        print(f"Identifier: {nwb.identifier}")
        
        # ── Units (single neurons) ──
        print(f"\n--- UNITS (Single Neurons) ---")
        if nwb.units is not None:
            print(f"  Number of units: {len(nwb.units)}")
            print(f"  Column names: {nwb.units.colnames}")
            # Print first few entries of each column to understand format
            for col in nwb.units.colnames:
                try:
                    data = nwb.units[col][0]
                    print(f"  {col}: type={type(data).__name__}, example={str(data)[:100]}")
                except Exception as e:
                    print(f"  {col}: ERROR reading - {e}")
            
            # ── Brain region distribution ──
            # The column name for brain region might be 'electrodes', 
            # 'electrode_group', 'brain_area', 'location', etc.
            # DISCOVER IT HERE, do not assume
            region_candidates = ['electrodes', 'electrode_group', 
                               'brain_area', 'location', 'origChannel']
            for rc in region_candidates:
                if rc in nwb.units.colnames:
                    print(f"\n  Region column found: '{rc}'")
                    try:
                        regions = [nwb.units[rc][i] for i in range(len(nwb.units))]
                        # Handle cases where region is stored differently
                        if hasattr(regions[0], 'location'):
                            regions = [r.location for r in regions]
                        unique_regions = set(str(r) for r in regions)
                        print(f"  Unique regions: {unique_regions}")
                        for region in sorted(unique_regions):
                            count = sum(1 for r in regions if str(r) == region)
                            print(f"    {region}: {count} neurons")
                    except Exception as e:
                        print(f"  Error reading regions: {e}")
        
        # ── Electrodes ──
        print(f"\n--- ELECTRODES ---")
        if nwb.electrodes is not None:
            print(f"  Number of electrodes: {len(nwb.electrodes)}")
            print(f"  Column names: {nwb.electrodes.colnames}")
            for col in nwb.electrodes.colnames:
                try:
                    data = nwb.electrodes[col][0]
                    print(f"  {col}: type={type(data).__name__}, example={str(data)[:100]}")
                except Exception as e:
                    print(f"  {col}: ERROR - {e}")
        
        # ── Electrode Groups ──
        print(f"\n--- ELECTRODE GROUPS ---")
        for name, group in nwb.electrode_groups.items():
            print(f"  {name}: location={group.location}, "
                  f"description={group.description}")
        
        # ── Processing modules ──
        print(f"\n--- PROCESSING MODULES ---")
        for mod_name, module in nwb.processing.items():
            print(f"  Module: {mod_name}")
            for container_name, container in module.data_interfaces.items():
                print(f"    {container_name}: {type(container).__name__}")
                if hasattr(container, 'colnames'):
                    print(f"      Columns: {container.colnames}")
                if hasattr(container, 'data'):
                    try:
                        shape = container.data.shape if hasattr(container.data, 'shape') else 'N/A'
                        print(f"      Data shape: {shape}")
                    except:
                        pass
        
        # ── Acquisition (raw data) ──
        print(f"\n--- ACQUISITION ---")
        for acq_name, acq in nwb.acquisition.items():
            print(f"  {acq_name}: {type(acq).__name__}")
            if hasattr(acq, 'data'):
                try:
                    shape = acq.data.shape if hasattr(acq.data, 'shape') else 'N/A'
                    rate = acq.rate if hasattr(acq, 'rate') else 'N/A'
                    print(f"    Shape: {shape}, Rate: {rate} Hz")
                except:
                    pass
        
        # ── Trials / Epochs ──
        print(f"\n--- TRIALS ---")
        if nwb.trials is not None:
            print(f"  Number of trials: {len(nwb.trials)}")
            print(f"  Column names: {nwb.trials.colnames}")
            for col in nwb.trials.colnames:
                try:
                    data = nwb.trials[col][0]
                    print(f"  {col}: type={type(data).__name__}, example={str(data)[:100]}")
                except Exception as e:
                    print(f"  {col}: ERROR - {e}")
        
        # ── Intervals ──
        print(f"\n--- INTERVALS ---")
        if hasattr(nwb, 'intervals') and nwb.intervals is not None:
            for name, interval in nwb.intervals.items():
                print(f"  {name}: {len(interval)} intervals")
                if hasattr(interval, 'colnames'):
                    print(f"    Columns: {interval.colnames}")
        
        # ── Stimulus ──
        print(f"\n--- STIMULUS ---")
        for stim_name, stim in nwb.stimulus.items():
            print(f"  {stim_name}: {type(stim).__name__}")
            if hasattr(stim, 'data'):
                try:
                    shape = stim.data.shape if hasattr(stim.data, 'shape') else 'N/A'
                    print(f"    Shape: {shape}")
                except:
                    pass
    
    print(f"\n{'='*80}")
    print("EXPLORATION COMPLETE")
    print(f"{'='*80}")


def explore_all_subjects(data_dir):
    """
    Run exploration on ALL subject NWB files.
    Build a summary table of neurons per region per subject.
    """
    data_path = Path(data_dir)
    nwb_files = sorted(data_path.glob("**/*.nwb"))
    
    print(f"Found {len(nwb_files)} NWB files")
    
    # Explore the first file in detail
    if nwb_files:
        explore_nwb(nwb_files[0])
    
    # Build summary across all subjects
    print(f"\n\n{'='*80}")
    print("CROSS-SUBJECT SUMMARY")
    print(f"{'='*80}")
    
    summary = {}
    for f in nwb_files:
        try:
            with pynwb.NWBHDF5IO(str(f), 'r') as io:
                nwb = io.read()
                subject_id = f.stem  # or extract from nwb.subject
                n_units = len(nwb.units) if nwb.units else 0
                n_trials = len(nwb.trials) if nwb.trials else 0
                
                # Attempt to get region distribution
                # (adapt column name based on exploration results)
                region_counts = {}
                if nwb.units and n_units > 0:
                    # TRY MULTIPLE POSSIBLE COLUMN NAMES
                    for col_candidate in ['electrodes', 'electrode_group', 
                                          'brain_area', 'location']:
                        if col_candidate in nwb.units.colnames:
                            for i in range(n_units):
                                region = str(nwb.units[col_candidate][i])
                                if hasattr(nwb.units[col_candidate][i], 'location'):
                                    region = nwb.units[col_candidate][i].location
                                region_counts[region] = region_counts.get(region, 0) + 1
                            break
                
                summary[subject_id] = {
                    'n_units': n_units,
                    'n_trials': n_trials,
                    'regions': region_counts
                }
                print(f"\n{subject_id}: {n_units} units, {n_trials} trials")
                for region, count in sorted(region_counts.items()):
                    print(f"  {region}: {count}")
        except Exception as e:
            print(f"\nERROR reading {f.name}: {e}")
    
    return summary


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/000623"
    summary = explore_all_subjects(data_dir)
```

**After running this script, record the following before proceeding:**
1. The exact column name used for brain region assignment (e.g., `'electrode_group'`, `'brain_area'`, `'location'`)
2. The exact string labels used for each region (e.g., `'amygdala'`, `'Amygdala'`, `'Amy'`, `'LA'`, `'BLA'`)
3. Whether amygdala subnuclei are distinguished (lateral, basal, central)
4. The structure of the trials table (column names for the recognition memory test)
5. The structure of LFP data (which acquisition key, which electrodes map to which regions)
6. The presence and structure of any interval tables (movie events, scene boundaries)
7. Whether eye tracking data is included and its column structure

---

## Phase 1: Patient Inventory and Inclusion Criteria

### 1.1 The Input-Output Requirement

For the DESCARTES transformation model, each patient must have neurons in **both** input regions (amygdala OR hippocampus) and **at least one** output region (ACC, preSMA, or vmPFC). The minimum neuron count per region must be sufficient for reliable population-level modeling.

```python
"""
exploration/patient_inventory.py

Build the inclusion/exclusion table.
Determine which patients have sufficient neurons for the 
limbic→prefrontal transformation model.
"""
import numpy as np
from collections import defaultdict

# These region groupings should be adjusted based on exploration results.
# The exact string labels MUST come from Phase 0 exploration.
# DO NOT hardcode these without running explore_nwb first.

INPUT_REGIONS = {
    'amygdala': ['amygdala', 'Amygdala', 'Amy', 'LA', 'BLA', 'CeA',
                 'Left Amygdala', 'Right Amygdala'],  # ADJUST AFTER EXPLORATION
    'hippocampus': ['hippocampus', 'Hippocampus', 'Hip', 'HC', 'CA1', 'CA3', 'DG',
                    'Left Hippocampus', 'Right Hippocampus']  # ADJUST AFTER EXPLORATION
}

OUTPUT_REGIONS = {
    'ACC': ['ACC', 'acc', 'anterior cingulate', 'dACC', 'sgACC',
            'Anterior Cingulate'],  # ADJUST AFTER EXPLORATION
    'preSMA': ['preSMA', 'pre-SMA', 'presma', 'SMA',
               'pre-supplementary motor area'],  # ADJUST AFTER EXPLORATION
    'vmPFC': ['vmPFC', 'vmpfc', 'ventromedial prefrontal',
              'mOFC', 'medial orbitofrontal']  # ADJUST AFTER EXPLORATION
}

# Minimum neuron counts for reliable modeling
MIN_INPUT_NEURONS = 5    # At least 5 neurons across input regions
MIN_OUTPUT_NEURONS = 5   # At least 5 neurons across output regions
MIN_TOTAL_NEURONS = 15   # At least 15 neurons total for meaningful probing


def classify_neuron_region(region_label, region_map):
    """
    Map a neuron's region label to our canonical groupings.
    Returns the canonical group name or 'other'.
    """
    region_label_lower = str(region_label).lower().strip()
    for group_name, aliases in region_map.items():
        for alias in aliases:
            if alias.lower() in region_label_lower:
                return group_name
    return 'other'


def build_patient_inventory(summary_dict):
    """
    From the exploration summary, build inclusion/exclusion table.
    
    Args:
        summary_dict: output of explore_all_subjects()
            {subject_id: {'n_units': int, 'regions': {label: count}}}
    
    Returns:
        inventory: list of dicts with per-patient neuron counts and inclusion status
    """
    inventory = []
    
    for subject_id, info in summary_dict.items():
        patient = {
            'subject_id': subject_id,
            'total_units': info['n_units'],
            'n_trials': info.get('n_trials', 0)
        }
        
        # Count neurons per canonical region
        input_count = 0
        output_count = 0
        region_detail = {}
        
        for raw_label, count in info.get('regions', {}).items():
            # Check input regions
            canonical = classify_neuron_region(raw_label, INPUT_REGIONS)
            if canonical != 'other':
                input_count += count
                region_detail[f'input_{canonical}'] = region_detail.get(
                    f'input_{canonical}', 0) + count
            else:
                # Check output regions
                canonical = classify_neuron_region(raw_label, OUTPUT_REGIONS)
                if canonical != 'other':
                    output_count += count
                    region_detail[f'output_{canonical}'] = region_detail.get(
                        f'output_{canonical}', 0) + count
                else:
                    region_detail[f'other_{raw_label}'] = count
        
        patient['n_input'] = input_count
        patient['n_output'] = output_count
        patient['region_detail'] = region_detail
        
        # Inclusion criteria
        patient['included'] = (
            input_count >= MIN_INPUT_NEURONS and
            output_count >= MIN_OUTPUT_NEURONS and
            (input_count + output_count) >= MIN_TOTAL_NEURONS
        )
        
        # Exclusion reason
        reasons = []
        if input_count < MIN_INPUT_NEURONS:
            reasons.append(f"input neurons too few ({input_count} < {MIN_INPUT_NEURONS})")
        if output_count < MIN_OUTPUT_NEURONS:
            reasons.append(f"output neurons too few ({output_count} < {MIN_OUTPUT_NEURONS})")
        if (input_count + output_count) < MIN_TOTAL_NEURONS:
            reasons.append(f"total too few ({input_count + output_count} < {MIN_TOTAL_NEURONS})")
        patient['exclusion_reason'] = '; '.join(reasons) if reasons else 'INCLUDED'
        
        inventory.append(patient)
    
    return inventory


def print_inventory_table(inventory):
    """Print a formatted inclusion/exclusion table."""
    print(f"\n{'='*100}")
    print("PATIENT INVENTORY FOR DESCARTES CIRCUIT 5")
    print(f"{'='*100}")
    print(f"{'Subject':<20} {'Total':<8} {'Input':<8} {'Output':<8} "
          f"{'Included':<10} {'Reason'}")
    print(f"{'-'*100}")
    
    included_count = 0
    for p in sorted(inventory, key=lambda x: x['n_input'] + x['n_output'], reverse=True):
        status = "✓" if p['included'] else "✗"
        print(f"{p['subject_id']:<20} {p['total_units']:<8} {p['n_input']:<8} "
              f"{p['n_output']:<8} {status:<10} {p['exclusion_reason']}")
        if p['included']:
            included_count += 1
    
    print(f"\n{included_count} / {len(inventory)} patients included")
    
    # Region breakdown for included patients
    print(f"\nRegion breakdown (included patients only):")
    for p in inventory:
        if p['included']:
            detail = ', '.join(f"{k}: {v}" for k, v in 
                              sorted(p['region_detail'].items()))
            print(f"  {p['subject_id']}: {detail}")
```

### 1.2 Expected Yield

Based on the paper (20 patients, ~1,450 total neurons across 5 regions), expect roughly 10–15 patients with sufficient coverage in both input and output regions. Patients with only unilateral amygdala or only ACC (no preSMA/vmPFC) may need to be excluded or analyzed separately.

**Quality filters (apply after inclusion):**
- Spike sorting quality: use the quality metrics in the NWB file to exclude poorly isolated units
- Minimum firing rate during movie: exclude units with < 0.5 Hz mean rate (insufficient data)
- Cross-condition correlation sanity check: after training the surrogate, verify CC > 0.3 per model

---

## Phase 2: Movie-Specific Preprocessing

### 2.1 The Continuous Data Problem

Unlike Circuits 3 and 4 (Sternberg WM with discrete encode/delay/probe trials), movie watching produces **continuous** neural activity with no natural trial boundaries during the 8-minute film. The DESCARTES pipeline requires temporal windows for training and evaluation. Three segmentation strategies are available, and the guide recommends using **all three** for robustness:

**Strategy A — Sliding Window (Primary):**
Fixed-length windows sliding across the entire movie. This is the most unbiased approach and should be the primary segmentation method for surrogate training.

**Strategy B — Event-Based:**
Use annotated scene boundaries and emotional peaks as segment markers. This creates variable-length "pseudo-trials" aligned to meaningful cognitive events.

**Strategy C — Emotion-Phase:**
Segment by emotional trajectory: rising tension, peak, resolution. This is the most theory-driven approach but requires the most manual annotation alignment.

### 2.2 Sliding Window Preprocessing

```python
"""
preprocessing/sliding_window.py

Primary segmentation strategy for continuous movie data.
Create overlapping windows of neural activity for surrogate training.
"""
import numpy as np
from typing import Tuple, Dict, List

# ── Window parameters ──
# These are tuned for the 8-minute movie at millisecond resolution.
# The window must be long enough to capture neural dynamics 
# (theta cycles ~125-250 ms, emotional state changes ~1-5 s)
# but short enough to provide many training examples.

WINDOW_MS = 2000       # 2-second windows (captures ~8-16 theta cycles)
STRIDE_MS = 500        # 500 ms stride → 75% overlap → ~940 windows from 8 min
BIN_MS = 20            # 20 ms bins → 50 Hz effective rate (sufficient for spike rates)
                       # This gives 100 time bins per 2-second window


def bin_spike_trains(spike_times_list: List[np.ndarray],
                     bin_ms: float,
                     duration_ms: float) -> np.ndarray:
    """
    Convert spike times to binned spike counts.
    
    The binning resolution (bin_ms) determines the temporal granularity
    of the surrogate model. 20 ms is standard for population-level
    modeling: fine enough to resolve theta-frequency modulation (~125 ms
    period = ~6 bins) but coarse enough to smooth single-spike noise.
    
    Args:
        spike_times_list: list of arrays, one per neuron, times in SECONDS
        bin_ms: bin width in milliseconds
        duration_ms: total duration in milliseconds
    
    Returns:
        binned: (n_neurons, n_bins) array of spike counts per bin
    """
    n_neurons = len(spike_times_list)
    n_bins = int(np.ceil(duration_ms / bin_ms))
    binned = np.zeros((n_neurons, n_bins), dtype=np.float32)
    
    bin_edges = np.arange(0, duration_ms + bin_ms, bin_ms) / 1000.0  # convert to seconds
    
    for i, spike_times in enumerate(spike_times_list):
        if len(spike_times) > 0:
            counts, _ = np.histogram(spike_times, bins=bin_edges)
            binned[i, :len(counts)] = counts
    
    return binned


def smooth_spike_trains(binned: np.ndarray, 
                        sigma_bins: float = 2.0) -> np.ndarray:
    """
    Gaussian smoothing of binned spike trains.
    
    sigma_bins=2 at 20ms bins → 40ms Gaussian kernel.
    This smoothing is important for two reasons:
    1. Converts discrete spike counts to continuous firing rate estimates
    2. Provides temporal context that helps the LSTM learn dynamics
    
    Args:
        binned: (n_neurons, n_bins) spike count matrix
        sigma_bins: Gaussian kernel width in bins
    
    Returns:
        smoothed: (n_neurons, n_bins) smoothed firing rates
    """
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(binned.astype(np.float32), sigma=sigma_bins, axis=1)


def create_sliding_windows(input_rates: np.ndarray,
                           output_rates: np.ndarray,
                           window_bins: int,
                           stride_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract overlapping windows from continuous firing rate traces.
    
    This is the core data preparation function for continuous movie data.
    Each window becomes one "trial" for the LSTM surrogate.
    
    Args:
        input_rates: (n_input_neurons, T) smoothed firing rates for input regions
        output_rates: (n_output_neurons, T) smoothed firing rates for output regions
        window_bins: number of time bins per window
        stride_bins: number of bins between window starts
    
    Returns:
        X: (n_windows, window_bins, n_input_neurons) input tensor
        Y: (n_windows, window_bins, n_output_neurons) output tensor
    """
    T = input_rates.shape[1]
    assert output_rates.shape[1] == T, "Input and output must have same duration"
    
    n_windows = (T - window_bins) // stride_bins + 1
    n_input = input_rates.shape[0]
    n_output = output_rates.shape[0]
    
    X = np.zeros((n_windows, window_bins, n_input), dtype=np.float32)
    Y = np.zeros((n_windows, window_bins, n_output), dtype=np.float32)
    
    for i in range(n_windows):
        start = i * stride_bins
        end = start + window_bins
        X[i] = input_rates[:, start:end].T  # (window_bins, n_input)
        Y[i] = output_rates[:, start:end].T  # (window_bins, n_output)
    
    return X, Y


def prepare_movie_data(nwb_filepath: str,
                       input_region_labels: List[str],
                       output_region_labels: List[str],
                       region_column: str,
                       movie_start_time: float,
                       movie_end_time: float,
                       bin_ms: float = BIN_MS,
                       window_ms: float = WINDOW_MS,
                       stride_ms: float = STRIDE_MS,
                       sigma_bins: float = 2.0) -> Dict:
    """
    Full preprocessing pipeline for one patient's movie watching data.
    
    CRITICAL: movie_start_time and movie_end_time must be determined
    from the NWB file's trial/interval structure or TTL markers.
    These define the epoch of the 8-minute movie within the recording.
    
    Args:
        nwb_filepath: path to the patient's NWB file
        input_region_labels: list of region labels for input neurons 
                            (from Phase 0 exploration)
        output_region_labels: list of region labels for output neurons
        region_column: NWB column name for region assignment 
                      (from Phase 0 exploration)
        movie_start_time: start of movie epoch in seconds (from NWB timestamps)
        movie_end_time: end of movie epoch in seconds
        bin_ms: temporal bin width
        window_ms: sliding window width
        stride_ms: stride between windows
        sigma_bins: Gaussian smoothing kernel width
    
    Returns:
        dict with keys:
            'X': (n_windows, window_bins, n_input) input tensor
            'Y': (n_windows, window_bins, n_output) output tensor
            'input_neuron_ids': list of unit indices used as input
            'output_neuron_ids': list of unit indices used as output
            'window_times': (n_windows,) start times of each window in seconds
            'bin_ms': bin width used
            'metadata': dict of preprocessing parameters
    """
    import pynwb
    
    with pynwb.NWBHDF5IO(str(nwb_filepath), 'r') as io:
        nwb = io.read()
        
        # ── Identify input and output neurons ──
        input_ids = []
        output_ids = []
        input_spikes = []
        output_spikes = []
        
        for i in range(len(nwb.units)):
            # Get this neuron's region label
            region = str(nwb.units[region_column][i])
            # Handle electrode group objects (may have .location attribute)
            if hasattr(nwb.units[region_column][i], 'location'):
                region = nwb.units[region_column][i].location
            
            # Get spike times within movie epoch
            all_spikes = nwb.units['spike_times'][i]
            movie_spikes = all_spikes[
                (all_spikes >= movie_start_time) & 
                (all_spikes <= movie_end_time)
            ] - movie_start_time  # Zero-reference to movie start
            
            # Classify into input or output
            region_lower = region.lower()
            is_input = any(lbl.lower() in region_lower 
                          for lbl in input_region_labels)
            is_output = any(lbl.lower() in region_lower 
                           for lbl in output_region_labels)
            
            if is_input:
                input_ids.append(i)
                input_spikes.append(movie_spikes)
            elif is_output:
                output_ids.append(i)
                output_spikes.append(movie_spikes)
    
    # ── Bin and smooth ──
    duration_ms = (movie_end_time - movie_start_time) * 1000.0
    
    input_binned = bin_spike_trains(input_spikes, bin_ms, duration_ms)
    output_binned = bin_spike_trains(output_spikes, bin_ms, duration_ms)
    
    input_rates = smooth_spike_trains(input_binned, sigma_bins)
    output_rates = smooth_spike_trains(output_binned, sigma_bins)
    
    # ── Create sliding windows ──
    window_bins = int(window_ms / bin_ms)
    stride_bins = int(stride_ms / bin_ms)
    
    X, Y = create_sliding_windows(input_rates, output_rates, 
                                   window_bins, stride_bins)
    
    # ── Window timestamps (for annotation alignment) ──
    n_windows = X.shape[0]
    window_starts = np.array([i * stride_bins * bin_ms / 1000.0 
                              for i in range(n_windows)])
    
    return {
        'X': X,
        'Y': Y,
        'input_neuron_ids': input_ids,
        'output_neuron_ids': output_ids,
        'window_times': window_starts,
        'input_rates_continuous': input_rates,
        'output_rates_continuous': output_rates,
        'bin_ms': bin_ms,
        'metadata': {
            'window_ms': window_ms,
            'stride_ms': stride_ms,
            'sigma_bins': sigma_bins,
            'n_input_neurons': len(input_ids),
            'n_output_neurons': len(output_ids),
            'n_windows': n_windows,
            'movie_duration_s': movie_end_time - movie_start_time,
        }
    }
```

### 2.3 Event-Based Segmentation

```python
"""
preprocessing/event_segmentation.py

Strategy B: Segment movie by annotated events.
The bmovie-release-NWB-BIDS repository provides scene cut annotations
and face annotations in the assets/annotations/ folder.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

def load_movie_annotations(annotations_dir: str) -> Dict:
    """
    Load movie annotation files from the bmovie-release-NWB-BIDS repository.
    
    CRITICAL: The exact file names and formats must be discovered by inspection.
    The repository provides annotations in assets/annotations/.
    Expected annotation types (verify by inspection):
      - Scene cuts / shot boundaries
      - Face presence annotations
      - Possibly arousal/valence ratings (check if available)
    
    Returns dict with annotation arrays keyed by type.
    """
    from pathlib import Path
    ann_path = Path(annotations_dir)
    
    annotations = {}
    
    # List all annotation files to discover what's available
    print("Available annotation files:")
    for f in sorted(ann_path.glob("*")):
        print(f"  {f.name} ({f.stat().st_size / 1024:.1f} KB)")
    
    # Load each file (format must be determined by inspection)
    # Common formats: CSV, TSV, JSON, MAT
    for f in ann_path.glob("*.csv"):
        try:
            df = pd.read_csv(f)
            annotations[f.stem] = df
            print(f"\n{f.stem}: {len(df)} rows, columns={list(df.columns)}")
        except Exception as e:
            print(f"Error loading {f.name}: {e}")
    
    for f in ann_path.glob("*.tsv"):
        try:
            df = pd.read_csv(f, sep='\t')
            annotations[f.stem] = df
            print(f"\n{f.stem}: {len(df)} rows, columns={list(df.columns)}")
        except Exception as e:
            print(f"Error loading {f.name}: {e}")
    
    return annotations


def create_event_segments(scene_boundaries: np.ndarray,
                          input_rates: np.ndarray,
                          output_rates: np.ndarray,
                          bin_ms: float,
                          min_segment_bins: int = 50,
                          max_segment_bins: int = 200) -> List[Dict]:
    """
    Create variable-length segments aligned to scene boundaries.
    
    Each segment spans from one scene boundary to the next.
    Segments that are too short or too long are handled:
    - Too short (< min_segment_bins): merged with adjacent segment
    - Too long (> max_segment_bins): split into sub-segments
    
    This is important because scene boundaries mark cognitive event
    boundaries — the brain reorganizes its processing at these points.
    Segments aligned to events capture more coherent neural dynamics
    than arbitrary sliding windows.
    
    Args:
        scene_boundaries: array of scene boundary times in seconds
        input_rates: (n_input, T_bins) continuous input rates
        output_rates: (n_output, T_bins) continuous output rates
        bin_ms: bin width in ms
        min_segment_bins: minimum segment length (default 50 = 1s at 20ms bins)
        max_segment_bins: maximum segment length (default 200 = 4s at 20ms bins)
    
    Returns:
        segments: list of dicts, each with 'input', 'output', 
                  'start_time', 'end_time', 'event_type'
    """
    T = input_rates.shape[1]
    
    # Convert boundary times to bin indices
    boundary_bins = (scene_boundaries * 1000.0 / bin_ms).astype(int)
    boundary_bins = boundary_bins[(boundary_bins >= 0) & (boundary_bins < T)]
    
    # Add start and end as implicit boundaries
    boundaries = np.concatenate([[0], boundary_bins, [T]])
    boundaries = np.unique(boundaries)
    
    segments = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        length = end - start
        
        if length < min_segment_bins:
            continue  # Skip very short segments (will be captured in sliding windows)
        
        if length > max_segment_bins:
            # Split into sub-segments
            for sub_start in range(start, end, max_segment_bins):
                sub_end = min(sub_start + max_segment_bins, end)
                if sub_end - sub_start >= min_segment_bins:
                    segments.append({
                        'input': input_rates[:, sub_start:sub_end].T,
                        'output': output_rates[:, sub_start:sub_end].T,
                        'start_time': sub_start * bin_ms / 1000.0,
                        'end_time': sub_end * bin_ms / 1000.0,
                        'start_bin': sub_start,
                        'end_bin': sub_end,
                    })
        else:
            segments.append({
                'input': input_rates[:, start:end].T,
                'output': output_rates[:, start:end].T,
                'start_time': start * bin_ms / 1000.0,
                'end_time': end * bin_ms / 1000.0,
                'start_bin': start,
                'end_bin': end,
            })
    
    return segments
```

### 2.4 Recognition Memory Test Preprocessing

The recognition memory test after the movie provides discrete trials (movie frames shown as old/new judgments). This data is preprocessed separately and serves two purposes: (a) validation bridge to Circuit 4 working memory results, and (b) computation of encoding success labels for movie-phase neural activity.

```python
"""
preprocessing/recognition_memory.py

Preprocess the recognition memory test for:
1. Computing encoding success labels (which movie moments were later remembered?)
2. Validation bridge to Circuit 4 (same Rutishauser lab, similar task structure)
"""
import numpy as np
from typing import Dict, List

def extract_recognition_trials(nwb_filepath: str,
                                trial_column_names: Dict = None) -> Dict:
    """
    Extract recognition memory test data from NWB.
    
    CRITICAL: Column names for the trials table must come from Phase 0 exploration.
    Do NOT assume they match DANDI 000576 (human WM dataset).
    
    Expected columns (verify with exploration):
        - start_time, stop_time: trial timing
        - stimulus type: old (from movie) vs new
        - response: old/new judgment
        - confidence: confidence rating
        - stimulus_frame or image_id: which movie frame was shown
        - possibly: movie_timestamp (when the frame appeared in the movie)
    
    Args:
        nwb_filepath: path to NWB file
        trial_column_names: dict mapping semantic role to actual column name
            e.g., {'stimulus_type': 'stim_type', 'response': 'response_value'}
            If None, attempts auto-detection from common patterns.
    
    Returns:
        trials: dict with arrays for each trial variable
    """
    import pynwb
    
    with pynwb.NWBHDF5IO(str(nwb_filepath), 'r') as io:
        nwb = io.read()
        
        if nwb.trials is None:
            print("WARNING: No trials table found. Recognition test may be "
                  "stored in a different location (intervals, processing module).")
            return {}
        
        trials = {}
        for col in nwb.trials.colnames:
            try:
                trials[col] = np.array([nwb.trials[col][i] 
                                       for i in range(len(nwb.trials))])
            except:
                trials[col] = [nwb.trials[col][i] 
                              for i in range(len(nwb.trials))]
        
        print(f"Recognition test: {len(nwb.trials)} trials")
        print(f"Columns: {list(trials.keys())}")
        
        return trials


def compute_encoding_success(recognition_trials: Dict,
                             movie_frame_times: np.ndarray = None) -> np.ndarray:
    """
    For each moment in the movie, compute whether that content was 
    subsequently remembered in the recognition test.
    
    This creates a binary (or continuous) label that can be aligned
    to movie-phase neural activity, enabling encoding success probing:
    "Do neurons during the movie predict later recognition?"
    
    The specific implementation depends on whether the NWB file 
    contains movie timestamps for each test frame. If not, this 
    must be computed from the stimulus frame identity.
    
    Returns:
        encoding_labels: array mapping movie timepoints to subsequent 
                        memory performance (hit/miss/not_tested)
    """
    # Implementation depends on trial table structure discovered in Phase 0
    # This is a skeleton — fill in after exploration
    
    # Identify hits (old items correctly identified as old)
    # and misses (old items incorrectly called new)
    # New items are not relevant for encoding success
    
    # Map each old item back to its movie timestamp
    # Label that timestamp window as "remembered" (hit) or "forgotten" (miss)
    
    raise NotImplementedError(
        "Implement after Phase 0 exploration reveals trial table structure. "
        "Key columns needed: stimulus type (old/new), response, and "
        "movie timestamp for each old stimulus."
    )
```

---

## Phase 3: Annotation Alignment

### 3.1 Available Annotations

The bmovie-release-NWB-BIDS GitHub repository provides movie annotations in the `assets/annotations/` folder. These annotations are critical for computing the probe targets — without them, we can only probe for emergent dynamics (theta, gamma, synchrony) but not for cognitive variables (arousal, valence, event boundaries).

**Annotations provided (verify by inspection):**
- **Scene cuts / shot boundaries:** timestamps where camera shots change
- **Face annotations:** when faces appear on screen and their attributes
- **Potentially:** arousal/valence ratings, character tracking, emotional categories

**Annotations that may need to be computed or sourced externally:**
- Continuous arousal and valence ratings (may need to derive from literature or conduct ratings)
- Social prediction errors (need manual annotation or computational modeling)
- Threat prediction errors (can be estimated from suspense structure)

### 3.2 Annotation Alignment Pipeline

```python
"""
preprocessing/annotation_alignment.py

Align movie annotations to neural timestamps.
All annotations must be expressed in the same time base as neural data.
"""
import numpy as np
from typing import Dict

def align_annotations_to_bins(annotations: Dict,
                               movie_start_time: float,
                               movie_duration_s: float,
                               bin_ms: float) -> Dict:
    """
    Convert annotation timestamps to bin indices aligned with neural data.
    
    The key challenge: annotations are typically in movie-relative time
    (0 = movie start), but neural data timestamps are in recording-
    relative time. The TTL synchronization markers in the NWB file 
    define the mapping.
    
    Args:
        annotations: dict of annotation arrays (from load_movie_annotations)
        movie_start_time: start of movie in neural recording time (seconds)
        movie_duration_s: duration of movie in seconds
        bin_ms: bin width used in neural preprocessing
    
    Returns:
        aligned: dict with binned annotation time series
    """
    n_bins = int(np.ceil(movie_duration_s * 1000.0 / bin_ms))
    aligned = {}
    
    # The specific alignment depends on the annotation format.
    # Common patterns:
    
    # Pattern 1: Event timestamps (e.g., scene boundaries)
    # → Convert to binary time series (1 at boundary, 0 elsewhere)
    
    # Pattern 2: Interval annotations (e.g., face present from t1 to t2)
    # → Convert to binary time series (1 during interval, 0 outside)
    
    # Pattern 3: Continuous ratings (e.g., arousal at each second)
    # → Interpolate to match bin resolution
    
    return aligned


def compute_arousal_from_annotations(scene_boundaries: np.ndarray,
                                      face_annotations: Dict,
                                      movie_duration_s: float,
                                      bin_ms: float) -> np.ndarray:
    """
    Estimate continuous arousal signal from available annotations.
    
    If explicit arousal ratings are not provided, arousal can be
    estimated from proxy measures:
    
    1. Scene cut rate: More frequent cuts → higher arousal (Hitchcock
       deliberately increases cutting rate during suspense sequences)
    2. Face density: More faces on screen → higher social engagement
    3. Music/sound annotations (if available): Crescendos → rising arousal
    
    The Hitchcock film "Bang! You're Dead" has a well-known suspense
    structure: arousal rises as the child carries the gun through
    increasingly dangerous situations, with peaks at moments where 
    adults nearly discover the danger.
    
    For the DESCARTES probing framework, even an imperfect arousal
    estimate is useful — the question is whether the LSTM discovers
    arousal-correlated representations, not whether our label is perfect.
    
    Returns:
        arousal_estimate: (n_bins,) estimated arousal time series
    """
    n_bins = int(np.ceil(movie_duration_s * 1000.0 / bin_ms))
    
    # Method: Kernel density of scene cuts as arousal proxy
    # More cuts per unit time = more editing = higher tension
    boundary_bins = (scene_boundaries * 1000.0 / bin_ms).astype(int)
    boundary_bins = boundary_bins[(boundary_bins >= 0) & (boundary_bins < n_bins)]
    
    # Create impulse train at scene boundaries
    cut_signal = np.zeros(n_bins)
    cut_signal[boundary_bins] = 1.0
    
    # Smooth with wide kernel to get local cut density
    from scipy.ndimage import gaussian_filter1d
    # 10-second kernel (captures arousal dynamics timescale)
    sigma = 10.0 * 1000.0 / bin_ms  # 10 seconds in bins
    arousal_estimate = gaussian_filter1d(cut_signal, sigma=sigma)
    
    # Normalize to [0, 1]
    if arousal_estimate.max() > 0:
        arousal_estimate = arousal_estimate / arousal_estimate.max()
    
    return arousal_estimate
```

### 3.3 External Arousal/Valence Ratings

If the dataset or repository does not include continuous arousal/valence ratings, there are two options:

**Option A — Literature-derived annotations:** The film "Bang! You're Dead" has been used in multiple neuroimaging studies. Search for published arousal/valence time courses from other studies using this same film clip. Several fMRI studies have used continuous behavioral ratings during Hitchcock films.

**Option B — Self-annotation:** Watch the 8-minute film and create manual arousal/valence annotations at 1-second resolution. While subjective, a single rater's annotations are sufficient for the DESCARTES framework because the question is not "does the LSTM perfectly predict arousal" but rather "does the LSTM develop internal representations that correlate with arousal, and are those representations mandatory?"

**Option C — Pupillometry proxy:** The NWB files include eye tracking data with pupil diameter. Pupil diameter is a well-established physiological correlate of arousal. Use pupil diameter as a continuous arousal proxy, after detrending for luminance changes (which can be estimated from the video).

```python
def extract_pupil_arousal_proxy(nwb_filepath: str,
                                 movie_start_time: float,
                                 movie_end_time: float,
                                 bin_ms: float) -> np.ndarray:
    """
    Extract pupil diameter as arousal proxy from eye tracking data.
    
    Pupil dilation is controlled by the locus coeruleus-norepinephrine
    system and correlates with autonomic arousal. Larger pupil → higher arousal.
    
    Caveats:
    - Must detrend for luminance (bright scenes constrict pupil regardless of arousal)
    - Missing data from blinks must be interpolated
    - Individual differences in baseline pupil size require z-scoring
    
    Returns:
        pupil_arousal: (n_bins,) z-scored, detrended pupil diameter
    """
    import pynwb
    
    with pynwb.NWBHDF5IO(str(nwb_filepath), 'r') as io:
        nwb = io.read()
        
        # Eye tracking data location varies — discover in Phase 0
        # Common locations: nwb.acquisition['EyeTracking'], 
        #                   nwb.processing['behavior']['PupilTracking'],
        #                   nwb.processing['behavior']['EyeTracking']
        
        # Skeleton — fill in after Phase 0 exploration
        raise NotImplementedError(
            "Implement after Phase 0 reveals eye tracking data structure. "
            "Need: pupil diameter time series, sampling rate, and timestamps."
        )
```

---

## Phase 4: Probe Target Computation

### 4.1 Probe Target Taxonomy

The probe targets for Circuit 5 span four domains: emotion, social/narrative, emergent dynamics, and memory. Each probe target is a time-varying signal that can be compared to the LSTM's hidden states using the standardized Ridge ΔR² methodology.

```python
"""
probing/probe_targets.py

Compute all probe targets for the movie emotion experiment.
Each function returns a (T_bins,) or (T_bins, n_dims) array
aligned to the binned neural data time axis.
"""
import numpy as np
from scipy.signal import hilbert, butter, filtfilt
from scipy.ndimage import gaussian_filter1d


# ═══════════════════════════════════════════════════════════════
# DOMAIN 1: EMOTION
# ═══════════════════════════════════════════════════════════════

def compute_valence_signal(arousal_ts: np.ndarray,
                           scene_valence_labels: np.ndarray = None) -> np.ndarray:
    """
    Valence signal: positive/negative emotional dimension.
    
    If scene-level valence annotations are available, use them directly.
    Otherwise, valence can be partially dissociated from arousal using
    the circumplex model: high arousal can be either positive (excitement)
    or negative (fear), so valence and arousal are orthogonal dimensions.
    
    For "Bang! You're Dead", the dominant valence is NEGATIVE (fear,
    anxiety) with brief POSITIVE moments (relief when danger passes).
    """
    if scene_valence_labels is not None:
        return scene_valence_labels
    
    # Placeholder: without explicit labels, return arousal negated 
    # (higher arousal in this suspense film = more negative valence)
    # This is a coarse approximation — the actual valence signal
    # should come from behavioral annotations when available
    return -arousal_ts


def compute_arousal_signal(pupil_diameter: np.ndarray = None,
                            scene_cut_density: np.ndarray = None,
                            arousal_ratings: np.ndarray = None) -> np.ndarray:
    """
    Arousal signal: intensity of emotional engagement.
    
    Hierarchy of sources (use best available):
    1. Behavioral arousal ratings (if collected)
    2. Pupil diameter (physiological proxy)
    3. Scene cut density (structural proxy)
    """
    if arousal_ratings is not None:
        return arousal_ratings
    if pupil_diameter is not None:
        # Z-score and smooth
        from scipy.stats import zscore
        return gaussian_filter1d(zscore(pupil_diameter), sigma=5)
    if scene_cut_density is not None:
        return scene_cut_density
    
    raise ValueError("At least one arousal source required")


def compute_threat_prediction_error(arousal_ts: np.ndarray,
                                      smooth_sigma: float = 25.0) -> np.ndarray:
    """
    Threat prediction error: unexpected scary moments.
    
    Defined as the derivative of arousal — moments where arousal
    increases faster than expected. Positive values = unexpected
    increase in threat; negative values = unexpected relief.
    
    In the predictive processing framework, prediction error is 
    computed as the difference between observed arousal and a
    slow-moving prediction (implemented as heavily smoothed arousal).
    
    This is neurally relevant because amygdala neurons respond 
    specifically to UNEXPECTED threats, not to sustained threat levels.
    """
    # Slow prediction: heavily smoothed arousal
    predicted = gaussian_filter1d(arousal_ts, sigma=smooth_sigma)
    
    # Prediction error: observed - predicted
    prediction_error = arousal_ts - predicted
    
    return prediction_error


def compute_emotional_category(arousal_ts: np.ndarray,
                                valence_ts: np.ndarray,
                                threshold: float = 0.5) -> np.ndarray:
    """
    Emotional category: fear vs surprise vs relief.
    
    Derived from circumplex coordinates:
    - Fear: high arousal, negative valence
    - Surprise: high arousal, valence transition
    - Relief: arousal decrease, valence shift toward positive
    
    Returns integer labels (0=neutral, 1=fear, 2=surprise, 3=relief)
    for use in classification probing.
    """
    n = len(arousal_ts)
    categories = np.zeros(n, dtype=np.int32)  # 0 = neutral
    
    # Fear: high arousal AND negative valence
    fear_mask = (arousal_ts > threshold) & (valence_ts < -threshold)
    categories[fear_mask] = 1
    
    # Surprise: rapid arousal increase (positive derivative)
    arousal_deriv = np.gradient(arousal_ts)
    surprise_mask = arousal_deriv > np.percentile(arousal_deriv, 90)
    categories[surprise_mask] = 2
    
    # Relief: rapid arousal decrease from high level
    relief_mask = (arousal_deriv < np.percentile(arousal_deriv, 10)) & \
                  (np.roll(arousal_ts, 5) > threshold)  # was recently high
    categories[relief_mask] = 3
    
    return categories


# ═══════════════════════════════════════════════════════════════
# DOMAIN 2: SOCIAL / NARRATIVE
# ═══════════════════════════════════════════════════════════════

def compute_event_boundary_signal(scene_boundaries: np.ndarray,
                                   n_bins: int,
                                   bin_ms: float,
                                   kernel_sigma_bins: float = 5.0) -> np.ndarray:
    """
    Event boundary signal: transitions between narrative scenes.
    
    Event boundaries are marked by scene cuts in the movie and are
    neurally significant because the hippocampus shows boundary-
    locked responses (event segmentation theory, Zacks & Swallow 2007).
    The Keles et al. paper validated that neurons in this dataset 
    respond to event boundaries.
    
    Returns a smooth signal peaking at each boundary.
    """
    signal = np.zeros(n_bins)
    boundary_bins = (scene_boundaries * 1000.0 / bin_ms).astype(int)
    boundary_bins = boundary_bins[(boundary_bins >= 0) & (boundary_bins < n_bins)]
    signal[boundary_bins] = 1.0
    
    # Smooth to capture peri-boundary neural dynamics
    return gaussian_filter1d(signal, sigma=kernel_sigma_bins)


def compute_face_presence(face_annotations: np.ndarray,
                           n_bins: int,
                           bin_ms: float) -> np.ndarray:
    """
    Character-directed attention: when faces are visible on screen.
    
    Face presence drives amygdala activation (the amygdala has 
    face-selective neurons, Rutishauser et al. 2011) and engages
    social cognition circuits.
    
    Returns binary or count signal of faces visible per time bin.
    """
    # Implementation depends on face annotation format from repository
    # Typical format: onset_time, offset_time, face_identity, face_emotion
    raise NotImplementedError(
        "Implement after loading face annotations from repository."
    )


# ═══════════════════════════════════════════════════════════════
# DOMAIN 3: EMERGENT DYNAMICS
# These are computed from the neural data itself, not from annotations.
# They are the primary candidates for cross-domain universality.
# ═══════════════════════════════════════════════════════════════

def compute_theta_modulation(lfp_data: np.ndarray,
                              lfp_rate: float,
                              bin_ms: float,
                              theta_band: tuple = (4, 8)) -> np.ndarray:
    """
    Theta modulation (4-8 Hz): CRITICAL cross-domain probe target.
    
    Theta was mandatory in Circuit 2 (hippocampal memory), Circuit 3
    (mouse thalamocortical WM), and some Circuit 4 patients.
    If theta is mandatory here in an emotion task, it constitutes
    evidence for a UNIVERSAL limbic-prefrontal computational role.
    
    Computed from LFP (not spike trains) because theta is a 
    field potential phenomenon reflecting population-level oscillatory
    coordination.
    
    Args:
        lfp_data: (n_channels, n_samples) LFP data from input regions
        lfp_rate: sampling rate of LFP in Hz
        bin_ms: target bin width for output
        theta_band: frequency range for theta
    
    Returns:
        theta_power: (n_bins,) theta band power time series
        theta_phase: (n_bins,) theta phase time series
    """
    # Bandpass filter in theta range
    nyq = lfp_rate / 2.0
    low = theta_band[0] / nyq
    high = theta_band[1] / nyq
    b, a = butter(4, [low, high], btype='band')
    
    # Average across channels (or use channel with strongest theta)
    if lfp_data.ndim == 2:
        lfp_mean = np.mean(lfp_data, axis=0)
    else:
        lfp_mean = lfp_data
    
    theta_filtered = filtfilt(b, a, lfp_mean)
    
    # Hilbert transform for instantaneous amplitude and phase
    analytic = hilbert(theta_filtered)
    theta_amplitude = np.abs(analytic)
    theta_phase = np.angle(analytic)
    
    # Downsample to match bin resolution
    samples_per_bin = int(bin_ms / 1000.0 * lfp_rate)
    n_bins = len(theta_amplitude) // samples_per_bin
    
    theta_power = np.array([
        np.mean(theta_amplitude[i*samples_per_bin:(i+1)*samples_per_bin])
        for i in range(n_bins)
    ])
    
    theta_phase_binned = np.array([
        np.angle(np.mean(np.exp(1j * theta_phase[i*samples_per_bin:(i+1)*samples_per_bin])))
        for i in range(n_bins)
    ])
    
    return theta_power, theta_phase_binned


def compute_gamma_modulation(lfp_data: np.ndarray,
                              lfp_rate: float,
                              bin_ms: float,
                              gamma_band: tuple = (30, 80)) -> np.ndarray:
    """
    Gamma modulation (30-80 Hz): mandatory in Circuit 2 (hippocampal memory).
    
    Gamma amplitude was the SOLE mandatory variable in the CA3→CA1
    transformation. Testing whether it persists as mandatory in a
    different cognitive domain is a key prediction of this experiment.
    
    High-frequency broadband (HFB) power, which overlaps with gamma,
    is also a well-established correlate of local neural population
    activity and is validated in the Keles et al. paper.
    """
    # Same structure as theta, different frequency band
    nyq = lfp_rate / 2.0
    low = gamma_band[0] / nyq
    high = gamma_band[1] / nyq
    b, a = butter(4, [low, high], btype='band')
    
    if lfp_data.ndim == 2:
        lfp_mean = np.mean(lfp_data, axis=0)
    else:
        lfp_mean = lfp_data
    
    gamma_filtered = filtfilt(b, a, lfp_mean)
    analytic = hilbert(gamma_filtered)
    gamma_amplitude = np.abs(analytic)
    
    samples_per_bin = int(bin_ms / 1000.0 * lfp_rate)
    n_bins = len(gamma_amplitude) // samples_per_bin
    
    gamma_power = np.array([
        np.mean(gamma_amplitude[i*samples_per_bin:(i+1)*samples_per_bin])
        for i in range(n_bins)
    ])
    
    return gamma_power


def compute_population_synchrony(spike_rates: np.ndarray,
                                  window_bins: int = 10) -> np.ndarray:
    """
    Population synchrony: pairwise correlations within a sliding window.
    
    Measures how coordinated neural firing is across the population.
    High synchrony may indicate coherent population states that the
    LSTM needs to track.
    
    This was probed in Circuit 4 (human WM) where it was mandatory
    in 30% of patients.
    """
    n_neurons, T = spike_rates.shape
    n_windows = T - window_bins + 1
    synchrony = np.zeros(n_windows)
    
    for t in range(n_windows):
        window = spike_rates[:, t:t+window_bins]
        # Pairwise correlations
        if n_neurons > 1:
            corr_matrix = np.corrcoef(window)
            # Extract upper triangle (excluding diagonal)
            upper_tri = corr_matrix[np.triu_indices(n_neurons, k=1)]
            # Mean absolute correlation as synchrony measure
            synchrony[t] = np.nanmean(np.abs(upper_tri))
    
    # Pad to original length
    pad = T - n_windows
    synchrony = np.concatenate([synchrony, np.full(pad, synchrony[-1])])
    
    return synchrony


def compute_temporal_stability(spike_rates: np.ndarray,
                                window_bins: int = 25) -> np.ndarray:
    """
    Temporal stability: how persistent is the population state?
    
    Measured as autocorrelation of the population vector at a short lag.
    High stability = persistent attractor state (e.g., sustained emotion).
    Low stability = rapid state transitions (e.g., surprise, boundary).
    
    This captures the dynamics of "attractor persistence" which is
    relevant to both working memory (delay period stability) and
    emotion (sustained emotional states vs transient reactions).
    """
    n_neurons, T = spike_rates.shape
    stability = np.zeros(T)
    
    lag = 5  # 5 bins = 100 ms at 20ms bins
    
    for t in range(lag, T):
        current = spike_rates[:, t]
        previous = spike_rates[:, t - lag]
        
        # Cosine similarity between population vectors
        norm_curr = np.linalg.norm(current)
        norm_prev = np.linalg.norm(previous)
        
        if norm_curr > 0 and norm_prev > 0:
            stability[t] = np.dot(current, previous) / (norm_curr * norm_prev)
    
    # Fill initial values
    stability[:lag] = stability[lag]
    
    return stability


# ═══════════════════════════════════════════════════════════════
# DOMAIN 4: MEMORY
# ═══════════════════════════════════════════════════════════════

def compute_encoding_success_signal(recognition_results: dict,
                                      movie_duration_s: float,
                                      bin_ms: float,
                                      kernel_sigma_s: float = 2.0) -> np.ndarray:
    """
    Encoding success: do neural patterns during the movie predict
    later recognition memory?
    
    This is a SUBSEQUENT MEMORY probe: for each movie time point,
    was the nearby content later remembered (hit) or forgotten (miss)?
    
    This creates a binary (or graded) label aligned to the movie timeline.
    If the LSTM develops mandatory representations that correlate with
    encoding success, it means the limbic→prefrontal transformation
    inherently carries memory-predictive information.
    
    This directly bridges to Circuit 4 results.
    """
    n_bins = int(np.ceil(movie_duration_s * 1000.0 / bin_ms))
    
    # Implementation depends on recognition trial structure
    # Key logic:
    # 1. For each "old" trial (movie frame), identify the movie timestamp
    # 2. Label that timestamp as "hit" (correct old) or "miss" (incorrect new)
    # 3. Smooth to create continuous encoding success signal
    
    raise NotImplementedError(
        "Implement after Phase 0 reveals recognition trial structure "
        "and movie frame timestamps."
    )
```

### 4.2 Probe Target Summary Table

```
┌─────────────────────────────┬───────────┬──────────────┬───────────────────┐
│ Probe Target                │ Domain    │ Source       │ Cross-Domain?     │
├─────────────────────────────┼───────────┼──────────────┼───────────────────┤
│ valence_signal              │ Emotion   │ Annotation   │ NEW — emotion     │
│ arousal_signal              │ Emotion   │ Pupil/Ann.   │ NEW — emotion     │
│ threat_prediction_error     │ Emotion   │ Computed     │ NEW — emotion     │
│ emotional_category          │ Emotion   │ Computed     │ NEW — emotion     │
├─────────────────────────────┼───────────┼──────────────┼───────────────────┤
│ event_boundary_signal       │ Narrative │ Annotation   │ NEW — narrative   │
│ face_presence               │ Social    │ Annotation   │ NEW — social      │
├─────────────────────────────┼───────────┼──────────────┼───────────────────┤
│ theta_modulation            │ Dynamics  │ LFP          │ YES — Circuits 2-4│
│ gamma_modulation            │ Dynamics  │ LFP          │ YES — Circuit 2   │
│ population_synchrony        │ Dynamics  │ Spikes       │ YES — Circuit 4   │
│ temporal_stability          │ Dynamics  │ Spikes       │ YES — Circuit 3-4 │
├─────────────────────────────┼───────────┼──────────────┼───────────────────┤
│ encoding_success            │ Memory    │ Behavior     │ YES — Circuit 4   │
└─────────────────────────────┴───────────┴──────────────┴───────────────────┘
```

---

## Phase 5: LSTM Surrogate Architecture

### 5.1 Model Architecture

The surrogate model is identical in structure to Circuits 3 and 4, adapted only in input/output dimensions. This architectural consistency is deliberate: the DESCARTES framework requires that the model architecture be held constant across circuits to isolate the effect of the transformation being learned.

```python
"""
models/surrogate_lstm.py

LSTM surrogate for the limbic→prefrontal transformation.
Architecture matches Circuits 3 and 4 for cross-circuit comparability.
"""
import torch
import torch.nn as nn
from typing import Dict, Tuple


class LimbicPrefrontalLSTM(nn.Module):
    """
    LSTM surrogate replacing the limbic→prefrontal transformation.
    
    Input:  (batch, time, n_input)  — amygdala + hippocampus firing rates
    Output: (batch, time, n_output) — ACC + preSMA + vmPFC firing rates
    Hidden: (batch, hidden_size)    — the candidate representations to probe
    
    The hidden state is the "black box" that DESCARTES probes to determine
    whether the LSTM discovers emotion-related intermediate variables.
    """
    
    def __init__(self, n_input: int, n_output: int, 
                 hidden_size: int = 128, n_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        # Input projection (optional — can help with small input dims)
        self.input_proj = nn.Linear(n_input, hidden_size)
        
        # Core LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, n_output)
        )
    
    def forward(self, x: torch.Tensor, 
                return_hidden: bool = False) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass.
        
        Args:
            x: (batch, time, n_input) input firing rates
            return_hidden: if True, also return hidden states for probing
        
        Returns:
            y_pred: (batch, time, n_output) predicted output rates
            hidden_states: (batch, time, hidden_size) if return_hidden
        """
        # Project input
        projected = self.input_proj(x)  # (batch, time, hidden_size)
        
        # LSTM forward
        lstm_out, _ = self.lstm(projected)  # (batch, time, hidden_size)
        
        # Project to output
        y_pred = self.output_proj(lstm_out)  # (batch, time, n_output)
        
        if return_hidden:
            return y_pred, lstm_out
        return y_pred
    
    def extract_hidden_states(self, x: torch.Tensor) -> np.ndarray:
        """
        Extract hidden states for DESCARTES probing.
        
        Returns the full hidden state trajectory as a numpy array
        suitable for Ridge regression probing.
        """
        self.eval()
        with torch.no_grad():
            _, hidden = self.forward(x, return_hidden=True)
        return hidden.cpu().numpy()


class MultiArchitectureSurrogate:
    """
    Container for multi-architecture comparison.
    Matches Circuit 4 protocol: LSTM, GRU, Transformer, Linear.
    
    Cross-architecture testing revealed in Circuit 4 that mandatory
    variables are ARCHITECTURE-SPECIFIC, not transformation-specific.
    This must be tested again for the emotion domain.
    """
    
    ARCHITECTURES = {
        'lstm': lambda n_in, n_out, h: LimbicPrefrontalLSTM(n_in, n_out, h),
        'gru': lambda n_in, n_out, h: GRUSurrogate(n_in, n_out, h),
        'transformer': lambda n_in, n_out, h: TransformerSurrogate(n_in, n_out, h),
        'linear': lambda n_in, n_out, h: LinearSurrogate(n_in, n_out, h),
    }
    
    @staticmethod
    def create(arch_name: str, n_input: int, n_output: int, 
               hidden_size: int = 128) -> nn.Module:
        if arch_name not in MultiArchitectureSurrogate.ARCHITECTURES:
            raise ValueError(f"Unknown architecture: {arch_name}")
        return MultiArchitectureSurrogate.ARCHITECTURES[arch_name](
            n_input, n_output, hidden_size)


class GRUSurrogate(nn.Module):
    """GRU variant — same interface as LSTM surrogate."""
    def __init__(self, n_input, n_output, hidden_size=128, n_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_proj = nn.Linear(n_input, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, 
                          batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, n_output))
    
    def forward(self, x, return_hidden=False):
        proj = self.input_proj(x)
        out, _ = self.gru(proj)
        y = self.output_proj(out)
        return (y, out) if return_hidden else y


class TransformerSurrogate(nn.Module):
    """Transformer variant — causal attention for temporal modeling."""
    def __init__(self, n_input, n_output, hidden_size=128, n_heads=4, n_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_proj = nn.Linear(n_input, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=n_heads, batch_first=True,
            dim_feedforward=hidden_size * 2)
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, n_output))
    
    def forward(self, x, return_hidden=False):
        proj = self.input_proj(x)
        # Causal mask: prevent attending to future timesteps
        seq_len = x.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        out = self.transformer(proj, mask=mask)
        y = self.output_proj(out)
        return (y, out) if return_hidden else y


class LinearSurrogate(nn.Module):
    """Linear baseline — no recurrence, no nonlinearity in hidden layer."""
    def __init__(self, n_input, n_output, hidden_size=128, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.proj = nn.Linear(n_input, hidden_size)
        self.output = nn.Linear(hidden_size, n_output)
    
    def forward(self, x, return_hidden=False):
        h = self.proj(x)
        y = self.output(h)
        return (y, h) if return_hidden else y
```

### 5.2 Training Protocol

```python
"""
models/train_surrogate.py

Training loop for the limbic→prefrontal surrogate.
Identical protocol to Circuits 3-4 for comparability.
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


def train_surrogate(model: nn.Module,
                    X_train: np.ndarray,
                    Y_train: np.ndarray,
                    X_val: np.ndarray,
                    Y_val: np.ndarray,
                    n_epochs: int = 200,
                    batch_size: int = 32,
                    learning_rate: float = 1e-3,
                    patience: int = 20,
                    save_dir: str = 'checkpoints',
                    seed: int = 42) -> Dict:
    """
    Train one surrogate model with early stopping.
    
    The training objective is pure output prediction — no biological
    loss term. The DESCARTES framework deliberately avoids incorporating
    biological variables into the training objective because the question
    is whether the model INDEPENDENTLY discovers them.
    
    Args:
        model: surrogate model instance
        X_train: (n_train, T, n_input) training input
        Y_train: (n_train, T, n_output) training targets
        X_val: (n_val, T, n_input) validation input
        Y_val: (n_val, T, n_output) validation targets
        n_epochs: maximum training epochs
        batch_size: batch size
        learning_rate: initial learning rate
        patience: early stopping patience
        save_dir: directory for checkpoints
        seed: random seed for reproducibility
    
    Returns:
        results: dict with training history and best model metrics
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=patience//2, factor=0.5)
    
    # MSE loss on firing rates
    # Poisson loss is an alternative but MSE is more robust for smoothed rates
    criterion = nn.MSELoss()
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    Y_train_t = torch.FloatTensor(Y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    Y_val_t = torch.FloatTensor(Y_val).to(device)
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(n_epochs):
        # ── Training ──
        model.train()
        n_batches = len(X_train_t) // batch_size
        epoch_loss = 0
        
        # Shuffle training data
        perm = torch.randperm(len(X_train_t))
        
        for b in range(n_batches):
            idx = perm[b*batch_size:(b+1)*batch_size]
            x_batch = X_train_t[idx]
            y_batch = Y_train_t[idx]
            
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            
            # Gradient clipping — prevents explosion with long sequences
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
        
        epoch_loss /= max(n_batches, 1)
        
        # ── Validation ──
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, Y_val_t).item()
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(val_loss)
        
        # ── Early stopping ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path / 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} — "
                  f"Train: {epoch_loss:.6f}, Val: {val_loss:.6f}")
    
    # Load best model
    model.load_state_dict(torch.load(save_path / 'best_model.pt'))
    
    # ── Compute cross-condition correlation ──
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_t).cpu().numpy()
    
    # CC per output neuron (averaged across time and windows)
    cc_values = []
    for j in range(Y_val.shape[2]):
        pred_flat = val_pred[:, :, j].flatten()
        true_flat = Y_val[:, :, j].flatten()
        cc = np.corrcoef(pred_flat, true_flat)[0, 1]
        cc_values.append(cc)
    
    mean_cc = np.nanmean(cc_values)
    
    # QUALITY CHECK: CC < 0.3 means probing results will be unreliable
    if mean_cc < 0.3:
        print(f"⚠️ WARNING: Mean CC = {mean_cc:.3f} < 0.3. "
              f"Model quality insufficient for reliable probing.")
    
    return {
        'history': history,
        'best_val_loss': best_val_loss,
        'cc_per_neuron': cc_values,
        'mean_cc': mean_cc,
        'n_epochs_trained': len(history['train_loss']),
    }
```

### 5.3 Training Data Split

For each patient, the 8-minute movie provides ~940 sliding windows (at 2s window, 500ms stride). Split as follows:

```
Temporal split (prevents data leakage from overlapping windows):
  Minutes 0-5 → Training (60%)  → ~560 windows
  Minutes 5-6.5 → Validation (20%) → ~180 windows  
  Minutes 6.5-8 → Test (20%)   → ~180 windows

IMPORTANT: Split by TIME, not by random window shuffling.
Adjacent windows overlap by 75%, so random splitting would leak
training data into the validation/test sets.
```

### 5.4 Multi-Seed Training

```python
# Train 10 random seeds per patient per architecture
# This enables DESCARTES seed-consistency analysis:
# A truly mandatory variable must be mandatory across seeds.

SEEDS = list(range(10))
ARCHITECTURES = ['lstm', 'gru', 'transformer', 'linear']
HIDDEN_SIZES = [64, 128, 256]  # Capacity sweep

# For the primary analysis, use hidden_size=128 and LSTM.
# Cross-architecture and capacity sweeps are secondary analyses.
```

---

## Phase 6: DESCARTES Probing Protocol

### 6.1 Ridge ΔR² Probing

This is the core DESCARTES methodology, identical across all circuits. The key metric is **ΔR² = R²(trained) − R²(untrained)**: the difference in Ridge regression decoding accuracy between a trained surrogate and an untrained (random initialization) control.

**ΔR² interpretation:**
- ΔR² ≈ 0 → variable decodable from random projections (ZOMBIE)
- ΔR² > 0.05 → variable genuinely learned by the surrogate (CANDIDATE)
- ΔR² > 0.15 → strongly learned (INTERESTING)

```python
"""
probing/ridge_probing.py

Standardized Ridge ΔR² probing from descartes_core.
This is architecture-agnostic and reused unchanged from Circuits 1-4.
"""
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from typing import Dict, Tuple


def ridge_delta_r2(hidden_states: np.ndarray,
                   target: np.ndarray,
                   hidden_untrained: np.ndarray,
                   n_folds: int = 5,
                   alphas: np.ndarray = np.logspace(-3, 3, 20)) -> Dict:
    """
    Compute ΔR² = R²(trained hidden) - R²(untrained hidden).
    
    This is THE metric of the DESCARTES framework. It controls for
    the baseline decodability that arises from random projections:
    high-dimensional random vectors can correlate with any target
    by chance, so raw R² is misleading. Only the DELTA tells us
    whether the trained model has genuinely learned the variable.
    
    Args:
        hidden_states: (T, hidden_size) trained model hidden states
        target: (T,) or (T, n_dims) probe target
        hidden_untrained: (T, hidden_size) untrained model hidden states
        n_folds: number of cross-validation folds
        alphas: Ridge regularization parameters to search
    
    Returns:
        results: dict with 'r2_trained', 'r2_untrained', 'delta_r2',
                 'alpha_trained', 'alpha_untrained'
    """
    # Ensure 2D target
    if target.ndim == 1:
        target = target.reshape(-1, 1)
    
    # ── Trained model probing ──
    kf = KFold(n_splits=n_folds, shuffle=False)  # No shuffle: preserves temporal structure
    r2_trained_folds = []
    
    for train_idx, test_idx in kf.split(hidden_states):
        ridge = RidgeCV(alphas=alphas)
        ridge.fit(hidden_states[train_idx], target[train_idx])
        r2 = ridge.score(hidden_states[test_idx], target[test_idx])
        r2_trained_folds.append(r2)
    
    r2_trained = np.mean(r2_trained_folds)
    
    # ── Untrained model probing ──
    r2_untrained_folds = []
    
    for train_idx, test_idx in kf.split(hidden_untrained):
        ridge = RidgeCV(alphas=alphas)
        ridge.fit(hidden_untrained[train_idx], target[train_idx])
        r2 = ridge.score(hidden_untrained[test_idx], target[test_idx])
        r2_untrained_folds.append(r2)
    
    r2_untrained = np.mean(r2_untrained_folds)
    
    # ── Delta ──
    delta_r2 = r2_trained - r2_untrained
    
    return {
        'r2_trained': r2_trained,
        'r2_untrained': r2_untrained,
        'delta_r2': delta_r2,
        'r2_trained_folds': r2_trained_folds,
        'r2_untrained_folds': r2_untrained_folds,
    }


def probe_all_targets(model: 'nn.Module',
                      untrained_model: 'nn.Module',
                      X_test: np.ndarray,
                      probe_targets: Dict[str, np.ndarray],
                      device: str = 'cpu') -> Dict:
    """
    Run Ridge ΔR² probing for all probe targets.
    
    Args:
        model: trained surrogate model
        untrained_model: untrained (random init) model of same architecture
        X_test: (n_windows, T, n_input) test input data
        probe_targets: dict of {target_name: (T_total,) or (T_total, dims)}
        device: torch device string
    
    Returns:
        results: dict of {target_name: probe_result_dict}
    """
    import torch
    
    # Extract hidden states
    model.eval()
    untrained_model.eval()
    
    with torch.no_grad():
        X_t = torch.FloatTensor(X_test).to(device)
        _, hidden_trained = model(X_t, return_hidden=True)
        _, hidden_untrained = untrained_model(X_t, return_hidden=True)
    
    # Flatten across windows: (n_windows, T, hidden) → (n_windows*T, hidden)
    h_trained = hidden_trained.cpu().numpy().reshape(-1, hidden_trained.shape[-1])
    h_untrained = hidden_untrained.cpu().numpy().reshape(-1, hidden_untrained.shape[-1])
    
    results = {}
    for name, target in probe_targets.items():
        # Align target to flattened hidden states
        # Target should be (T_total,) where T_total = n_windows * T_per_window
        # If target is continuous, slice to match test windows
        
        if len(target) != len(h_trained):
            print(f"WARNING: {name} target length ({len(target)}) != "
                  f"hidden length ({len(h_trained)}). Truncating.")
            min_len = min(len(target), len(h_trained))
            target = target[:min_len]
            h_t = h_trained[:min_len]
            h_u = h_untrained[:min_len]
        else:
            h_t = h_trained
            h_u = h_untrained
        
        results[name] = ridge_delta_r2(h_t, target, h_u)
        print(f"  {name}: ΔR² = {results[name]['delta_r2']:.4f} "
              f"(trained={results[name]['r2_trained']:.4f}, "
              f"untrained={results[name]['r2_untrained']:.4f})")
    
    return results
```

---

## Phase 7: Resample Ablation Protocol

### 7.1 Why Resample, Not Mean-Clamp

**This is the most critical methodological lesson from the DESCARTES program.** Circuit 1 (L5PC) proved that mean-clamping produces catastrophically false positive causality findings. Mean-clamping replaces the clamped dimensions with their mean value, but this mean vector is **out-of-distribution (OOD)** for the trained LSTM — no real hidden state ever has all dimensions simultaneously at their mean. The LSTM detects this OOD input and produces degraded output, which is then misinterpreted as "those dimensions were causally necessary."

Resample ablation fixes this by replacing clamped dimensions with values drawn from their **empirical distribution**, maintaining the marginal statistics of each dimension. The LSTM receives inputs that are still within distribution, so any output degradation is genuine evidence of causal necessity.

```python
"""
ablation/resample_ablation.py

Resample ablation from descartes_core.
NEVER use mean-clamping. This is validated across Circuits 1-4.
"""
import numpy as np
from typing import Dict, List, Tuple


def resample_ablation(model,
                      X_test: np.ndarray,
                      Y_test: np.ndarray,
                      dims_to_ablate: List[int],
                      n_resamples: int = 100,
                      device: str = 'cpu') -> Dict:
    """
    Resample ablation: test causal necessity of specific hidden dims.
    
    For each resample iteration:
    1. Run model forward to get hidden states
    2. Replace specified dims with values randomly sampled from
       other timepoints' values for those dims (preserving marginals)
    3. Continue the forward pass from the modified hidden states
    4. Measure output degradation
    
    If output degrades significantly (z-score < -2), the ablated 
    dimensions are causally necessary for the transformation.
    
    CRITICAL: This operates on hidden states DURING the forward pass,
    not on input dimensions. The model runs normally up to the LSTM,
    then hidden state dimensions are resampled before the output layer.
    
    Args:
        model: trained surrogate model
        X_test: (n_windows, T, n_input) test input
        Y_test: (n_windows, T, n_output) test targets
        dims_to_ablate: list of hidden state dimension indices to ablate
        n_resamples: number of resample iterations
        device: torch device
    
    Returns:
        results: dict with z-scores, p-values, output degradation
    """
    import torch
    
    model.eval()
    
    X_t = torch.FloatTensor(X_test).to(device)
    
    # ── Baseline performance (no ablation) ──
    with torch.no_grad():
        y_pred_baseline, hidden_baseline = model(X_t, return_hidden=True)
    
    baseline_mse = np.mean((y_pred_baseline.cpu().numpy() - Y_test) ** 2)
    
    # ── Collect hidden state statistics for resampling ──
    hidden_np = hidden_baseline.cpu().numpy()  # (n_windows, T, hidden_size)
    # Flatten to (n_windows*T, hidden_size) for sampling pool
    hidden_flat = hidden_np.reshape(-1, hidden_np.shape[-1])
    
    # ── Resample iterations ──
    ablated_mses = []
    
    for r in range(n_resamples):
        # Create resampled hidden states
        hidden_resampled = hidden_np.copy()
        
        for dim in dims_to_ablate:
            # Sample replacement values from empirical distribution of this dim
            n_total = hidden_flat.shape[0]
            replacement_idx = np.random.choice(n_total, 
                                                size=hidden_resampled.shape[0] * hidden_resampled.shape[1])
            replacement_values = hidden_flat[replacement_idx, dim]
            hidden_resampled[:, :, dim] = replacement_values.reshape(
                hidden_resampled.shape[0], hidden_resampled.shape[1])
        
        # Forward pass from resampled hidden states through output layer
        with torch.no_grad():
            h_resampled = torch.FloatTensor(hidden_resampled).to(device)
            y_pred_ablated = model.output_proj(h_resampled)
        
        ablated_mse = np.mean((y_pred_ablated.cpu().numpy() - Y_test) ** 2)
        ablated_mses.append(ablated_mse)
    
    # ── Statistical test ──
    ablated_mses = np.array(ablated_mses)
    mean_ablated = np.mean(ablated_mses)
    std_ablated = np.std(ablated_mses)
    
    # Z-score: how much worse is ablated vs baseline?
    # Negative z = ablation hurts performance = dimensions are necessary
    if std_ablated > 0:
        z_score = (baseline_mse - mean_ablated) / std_ablated
    else:
        z_score = 0.0
    
    # Effect size: relative MSE increase
    relative_degradation = (mean_ablated - baseline_mse) / max(baseline_mse, 1e-10)
    
    return {
        'baseline_mse': baseline_mse,
        'mean_ablated_mse': mean_ablated,
        'std_ablated_mse': std_ablated,
        'z_score': z_score,
        'relative_degradation': relative_degradation,
        'n_dims_ablated': len(dims_to_ablate),
        'dims_ablated': dims_to_ablate,
        'all_ablated_mses': ablated_mses,
    }


def find_mandatory_dimensions(model,
                               X_test: np.ndarray,
                               Y_test: np.ndarray,
                               probe_results: Dict,
                               z_threshold: float = -2.0,
                               k_values: List[int] = [10, 25, 50],
                               device: str = 'cpu') -> Dict:
    """
    Identify mandatory hidden dimensions for each probe target.
    
    Strategy:
    1. For each probe target with high ΔR², identify the hidden dims
       most correlated with that target (using Ridge coefficients)
    2. Ablate those dims using resample ablation
    3. If ablation degrades output (z < threshold), the target is MANDATORY
    
    This is the pipeline that identified gamma_amp as mandatory in Circuit 2
    and theta_modulation as mandatory in Circuit 3.
    """
    results = {}
    
    for target_name, probe_result in probe_results.items():
        if probe_result['delta_r2'] < 0.05:
            # Skip targets that weren't genuinely learned
            results[target_name] = {
                'status': 'ZOMBIE (not learned)',
                'delta_r2': probe_result['delta_r2']
            }
            continue
        
        print(f"\n--- Ablation test: {target_name} (ΔR²={probe_result['delta_r2']:.4f}) ---")
        
        # For each k value (number of dims to ablate)
        target_results = {}
        for k in k_values:
            # Identify top-k dims most associated with this probe target
            # (Use Ridge coefficient magnitudes from the probing step)
            # This requires access to the Ridge model — skeleton here
            
            # For now: ablate random k dims as baseline comparison
            hidden_size = model.hidden_size
            dims = np.random.choice(hidden_size, size=min(k, hidden_size), replace=False)
            
            abl_result = resample_ablation(model, X_test, Y_test, 
                                           dims.tolist(), device=device)
            
            target_results[f'k={k}'] = abl_result
            
            status = "MANDATORY" if abl_result['z_score'] < z_threshold else "NOT MANDATORY"
            print(f"  k={k}: z={abl_result['z_score']:.2f}, "
                  f"degradation={abl_result['relative_degradation']:.4f} → {status}")
        
        results[target_name] = target_results
    
    return results
```

---

## Phase 8: Cross-Patient Universality Test

### 8.1 Per-Patient Pipeline

Following the Circuit 4 protocol, run the entire pipeline independently per patient:

```python
"""
analysis/cross_patient_universality.py

Per-patient DESCARTES pipeline and cross-patient comparison.
Identical protocol to Circuit 4 (DANDI 000576).
"""

def run_patient_pipeline(patient_id: str,
                          nwb_filepath: str,
                          config: Dict) -> Dict:
    """
    Complete DESCARTES pipeline for one patient.
    
    Steps:
    1. Preprocess neural data (sliding windows)
    2. Compute probe targets (emotion, dynamics, memory)
    3. Train surrogate (10 seeds × primary architecture)
    4. Ridge ΔR² probing (all targets × all seeds)
    5. Resample ablation (for targets with high ΔR²)
    6. Aggregate results
    
    This function should be called once per included patient.
    """
    results = {
        'patient_id': patient_id,
        'preprocessing': {},
        'model_quality': {},
        'probing': {},
        'ablation': {},
    }
    
    # Step 1: Preprocess
    # (calls prepare_movie_data from Phase 2)
    
    # Step 2: Compute probe targets
    # (calls all functions from Phase 4)
    
    # Step 3: Train surrogates
    # (calls train_surrogate from Phase 5, 10 seeds)
    
    # Step 4: Ridge ΔR² probing
    # (calls probe_all_targets from Phase 6)
    
    # Step 5: Resample ablation
    # (calls find_mandatory_dimensions from Phase 7)
    
    # Step 6: Aggregate
    
    return results


def cross_patient_summary(all_results: List[Dict]) -> Dict:
    """
    Aggregate results across patients.
    
    Key questions:
    1. Which probe targets are mandatory in ≥50% of patients?
    2. Are there distinct computational strategies (as in Circuit 4)?
    3. Is theta mandatory universally or individually variable?
    """
    n_patients = len(all_results)
    
    summary = {}
    
    # For each probe target, count how many patients show it as mandatory
    all_targets = set()
    for r in all_results:
        all_targets.update(r.get('ablation', {}).keys())
    
    for target in all_targets:
        mandatory_count = 0
        learned_count = 0
        delta_r2_values = []
        
        for r in all_results:
            if target in r.get('probing', {}):
                dr2 = r['probing'][target]['delta_r2']
                delta_r2_values.append(dr2)
                if dr2 > 0.05:
                    learned_count += 1
            
            if target in r.get('ablation', {}):
                # Check if mandatory at any k
                for k_result in r['ablation'][target].values():
                    if isinstance(k_result, dict) and k_result.get('z_score', 0) < -2.0:
                        mandatory_count += 1
                        break
        
        summary[target] = {
            'mandatory_fraction': mandatory_count / n_patients,
            'learned_fraction': learned_count / n_patients,
            'mean_delta_r2': np.mean(delta_r2_values) if delta_r2_values else 0,
            'mandatory_count': mandatory_count,
            'n_patients': n_patients,
        }
    
    return summary
```

### 8.2 Cross-Domain Comparison

The ultimate analysis: compare mandatory variables across all five circuits.

```python
"""
analysis/cross_domain_comparison.py

Compare mandatory variables across Circuits 1-5.
This is the paper's central figure.
"""

def build_cross_domain_table():
    """
    Construct the cross-domain mandatory variable comparison table.
    
    ┌──────────────────────┬─────────┬──────────┬──────────┬──────────┬──────────┐
    │ Variable             │ C1:L5PC │ C2:HC    │ C3:Mouse │ C4:Human │ C5:Movie │
    │                      │         │ Memory   │ WM       │ WM       │ Emotion  │
    ├──────────────────────┼─────────┼──────────┼──────────┼──────────┼──────────┤
    │ theta_modulation     │ N/A     │ Mandatory│ Mandatory│ Variable │ ???      │
    │ gamma_amplitude      │ N/A     │ Mandatory│ N/A      │ N/A      │ ???      │
    │ population_synchrony │ N/A     │ N/A      │ N/A      │ Variable │ ???      │
    │ temporal_stability   │ N/A     │ N/A      │ Mandatory│ Variable │ ???      │
    │ arousal              │ N/A     │ N/A      │ N/A      │ N/A      │ ???      │
    │ valence              │ N/A     │ N/A      │ N/A      │ N/A      │ ???      │
    │ event_boundaries     │ N/A     │ N/A      │ N/A      │ N/A      │ ???      │
    │ encoding_success     │ N/A     │ N/A      │ N/A      │ N/A      │ ???      │
    └──────────────────────┴─────────┴──────────┴──────────┴──────────┴──────────┘
    
    The prediction: If theta fills the ??? with "Mandatory", theta is
    the universal computational signature of the limbic-prefrontal axis.
    """
    pass  # Implement as visualization after running Circuit 5
```

---

## Phase 9: Claude Code Task Sequence

### Task 0: Data Access and Exploration (PREREQUISITE — DO FIRST)
```
Install dependencies (pynwb, dandi, torch, sklearn, mne, scipy)
Download DANDI 000623 (or stream for initial exploration)
Clone bmovie-release-NWB-BIDS for annotations and reference code

Run exploration/explore_nwb_structure.py on ALL subject files
Record: region column name, region labels, trial columns, LFP structure,
        eye tracking structure, interval tables

Build patient inventory: exploration/patient_inventory.py
Identify included patients (≥5 input + ≥5 output neurons)

DELIVERABLE: patient_inventory.json with per-patient neuron counts
```

### Task 1: Annotation Discovery and Alignment
```
Inspect assets/annotations/ in the bmovie repository
Identify available annotation types and formats
Load and parse each annotation file

Determine movie start/end times from NWB TTL markers or trial structure
Align all annotations to neural data time base

If arousal/valence ratings are not available:
  Option A: Extract pupil diameter as arousal proxy
  Option B: Compute arousal from scene cut density
  Option C: Search literature for published ratings of this film clip

DELIVERABLE: aligned_annotations.npz per patient
```

### Task 2: Preprocessing Pipeline
```
Create preprocessing/sliding_window.py (from Phase 2.2)
Create preprocessing/event_segmentation.py (from Phase 2.3)
Create preprocessing/recognition_memory.py (from Phase 2.4)

For each included patient:
  Extract input region spike times (amygdala + hippocampus)
  Extract output region spike times (ACC + preSMA + vmPFC)
  Bin at 20 ms, smooth with σ=2 bins
  Create sliding windows (2s window, 500ms stride)
  Split 60/20/20 by TIME (not random)
  Save as preprocessed_data/{patient_id}.npz

DELIVERABLE: preprocessed data files for all included patients
```

### Task 3: Probe Target Computation
```
Create probing/probe_targets.py (from Phase 4)

For each included patient, compute ALL probe targets:
  EMOTION: valence, arousal, threat_PE, emotional_category
  NARRATIVE: event_boundaries, face_presence
  DYNAMICS: theta_mod, gamma_mod, population_synchrony, temporal_stability
  MEMORY: encoding_success (from recognition test)

LFP extraction for theta/gamma requires:
  Identify LFP channels from input regions
  Bandpass filter and Hilbert transform
  Downsample to match spike rate binning

DELIVERABLE: probe_targets/{patient_id}.npz with all target time series
```

### Task 4: Surrogate Training
```
Create models/surrogate_lstm.py (from Phase 5)
Create models/train_surrogate.py (from Phase 5)

For each included patient:
  For each of 10 seeds:
    Train LSTM (hidden=128) with early stopping
    Record: training history, val loss, CC per neuron, mean CC
    
    QUALITY CHECK: if mean CC < 0.3, flag this patient/seed
    
    Save model checkpoint and training metrics

DELIVERABLE: checkpoints/{patient_id}/seed_{s}/best_model.pt
             training_metrics/{patient_id}/seed_{s}/metrics.json
```

### Task 5: Ridge ΔR² Probing
```
Create probing/ridge_probing.py (from Phase 6)

For each included patient:
  For each trained model (10 seeds):
    Create UNTRAINED model (same architecture, random init)
    Extract hidden states from trained and untrained on TEST data
    
    For each probe target:
      Run Ridge ΔR² (5-fold CV, alpha search)
      Record: R²_trained, R²_untrained, ΔR²
    
    CRITICAL: Use same random seed for untrained model within each comparison

DELIVERABLE: probing_results/{patient_id}/seed_{s}/delta_r2.json
```

### Task 6: Resample Ablation
```
Create ablation/resample_ablation.py (from Phase 7)

For each probe target with mean ΔR² > 0.05 across seeds:
  Identify top-k hidden dimensions (from Ridge coefficients)
  Run resample ablation at k = [10, 25, 50, hidden_size*0.39, hidden_size*0.94]
  
  These k values are chosen to match Circuit 2 protocol:
    k=50 (39% of 128 dims): the threshold where gamma_amp broke in Circuit 2
    k=120 (94% of 128 dims): near-total ablation
  
  Record z-scores and relative degradation

DELIVERABLE: ablation_results/{patient_id}/seed_{s}/{target_name}.json
```

### Task 7: Cross-Patient Universality Analysis
```
Create analysis/cross_patient_universality.py (from Phase 8)

Aggregate across all patients:
  For each probe target:
    Fraction of patients where mandatory (z < -2)
    Fraction of patients where learned (ΔR² > 0.05)
    Mean and std of ΔR² across patients
    Seed consistency (mandatory in ≥8/10 seeds?)
  
  Test for computational strategies:
    Cluster patients by mandatory variable profiles
    Compare to Circuit 4 oscillatory vs rate-based dichotomy

DELIVERABLE: universality_summary.json
```

### Task 8: Cross-Architecture Comparison (SECONDARY)
```
For 3-5 representative patients, repeat Tasks 4-6 with:
  GRU, Transformer, Linear surrogates

Compare mandatory variables across architectures.
Circuit 4 showed architecture-specific mandatoriness — test if this
holds for emotion processing.

DELIVERABLE: architecture_comparison.json
```

### Task 9: Cross-Domain Comparison (FINAL ANALYSIS)
```
Create analysis/cross_domain_comparison.py (from Phase 8.2)

Load results from Circuits 1-4 (from existing descartes_core results)
Combine with Circuit 5 results

Build the cross-domain table:
  Which variables are mandatory across multiple domains?
  Is theta universal?
  Are emotion-specific variables truly domain-specific?

Generate publication figures:
  - 5-circuit comparison heatmap
  - Theta mandatoriness across circuits
  - Patient-level variability comparison

DELIVERABLE: cross_domain_results.json, publication figures
```

---

## Phase 10: Expected Results and Interpretation

### 10.1 Predicted Outcomes

**Strong prediction (theta universality):**
If theta_modulation is mandatory in ≥40% of patients in Circuit 5 (emotion), after being mandatory in Circuits 2 (memory), 3 (WM), and 4 (human WM), the narrative is: **theta oscillation is the universal computational clock of the limbic-prefrontal axis.** Any LSTM that learns to transform limbic input to prefrontal output must track theta, regardless of whether the cognitive task involves memory encoding, working memory maintenance, or emotional processing.

**Alternative (domain specificity):**
If emotion-specific variables (arousal, valence, threat PE) are mandatory but theta is not, the narrative shifts to: **mandatory variables are domain-specific.** Memory circuits require theta; emotion circuits require arousal tracking. The universal constraint is not a specific oscillation but rather the existence of *some* mandatory intermediate — the identity of that intermediate depends on what the transformation computes.

**Null result (universal zombie):**
If nothing is mandatory in the emotion domain (like Circuit 1 L5PC), this extends the zombie gradient: single neurons and emotion processing produce zombies; only working memory circuits produce non-zombie surrogates. This would suggest that mandatory variables require recurrent dynamics under explicit temporal demands.

### 10.2 Publication Narrative

The five-circuit program tells a story about computational inevitability:

```
From ion channels to emotion:
  Which computations are so fundamental that any system solving 
  the same transformation must discover them?

Single neuron: none. The transformation is simple enough to solve
  without meaningful intermediates. Universal zombie.

Hippocampal memory: one — gamma oscillation amplitude. The CA3→CA1 
  transformation cannot be solved without tracking oscillatory timing.

Working memory (mouse): many — theta, choice, stability. The 
  ALM→thalamus transformation requires a rich internal model.

Working memory (human): individually variable — different brains
  use different mandatory intermediates for the same task.

Emotion (THIS): ???
```

---

## Appendix A: Key Parameters Reference

```
┌──────────────────────────────────┬──────────┬──────────────────────────────┐
│ Parameter                        │ Value    │ Rationale                    │
├──────────────────────────────────┼──────────┼──────────────────────────────┤
│ Bin width                        │ 20 ms    │ Resolves theta modulation    │
│ Gaussian smoothing σ             │ 2 bins   │ 40 ms kernel, standard       │
│ Sliding window length            │ 2000 ms  │ ~8-16 theta cycles           │
│ Sliding window stride            │ 500 ms   │ 75% overlap, ~940 windows    │
│ LSTM hidden size (primary)       │ 128      │ Matches Circuits 3-4         │
│ LSTM layers                      │ 2        │ Matches Circuits 3-4         │
│ Training epochs (max)            │ 200      │ Early stopping at patience=20│
│ Learning rate                    │ 1e-3     │ Adam optimizer               │
│ Random seeds                     │ 10       │ Seed consistency analysis    │
│ Ridge CV alphas                  │ 1e-3–1e3 │ 20 log-spaced values         │
│ Ridge CV folds                   │ 5        │ Temporal (not shuffled)       │
│ ΔR² learned threshold            │ 0.05     │ Conservative, from Circuit 2 │
│ Ablation z-score threshold       │ -2.0     │ p < 0.05 equivalent          │
│ Resample iterations              │ 100      │ Sufficient for z estimation  │
│ Min input neurons                │ 5        │ Per-region minimum            │
│ Min output neurons               │ 5        │ Per-region minimum            │
│ Min CC for valid model           │ 0.3      │ Below this, probing invalid  │
│ Theta band                       │ 4-8 Hz   │ Standard hippocampal theta   │
│ Gamma band                       │ 30-80 Hz │ Matches Circuit 2 definition │
│ Train/val/test split             │ 60/20/20 │ Temporal split, no shuffle   │
└──────────────────────────────────┴──────────┴──────────────────────────────┘
```

## Appendix B: Relationship to Other Circuits

```
Circuit 1 (L5PC):        Simulated single neuron → biophysical ground truth
Circuit 2 (CA3→CA1):     Simulated circuit → gamma_amp discovery
Circuit 3 (ALM→Thalamus): Real mouse data (DANDI 000363) → theta, choice, stability
Circuit 4 (MTL→Frontal):  Real human data (DANDI 000576) → individual variability
Circuit 5 (THIS):         Real human data (DANDI 000623) → emotion/cross-domain test

Pipeline reuse:
  descartes_core (Ridge ΔR², resample ablation, classification) → reuse unchanged
  human_wm NWB loader → template for Rutishauser lab NWB format
  multi-architecture surrogate code → reuse unchanged
  Sliding window preprocessing → NEW for continuous data
  Emotion probe targets → NEW domain-specific targets
  Annotation alignment → NEW for movie paradigm
```

## Appendix C: Risk Mitigation

```
Risk: Insufficient neurons in both input and output regions
  Mitigation: Relax MIN_NEURONS thresholds, pool bilateral regions,
              consider using LFP-derived features as supplementary input

Risk: No arousal/valence annotations available
  Mitigation: Use pupil diameter proxy, scene cut density proxy,
              or conduct self-annotation of the 8-minute clip

Risk: Continuous data makes probe target alignment imprecise
  Mitigation: Use multiple segmentation strategies (sliding window + 
              event-based) and verify consistency

Risk: Movie too short (8 min) for reliable LSTM training
  Mitigation: Heavy window overlap (75%) + cross-patient data pooling
              for architecture validation

Risk: Recognition memory test has too few trials for encoding success
  Mitigation: Analyze encoding success as secondary probe target;
              primary analysis focuses on movie-phase dynamics

Risk: Results are architecture-specific (as in Circuit 4)
  Mitigation: Multi-architecture comparison on representative patients
              (LSTM, GRU, Transformer, Linear)
```
