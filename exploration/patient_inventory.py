"""
exploration/patient_inventory.py

Build the inclusion/exclusion table.
Determine which patients have sufficient neurons for the
limbic->prefrontal transformation model.

Uses the exact region labels discovered from explore_nwb_structure.py:
  Left/Right amygdala, Left/Right hippocampus,
  Left/Right ACC, Left/Right preSMA, Left/Right vmPFC
"""
import json
import pynwb
import numpy as np
from pathlib import Path
from collections import defaultdict

# Exact region labels from exploration (Phase 0.3 output)
INPUT_REGIONS = {
    'amygdala': ['Left amygdala', 'Right amygdala'],
    'hippocampus': ['Left hippocampus', 'Right hippocampus']
}

OUTPUT_REGIONS = {
    'ACC': ['Left ACC', 'Right ACC'],
    'preSMA': ['Left preSMA', 'Right preSMA'],
    'vmPFC': ['Left vmPFC', 'Right vmPFC']
}

# Minimum neuron counts
MIN_INPUT_NEURONS = 5
MIN_OUTPUT_NEURONS = 5
MIN_TOTAL_NEURONS = 15


def classify_neuron_region(region_label):
    """Map a neuron's region label to canonical grouping."""
    for group_name, labels in INPUT_REGIONS.items():
        if region_label in labels:
            return 'input', group_name
    for group_name, labels in OUTPUT_REGIONS.items():
        if region_label in labels:
            return 'output', group_name
    return 'other', region_label


def get_region_counts_from_nwb(filepath):
    """Extract per-region neuron counts from a single NWB file."""
    region_counts = {}
    n_units = 0
    n_trials = 0

    with pynwb.NWBHDF5IO(str(filepath), 'r') as io:
        nwb = io.read()
        n_units = len(nwb.units) if nwb.units else 0
        n_trials = len(nwb.trials) if nwb.trials else 0

        if nwb.units and n_units > 0 and 'electrodes' in nwb.units.colnames:
            for i in range(n_units):
                try:
                    electrode_row = nwb.units['electrodes'][i]
                    if hasattr(electrode_row, 'iloc'):
                        if 'group_name' in electrode_row.columns:
                            gname = electrode_row['group_name'].iloc[0]
                            if gname in nwb.electrode_groups:
                                region = nwb.electrode_groups[gname].location
                            else:
                                region = gname
                        elif 'location' in electrode_row.columns:
                            region = electrode_row['location'].iloc[0]
                        else:
                            region = 'unknown'
                    else:
                        region = str(electrode_row)
                    region_counts[region] = region_counts.get(region, 0) + 1
                except Exception:
                    region_counts['parse_error'] = region_counts.get('parse_error', 0) + 1

    return n_units, n_trials, region_counts


def build_patient_inventory(data_dir):
    """Build inclusion/exclusion table across all subjects."""
    data_path = Path(data_dir)
    nwb_files = sorted(data_path.glob("**/*.nwb"))

    # Group files by subject (use R1 = movie session preferentially)
    subject_files = {}
    for f in nwb_files:
        # Extract subject ID from filename like sub-CS41_ses-P41CSR1_...
        subject = f.parent.name  # e.g., "sub-CS41"
        session_type = 'R1' if 'R1' in f.stem else 'R2'
        if subject not in subject_files:
            subject_files[subject] = {}
        subject_files[subject][session_type] = f

    print(f"Found {len(subject_files)} subjects")

    inventory = []
    for subject_id in sorted(subject_files.keys()):
        files = subject_files[subject_id]
        # Use R1 (movie session) as primary, fall back to R2
        primary_file = files.get('R1', files.get('R2'))
        session_label = 'R1' if 'R1' in files else 'R2'

        n_units, n_trials, region_counts = get_region_counts_from_nwb(primary_file)

        # Also get R2 counts if available
        r2_info = None
        if 'R2' in files:
            n_u2, n_t2, rc2 = get_region_counts_from_nwb(files['R2'])
            r2_info = {'n_units': n_u2, 'n_trials': n_t2, 'regions': rc2}

        # Classify neurons
        input_count = 0
        output_count = 0
        region_detail = {}

        for raw_label, count in region_counts.items():
            role, canonical = classify_neuron_region(raw_label)
            if role == 'input':
                input_count += count
                region_detail[f'input_{canonical}'] = region_detail.get(
                    f'input_{canonical}', 0) + count
            elif role == 'output':
                output_count += count
                region_detail[f'output_{canonical}'] = region_detail.get(
                    f'output_{canonical}', 0) + count
            else:
                region_detail[f'other_{raw_label}'] = count

        # Build patient record
        patient = {
            'subject_id': subject_id,
            'primary_session': session_label,
            'has_R2': 'R2' in files,
            'total_units': n_units,
            'n_trials': n_trials,
            'n_input': input_count,
            'n_output': output_count,
            'region_detail': region_detail,
            'raw_regions': region_counts,
            'included': (
                input_count >= MIN_INPUT_NEURONS and
                output_count >= MIN_OUTPUT_NEURONS and
                (input_count + output_count) >= MIN_TOTAL_NEURONS
            )
        }

        # Exclusion reasons
        reasons = []
        if input_count < MIN_INPUT_NEURONS:
            reasons.append(f"input neurons too few ({input_count} < {MIN_INPUT_NEURONS})")
        if output_count < MIN_OUTPUT_NEURONS:
            reasons.append(f"output neurons too few ({output_count} < {MIN_OUTPUT_NEURONS})")
        if (input_count + output_count) < MIN_TOTAL_NEURONS:
            reasons.append(f"total too few ({input_count + output_count} < {MIN_TOTAL_NEURONS})")
        patient['exclusion_reason'] = '; '.join(reasons) if reasons else 'INCLUDED'

        if r2_info:
            patient['R2_n_units'] = r2_info['n_units']

        inventory.append(patient)

    return inventory


def print_inventory_table(inventory):
    """Print a formatted inclusion/exclusion table."""
    print(f"\n{'='*100}")
    print("PATIENT INVENTORY FOR DESCARTES CIRCUIT 5")
    print(f"{'='*100}")
    print(f"{'Subject':<12} {'Total':<7} {'Input':<7} {'Output':<7} "
          f"{'Inc?':<6} {'Reason'}")
    print(f"{'-'*100}")

    included_count = 0
    for p in sorted(inventory, key=lambda x: x['n_input'] + x['n_output'], reverse=True):
        status = "YES" if p['included'] else "NO"
        print(f"{p['subject_id']:<12} {p['total_units']:<7} {p['n_input']:<7} "
              f"{p['n_output']:<7} {status:<6} {p['exclusion_reason']}")
        if p['included']:
            included_count += 1

    print(f"\n{included_count} / {len(inventory)} patients INCLUDED")

    # Region breakdown for included patients
    print(f"\nRegion breakdown (included patients only):")
    for p in sorted(inventory, key=lambda x: x['subject_id']):
        if p['included']:
            detail = ', '.join(f"{k}: {v}" for k, v in
                              sorted(p['region_detail'].items()))
            print(f"  {p['subject_id']}: {detail}")


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/000623"
    inventory = build_patient_inventory(data_dir)
    print_inventory_table(inventory)

    # Save as JSON
    output_path = "patient_inventory.json"
    with open(output_path, 'w') as f:
        json.dump(inventory, f, indent=2, default=str)
    print(f"\nInventory saved to {output_path}")
