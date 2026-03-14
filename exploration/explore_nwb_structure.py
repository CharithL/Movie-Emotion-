"""
exploration/explore_nwb_structure.py

TASK 0: Run this FIRST on every subject file.
Discover the actual NWB structure before writing any analysis code.
"""
import pynwb
import numpy as np
from pathlib import Path
import json
import sys


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

        # -- Basic metadata --
        print(f"\nSession: {nwb.session_description}")
        print(f"Subject: {nwb.subject}")
        print(f"Session start: {nwb.session_start_time}")
        print(f"Identifier: {nwb.identifier}")

        # -- Units (single neurons) --
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

            # -- Brain region distribution --
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

        # -- Electrodes --
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

        # -- Electrode Groups --
        print(f"\n--- ELECTRODE GROUPS ---")
        for name, group in nwb.electrode_groups.items():
            print(f"  {name}: location={group.location}, "
                  f"description={group.description}")

        # -- Processing modules --
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
                if hasattr(container, 'spatial_series'):
                    print(f"      Spatial series keys: {list(container.spatial_series.keys())}")
                    for ss_name, ss in container.spatial_series.items():
                        try:
                            print(f"        {ss_name}: shape={ss.data.shape}, rate={getattr(ss, 'rate', 'N/A')} Hz")
                        except:
                            print(f"        {ss_name}: (could not read shape)")

        # -- Acquisition (raw data) --
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
            if hasattr(acq, 'electrodes') and acq.electrodes is not None:
                try:
                    print(f"    Electrodes table: {len(acq.electrodes)} entries")
                except:
                    pass

        # -- Trials / Epochs --
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
            # Print first 3 trial rows
            print(f"\n  First 3 trials:")
            for i in range(min(3, len(nwb.trials))):
                row = {}
                for col in nwb.trials.colnames:
                    try:
                        row[col] = str(nwb.trials[col][i])[:50]
                    except:
                        row[col] = 'ERROR'
                print(f"    Trial {i}: {row}")

        # -- Intervals --
        print(f"\n--- INTERVALS ---")
        if hasattr(nwb, 'intervals') and nwb.intervals is not None:
            for name, interval in nwb.intervals.items():
                print(f"  {name}: {len(interval)} intervals")
                if hasattr(interval, 'colnames'):
                    print(f"    Columns: {interval.colnames}")

        # -- Stimulus --
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
                subject_id = f.stem
                n_units = len(nwb.units) if nwb.units else 0
                n_trials = len(nwb.trials) if nwb.trials else 0

                # Get region distribution via electrode_group column or
                # by resolving through electrodes table -> group -> location
                region_counts = {}
                if nwb.units and n_units > 0:
                    if 'electrode_group' in nwb.units.colnames:
                        for i in range(n_units):
                            eg = nwb.units['electrode_group'][i]
                            region = eg.location if hasattr(eg, 'location') else str(eg)
                            region_counts[region] = region_counts.get(region, 0) + 1
                    elif 'electrodes' in nwb.units.colnames:
                        # electrodes is a DynamicTableRegion -> pandas DataFrame/Series
                        # We need to get the electrode indices and look up their group
                        for i in range(n_units):
                            try:
                                electrode_row = nwb.units['electrodes'][i]
                                # electrode_row is a DataFrame; get the group_name or group
                                if hasattr(electrode_row, 'iloc'):
                                    # It's a DataFrame, get first row's group
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
                    else:
                        # Try brain_area or location directly
                        for col_candidate in ['brain_area', 'location']:
                            if col_candidate in nwb.units.colnames:
                                for i in range(n_units):
                                    region = str(nwb.units[col_candidate][i])
                                    region_counts[region] = region_counts.get(region, 0) + 1
                                break

                # Also record electrode_groups info
                electrode_groups_info = {}
                for name, group in nwb.electrode_groups.items():
                    electrode_groups_info[name] = group.location

                summary[subject_id] = {
                    'n_units': n_units,
                    'n_trials': n_trials,
                    'regions': region_counts,
                    'electrode_groups': electrode_groups_info
                }
                print(f"\n{subject_id}: {n_units} units, {n_trials} trials")
                for region, count in sorted(region_counts.items()):
                    print(f"  {region}: {count}")
        except Exception as e:
            print(f"\nERROR reading {f.name}: {e}")
            import traceback
            traceback.print_exc()

    return summary


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/000623"
    summary = explore_all_subjects(data_dir)

    # Save summary as JSON
    serializable_summary = {}
    for k, v in summary.items():
        serializable_summary[k] = {
            'n_units': v['n_units'],
            'n_trials': v['n_trials'],
            'regions': v['regions'],
            'electrode_groups': v.get('electrode_groups', {})
        }

    with open('exploration_summary.json', 'w') as f:
        json.dump(serializable_summary, f, indent=2)
    print(f"\nSummary saved to exploration_summary.json")
