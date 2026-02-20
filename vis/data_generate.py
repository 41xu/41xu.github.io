# we should have a script to generate the data.json for us...
"""
final format in data.json:
{id:{"motions": "", "motiont": "", "captions": "", "captiont": "", "text1": gt, "text2": ours}}
"""
import json
import os
import argparse

def load_json(filename):
    """Safely load a JSON file."""
    if not os.path.exists(filename):
        # Return empty structure if file missing
        return {}
    with open(filename, "r") as f:
        return json.load(f)

def load_generated_map(filename):
    """Loads a result file and returns a dict mapping id -> generated text."""
    data = load_json(filename)
    if isinstance(data, list):
         return {item["id"]: item.get("generated", "") for item in data}
    return {}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter_ids", type=str, help="Path to a JSON file containing a list of IDs to filter by")
    args = parser.parse_args()

    # 1. Base Data
    datas = load_json("amt_motionfix_latest.json") # text1, gt
    splits = load_json("splits.json")
    captions = load_json("mofix_cap.json")

    # 2. Files Mapping
    results_files = {
        "text2": "base.json",
        "text3": "base+bp+cont_poe+difftok+sep.json",
        "text4": "mardm.json",
        "text5": "simple_h3d.json",
        "text6": "stmc.json"
        # "text6": "demo_contrastive_poe.json",
        # "text7": "demo_simple_h3d.json",
    }

    # 3. Load Results
    results_data = {}
    for key, fname in results_files.items():
        results_data[key] = load_generated_map(fname)

    # --- Load Metrics ---
    # Define mapping from result key (demo/ours) to metrics filename
    # Keys here correspond to how we want to label them in the final JSON: "demo" vs "ours"
    # Matches the structure requested: metrics: { demo: {...}, ours: {...} }
    metrics_files_map = {
        "demo": "metrics/base.json",
        "ours": "metrics/base+bp+cont_poe+difftok+sep.json",
        "mardm": "metrics/mardm.json",
        "simple_h3d": "metrics/simple_h3d.json",
        "stmc": "metrics/stmc_metrics.json"
    }
    
    loaded_metrics = {} # Stores { "demo": { average_metrics:..., per_sample_metrics:... }, ... }
    
    for label, fname in metrics_files_map.items():
        if os.path.exists(fname):
             with open(fname, 'r') as f:
                 loaded_metrics[label] = json.load(f)
        else:
             print(f"Warning: Metrics file not found: {fname}")
             loaded_metrics[label] = {"average_metrics": {}, "per_sample_metrics": {}}

    # Define which metrics keys we want to keep and rename if necessary
    # User asked for BERT, BLEURT, CIDEr, METEOR. 
    # File has BERTScore_F1 (commonly used as BERT), BLEURT, CIDEr, METEOR.
    METRIC_KEYS_MAP = {
        "BERTScore_F1": "BERT", 
        "BLEURT": "BLEURT",
        "CIDEr": "CIDEr",
        "METEOR": "METEOR"
    }

    def filter_metrics(m_dict):
        out = {}
        for src_key, target_key in METRIC_KEYS_MAP.items():
            if src_key in m_dict:
                out[target_key] = m_dict[src_key]
        return out

    # 4. Build Output
    test_ids = splits.get("test", [])
    
    # --- FILTER LOGIC ---
    # Set TARGET_IDS to a list of strings to only generate data for those IDs.
    TARGET_IDS = None 

    if args.filter_ids:
        if os.path.exists(args.filter_ids):
             with open(args.filter_ids, 'r') as f:
                 loaded_ids = json.load(f)
                 if isinstance(loaded_ids, list):
                     TARGET_IDS = [str(x) for x in loaded_ids]
                 elif isinstance(loaded_ids, dict):
                     TARGET_IDS = [str(x) for x in loaded_ids.keys()]
                 else:
                     print("Error: Filter file format not recognized (must be list or dict keys)")
        else:
             print(f"Error: Filter file {args.filter_ids} not found")
    
    if TARGET_IDS:
        test_ids_filtered = [mid for mid in test_ids if mid in TARGET_IDS]
        if not test_ids_filtered and TARGET_IDS:
             # Fallback if no intersection found (maybe user gave IDs not in test set?)
             # User might want to generate even if not in 'test' split if forced?
             # But main loop iterates over test_ids.
             # If user supplied IDs are not in `splits['test']`, they will be ignored by current logic.
             # If user wants to force them, we should probably use TARGET_IDS as base if provided.
             print(f"Warning: None of the filter IDs were found in the test set.")
        
        # Actually, if we have a filter, we probably want to use IT as the source of truth, 
        # provided they exist in `datas` (the big amt_motionfix json).
        # Let's intersection with `datas` keys instead of `test_ids` if strictly filtering.
        # But for now, let's just stick to filtering the test set as requested.
        test_ids = test_ids_filtered
        print(f"Filtering: Keeping {len(test_ids)} IDs from target list.")
    # --------------------

    final_output = {}

    for mid in test_ids:
        if mid not in datas:
            continue

        entry = {
            "motions": datas[mid].get("motion_a", ""),
            "motiont": datas[mid].get("motion_b", ""),
            "overlap": datas[mid].get("motion_overlaid", ""), 
            "captions": captions.get(f"{mid}_s", ""),
            "captiont": captions.get(f"{mid}_t", ""),
            "text1": datas[mid].get("annotation", ""), 
        }

        # Populate mapped texts
        for key in results_files:
            entry[key] = results_data[key].get(mid, "")
            
        # Populate metrics
        # Structure: metrics: { demo: {BERT: ...}, ours: {BERT: ...} }
        entry_metrics = {}
        for label, data in loaded_metrics.items():
            per_sample = data.get("per_sample_metrics", {})
            # Note: per_sample keys might be "000004" (string) or int. Check json content. 
            # Usually IDs in json keys are strings.
            if mid in per_sample:
                entry_metrics[label] = filter_metrics(per_sample[mid])
            else:
                entry_metrics[label] = {} 
        
        entry["metrics"] = entry_metrics

        final_output[f"{mid}_test"] = entry

    # 5. Add Dataset Level Metrics
    dataset_metrics = {}
    for label, data in loaded_metrics.items():
        avg = data.get("average_metrics", {})
        dataset_metrics[label] = filter_metrics(avg)
    
    final_output["DATASET_METRICS"] = dataset_metrics

    # 6. Save
    with open("data.json", "w") as f:
        json.dump(final_output, f, indent=4)
        
    print("Done. Generated data.json with metrics.")

if __name__ == "__main__":
    main()
