# we should have a script to generate the data.json for us...
"""
final format in data.json:
{id:{"motions": "", "motiont": "", "captions": "", "captiont": "", "text1": gt, "text2": ours}}
"""
import json
import os

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
    # 1. Base Data
    datas = load_json("amt_motionfix_latest.json") # text1, gt
    splits = load_json("splits.json")
    captions = load_json("mofix_cap.json")

    # 2. Files Mapping
    results_files = {
        "text2": "base.json",
        "text3": "base+bp+cont_poe+difftok+sep.json",
        # "text4": "demo_poe.json",
        # "text5": "demo_contrastive.json",
        # "text6": "demo_contrastive_poe.json",
        # "text7": "demo_simple_h3d.json",
    }

    # 3. Load Results
    results_data = {}
    for key, fname in results_files.items():
        results_data[key] = load_generated_map(fname)

    # 4. Build Output
    test_ids = splits.get("test", [])
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
            
        final_output[f"{mid}_test"] = entry

    # 5. Save
    with open("data.json", "w") as f:
        json.dump(final_output, f, indent=4)
        
    print("Done. Generated data.json.")

if __name__ == "__main__":
    main()
