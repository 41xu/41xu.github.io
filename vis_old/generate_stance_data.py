import json
import os

def load_json(filename):
    """Safely load a JSON file."""
    if not os.path.exists(filename):
        print(f"Warning: File not found: {filename}")
        return {}
    with open(filename, "r") as f:
        return json.load(f)

def load_generated_map(filename):
    """Loads a result file and returns a dict mapping id -> generated text."""
    data = load_json(filename)
    if isinstance(data, list):
         # Handle list of objects: [{"id": "...", "generated": "..."}]
         return {str(item["id"]): item.get("generated", "") for item in data}
    elif isinstance(data, dict):
         # Handle direct dict: {"id": "text"}
         return {str(k): v for k, v in data.items()}
    return {}

def main():
    # --- Configuration ---
    
    # Base directory for stance files
    STANCE_DIR = "stance"
    
    # Path to the list of test IDs
    TEST_IDS_FILE = os.path.join(STANCE_DIR, "adjust_test.json")
    
    # Map output fields (text2, text3, etc.) to source JSON filenames
    # Keys should match what index.html expects in "extraData"
    SOURCES = {
        "text2": "stance_base.json",           # Baseline
        "text3": "stance_cont_bp_stneg_poe.json", # Ours / Cont+BP+...
        # Add more mappings here as needed, e.g.:
        # "text7": "stance_final_ours.json"
    }
    
    # Output file
    OUTPUT_FILE = os.path.join(STANCE_DIR, "data.json")

    # ---------------------

    print(f"Loading test IDs from {TEST_IDS_FILE}...")
    test_ids = load_json(TEST_IDS_FILE)
    
    # Ensure test_ids is a list of strings
    if not isinstance(test_ids, list):
        print("Error: Test IDs file must contain a list.")
        return
    test_ids = [str(tid) for tid in test_ids]
    print(f"Found {len(test_ids)} test IDs.")

    # Load all source data
    loaded_sources = {}
    for field, filename in SOURCES.items():
        filepath = os.path.join(STANCE_DIR, filename)
        print(f"Loading {field} from {filepath}...")
        loaded_sources[field] = load_generated_map(filepath)

    # Build the final data structure
    final_data = {}
    count = 0
    
    for tid in test_ids:
        entry = {}
        has_data = False
        
        for field, source_data in loaded_sources.items():
            if tid in source_data:
                entry[field] = source_data[tid]
                has_data = True
            else:
                # Optional: Warn if missing?
                # print(f"Warning: ID {tid} missing in {field}")
                entry[field] = ""
        
        if has_data:
            final_data[tid] = entry
            count += 1
            
    print(f"Generated data for {count} IDs.")
    
    # Write to file
    print(f"Writing to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(final_data, f, indent=4)
    print("Done.")

if __name__ == "__main__":
    main()
