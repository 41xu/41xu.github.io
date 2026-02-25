"""
Generates data.json for HTML visualization.

Folder layout expected:
  predict/{model}.json              -> list of {id, generated, ...}
  metric/{model}-metrics-new.json   -> {average_metrics, per_sample_metrics}
  retrieval/{model}-*.json          -> list of {id, gt, predicted, top3, top5}

Base files (root dir):
  amt_motionfix_latest.json   -> {id: {motion_a, motion_b, motion_overlaid, annotation}}
  splits.json                 -> {test: [id, ...]}
  mofix_cap.json              -> {"{id}_s": caption_src, "{id}_t": caption_tgt}

Output data.json structure:
  {
    "{id}_test": {
      motions, motiont, overlap, captions, captiont,
      text1  (ground truth),
      texts: { "{model}": generated_text, ... },
      metrics: { "{model}": { BERT, CIDEr, BLEU@1, BLEURT, ROBERTA } },
      retrieval: { "{model}": [ {rank, text, score}, ... ] }
    },
    ...
    "DATASET_METRICS": {
      "_models": [model, ...],
      "{model}": { BERT, CIDEr, BLEU@1, BLEURT, ROBERTA, R@3, R@5, MedR }
    }
  }

Note: R@3/R@5 computed by exact GT string match in top-5 retrieved texts.
      MedR uses rank=1000 for samples where GT is not found in top-5.
"""

import json
import os
import glob
import argparse

PREDICT_DIR = "predict"
METRIC_DIR = "metric"
RETRIEVAL_DIR = "retrieval"
METRIC_SUFFIX = "-metrics-new.json"

# Map raw metric keys to display names used in data.json
METRIC_KEYS_MAP = {
    "BERTScore_F1": "BERT",
    "BLEURT": "BLEURT",
    "CIDEr": "CIDEr",
    "Bleu_1": "BLEU@1",
    "CosSim": "ROBERTA",
}


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def filter_metrics(m_dict):
    return {v: m_dict[k] for k, v in METRIC_KEYS_MAP.items() if k in m_dict}


def discover_models():
    """Return sorted list of model names found in predict/."""
    paths = sorted(glob.glob(os.path.join(PREDICT_DIR, "*.json")))
    return [os.path.splitext(os.path.basename(p))[0] for p in paths]


def load_predict(model):
    """Load predict/{model}.json -> dict mapping id -> generated text."""
    data = load_json(os.path.join(PREDICT_DIR, f"{model}.json"))
    if not isinstance(data, list):
        print(f"Warning: predict/{model}.json is not a list, skipping.")
        return {}
    return {item["id"]: item.get("generated", "") for item in data if "id" in item}


def load_metrics(model):
    """Load metric/{model}-metrics-new.json -> (average_dict, per_sample_dict)."""
    path = os.path.join(METRIC_DIR, f"{model}{METRIC_SUFFIX}")
    data = load_json(path)
    if data is None:
        print(f"Warning: metric file not found: {path}")
        return {}, {}
    return data.get("average_metrics", {}), data.get("per_sample_metrics", {})


def load_retrieval(model):
    """Load retrieval/{model}-*.json -> dict mapping id -> full retrieval item."""
    matches = sorted(glob.glob(os.path.join(RETRIEVAL_DIR, f"{model}-*.json")))
    if not matches:
        print(f"Warning: no retrieval file found for model {model}")
        return {}
    data = load_json(matches[0])
    if not isinstance(data, list):
        return {}
    return {item["id"]: item for item in data if "id" in item}


def compute_retrieval_metrics(retrieval_by_id):
    """Compute R@3, R@5, MedR from retrieval data (exact GT string match)."""
    ranks = []
    for item in retrieval_by_id.values():
        gt = item.get("gt", "").strip().lower()
        top5 = item.get("top5", [])
        rank = None
        for entry in top5:
            if entry.get("text", "").strip().lower() == gt:
                rank = entry["rank"]
                break
        ranks.append(rank if rank is not None else 1000)

    if not ranks:
        return {}

    total = len(ranks)
    r3 = sum(1 for r in ranks if r <= 3) / total
    r5 = sum(1 for r in ranks if r <= 5) / total
    sorted_r = sorted(ranks)
    if total % 2 == 1:
        medr = sorted_r[total // 2]
    else:
        medr = (sorted_r[total // 2 - 1] + sorted_r[total // 2]) / 2

    return {"R@3": r3, "R@5": r5, "MedR": medr}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filter_ids",
        type=str,
        help="Path to a JSON file (list or dict) of IDs to include",
    )
    args = parser.parse_args()

    # Base data
    datas = load_json("amt_motionfix_latest.json") or {}
    splits = load_json("splits.json") or {}
    captions = load_json("mofix_cap.json") or {}

    # Auto-discover models from predict/
    models = discover_models()
    if not models:
        print("No model files found in predict/. Exiting.")
        return
    print(f"Discovered models: {models}")

    # Load all data
    predict_data = {m: load_predict(m) for m in models}
    metric_avg, metric_per_sample = {}, {}
    for m in models:
        metric_avg[m], metric_per_sample[m] = load_metrics(m)
    retrieval_data = {m: load_retrieval(m) for m in models}

    # Determine test IDs
    test_ids = splits.get("test", [])

    if args.filter_ids:
        filter_file = load_json(args.filter_ids)
        if filter_file is None:
            print(f"Error: filter file {args.filter_ids} not found.")
        else:
            if isinstance(filter_file, list):
                target = {str(x) for x in filter_file}
            elif isinstance(filter_file, dict):
                target = {str(x) for x in filter_file.keys()}
            else:
                target = set()
                print("Error: filter file must be a list or dict.")
            test_ids = [mid for mid in test_ids if mid in target]
            print(f"Filtering: keeping {len(test_ids)} IDs.")

    # Build per-sample output
    final_output = {}

    for mid in test_ids:
        if mid not in datas:
            continue

        entry = {
            "motions":   datas[mid].get("motion_a", ""),
            "motiont":   datas[mid].get("motion_b", ""),
            "overlap":   datas[mid].get("motion_overlaid", ""),
            "captions":  captions.get(f"{mid}_s", ""),
            "captiont":  captions.get(f"{mid}_t", ""),
            "text1":     datas[mid].get("annotation", ""),
            "texts":     {m: predict_data[m].get(mid, "") for m in models},
            "metrics": {
                m: filter_metrics(metric_per_sample[m].get(mid, {}))
                for m in models
            },
            "retrieval": {
                m: retrieval_data[m].get(mid, {}).get("top5", [])
                for m in models
            },
        }

        final_output[f"{mid}_test"] = entry

    # Dataset-level metrics
    dataset_metrics = {"_models": models}
    for m in models:
        dm = filter_metrics(metric_avg[m])
        dm.update(compute_retrieval_metrics(retrieval_data[m]))
        dataset_metrics[m] = dm

    final_output["DATASET_METRICS"] = dataset_metrics

    with open("data.json", "w") as f:
        json.dump(final_output, f, indent=4)

    print(f"Done. Generated data.json: {len(final_output) - 1} samples, {len(models)} models.")


if __name__ == "__main__":
    main()
