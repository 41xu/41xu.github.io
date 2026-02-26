"""
Generates data.json for HTML visualization.

Folder layout expected:
  predict/{model}.json                    -> list of {id, generated, ...}
  metric/{model}-metrics-new.json         -> {average_metrics, per_sample_metrics}
  retrieval/{model}-top*_retrievals.json  -> TMR retrieval (id, gt, predicted, top3, top5)
  retrieval/{model}-llm-top*.json         -> LLM retrieval (same structure)

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
      retrieval: { "{model}": { "tmr": [...top5...], "llm": [...top5...] } }
    },
    ...
    "DATASET_METRICS": {
      "_models": [model, ...],
      "{model}": { BERT, CIDEr, BLEU@1, BLEURT, ROBERTA,
                   TMR_R@3, TMR_R@5, TMR_MedR,
                   LLM_R@3, LLM_R@5, LLM_MedR }
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


def load_gt_ranks(model):
    """Load retrieval/{model}-gt-ranks.json -> dict mapping id -> {gt_rank, gt_score}.
    Returns empty dict if file not found (not all models have this file)."""
    path = os.path.join(RETRIEVAL_DIR, f"{model}-gt-ranks.json")
    data = load_json(path)
    if not isinstance(data, list):
        return {}
    return {
        item["id"]: {"gt_rank": item.get("gt_rank"), "gt_score": item.get("gt_score")}
        for item in data if "id" in item
    }


def load_retrieval(model, kind="tmr"):
    """Load retrieval file for a model.

    kind='tmr' -> {model}-top*_retrievals.json (excludes llm files)
    kind='llm' -> {model}-llm-top*.json

    Returns (metrics_dict, retrieval_by_id) where:
      metrics_dict: pre-computed {R@3, R@5, MedR} from metrics.normal.pred_text2gt_text
      retrieval_by_id: dict mapping id -> item (with top5 list)
    """
    all_matches = sorted(glob.glob(os.path.join(RETRIEVAL_DIR, f"{model}-*.json")))
    if kind == "tmr":
        matches = [p for p in all_matches
                   if "llm" not in os.path.basename(p)
                   and "gt-ranks" not in os.path.basename(p)]
    else:
        matches = [p for p in all_matches if "llm" in os.path.basename(p)]

    if not matches:
        print(f"Warning: no {kind} retrieval file found for model {model}")
        return {}, {}

    data = load_json(matches[0])
    if not isinstance(data, dict):
        print(f"Warning: unexpected format in {matches[0]}")
        return {}, {}

    # Pre-computed metrics (pred_text -> gt_text direction)
    raw_m = data.get("metrics", {}).get("normal", {}).get("pred_text2gt_text", {})
    metrics_dict = {
        "R@3":  raw_m.get("R03", 0) / 100,   # convert from % to fraction
        "R@5":  raw_m.get("R05", 0) / 100,
        "MedR": raw_m.get("MedR", 1000),
    }

    retrievals = data.get("retrievals", [])
    if not isinstance(retrievals, list):
        return metrics_dict, {}

    return metrics_dict, {item["id"]: item for item in retrievals if "id" in item}


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
    retrieval_tmr_metrics, retrieval_tmr = {}, {}
    retrieval_llm_metrics, retrieval_llm = {}, {}
    gt_ranks = {}
    for m in models:
        retrieval_tmr_metrics[m], retrieval_tmr[m] = load_retrieval(m, kind="tmr")
        retrieval_llm_metrics[m], retrieval_llm[m] = load_retrieval(m, kind="llm")
        gt_ranks[m] = load_gt_ranks(m)

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
                m: {
                    "tmr": retrieval_tmr[m].get(mid, {}).get("top5", []),
                    "llm": retrieval_llm[m].get(mid, {}).get("top5", []),
                    **gt_ranks[m].get(mid, {}),  # gt_rank, gt_score if available
                }
                for m in models
            },

        }

        final_output[f"{mid}_test"] = entry

    # Dataset-level metrics
    dataset_metrics = {"_models": models}
    for m in models:
        dm = filter_metrics(metric_avg[m])
        tmr_ret = retrieval_tmr_metrics[m]
        dm["TMR_R@3"]  = tmr_ret.get("R@3",  0)
        dm["TMR_R@5"]  = tmr_ret.get("R@5",  0)
        dm["TMR_MedR"] = tmr_ret.get("MedR", 1000)
        llm_ret = retrieval_llm_metrics[m]
        dm["LLM_R@3"]  = llm_ret.get("R@3",  0)
        dm["LLM_R@5"]  = llm_ret.get("R@5",  0)
        dm["LLM_MedR"] = llm_ret.get("MedR", 1000)
        dataset_metrics[m] = dm

    final_output["DATASET_METRICS"] = dataset_metrics

    with open("data.json", "w") as f:
        json.dump(final_output, f, indent=4)

    print(f"Done. Generated data.json: {len(final_output) - 1} samples, {len(models)} models.")


if __name__ == "__main__":
    main()
