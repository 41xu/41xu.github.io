# we should have a script to generate the data.json for us...
"""
final format in data.json:
{id:{"motions": "", "motiont": "", "captions": "", "captiont": "", "text1": gt, "text2": ours}}
"""
import json

with open("amt_motionfix_latest.json", "r") as f:
    datas = json.load(f)

with open("splits.json", "r") as f:
    splits = json.load(f)

with open("mofix_cap.json", "r") as f:
    captions = json.load(f)

with open("our_results.json", "r") as f:
    our_res = json.load(f)


our_res_ = {}
for x in our_res:
    our_res_[x["id"]] = x["generated"]



test_ids = splits["test"]

results = {}

for id in test_ids:
    results[f"{id}_test"] = {
        "motions": datas[id]["motion_a"],
        "motiont": datas[id]["motion_b"],
        "captions": captions[f"{id}_s"],
        "captiont": captions[f"{id}_t"],
        "text1": datas[id]["annotation"],
        "text2": our_res_[id],
    }
with open("data.json", "w") as f:
    json.dump(results, f, indent=4)
