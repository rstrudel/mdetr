import os
from pathlib import Path
import json

root = Path(os.path.expandvars("$WORK/mdetr_models"))
paths = dict(
    effnet_bert_s_detr_s="600224",
    effnet_bert_s="658295",
    vit_s_bert_s="644998",
    vit_s_roberta="645041",
)

keys = ["test_gqa_coco_eval_bbox", "test_flickr_coco_eval_bbox"]

val = dict()
for k in keys:
    val[k] = dict()

for name, path in paths.items():
    log = root / path / "log.txt"
    for k in keys:
        val[k][name] = []
    if not log.exists():
        continue
    with open(log, "r") as f:
        lines = f.readlines()
    for line in lines:
        d = json.loads(line)
        for k in keys:
            val[k][name].append(d[k][1])

# print(f"keys: {d.keys()}")
for k_metric, d in val.items():
    print(f"key: {k_metric}")
    for name, v in d.items():
        v = [round(100 * x, 1) for x in v]
        print(f"{name}: {v}")
    print("")
