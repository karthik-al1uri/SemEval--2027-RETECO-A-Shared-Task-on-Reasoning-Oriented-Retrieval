"""Assign temporal types to train and dev queries.

    PYTHONPATH=src RETECO_DATA=data/data/track1_tempo python scripts/analysis_query_types.py

Train: type = guidance_train label (mapped intent); the classifier's 5-fold
out-of-fold prediction is stored next to it. Dev: type = prediction of the
method with the best train-CV macro-F1 (classifier or rules), fit on all of
train. guidance_dev.jsonl is never read.

Writes
    results/analysis_query_types.csv   one row per train/dev query
    results/analysis_qtype_cv.csv      train CV of every method (+ per-type F1)
"""
from __future__ import annotations

import csv
from collections import Counter

from sklearn.metrics import classification_report

from reteco.analysis_tools.qtypes import (TYPES, TypeClassifier, cross_validate, load_split,
                                          load_train_labels, rule_type)
from reteco.data import REPO_ROOT

OUT = REPO_ROOT / "results"


def main() -> None:
    train = load_split("train")
    labels = load_train_labels()
    y = [labels[it["qid"]]["type"] for it in train]
    print("train type counts:", Counter(y).most_common())

    cv = cross_validate(train, y)
    for k, v in cv.items():
        print(f"{k:<18} acc {v['accuracy']:.3f}  macro-F1 {v['macro_f1']:.3f}")
    best = max(cv, key=lambda k: (cv[k]["macro_f1"], cv[k]["accuracy"]))
    print("selected:", best)
    print(classification_report(y, cv[best]["pred"], labels=TYPES, zero_division=0, digits=3))
    rep = classification_report(y, cv[best]["pred"], labels=TYPES, zero_division=0,
                                output_dict=True)

    with open(OUT / "analysis_qtype_cv.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "selected", "accuracy", "macro_f1"] + [f"f1_{t}" for t in TYPES])
        for k, v in cv.items():
            per = classification_report(y, v["pred"], labels=TYPES, zero_division=0,
                                        output_dict=True)
            w.writerow([k, int(k == best), f"{v['accuracy']:.4f}", f"{v['macro_f1']:.4f}"]
                       + [f"{per[t]['f1-score']:.4f}" for t in TYPES])
    del rep

    dev = load_split("dev")
    if best == "rules":
        dev_pred, source = [rule_type(it) for it in dev], "rules"
    else:
        C = float(best.split("_C")[1])
        clf = TypeClassifier(C).fit(train, [it["domain"] for it in train], y)
        dev_pred, source = clf.predict(dev, [it["domain"] for it in dev]), best
    print("dev predicted type counts:", Counter(dev_pred).most_common())

    with open(OUT / "analysis_query_types.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "domain", "qid", "type", "type_source", "cv_pred", "rule_type",
                    "anchored", "present_expr", "n_words", "intent", "title"])
        for it, t, p in zip(train, y, cv[best]["pred"]):
            lab = labels[it["qid"]]
            w.writerow(["train", it["domain"], it["qid"], t, "guidance_train", p, rule_type(it),
                        int(it["anchored"]), int(it["present_expr"]), it["n_words"],
                        lab["intent"], it["title"][:200]])
        for it, t in zip(dev, dev_pred):
            w.writerow(["dev", it["domain"], it["qid"], t, source, "", rule_type(it),
                        int(it["anchored"]), int(it["present_expr"]), it["n_words"], "",
                        it["title"][:200]])
    print("wrote", OUT / "analysis_query_types.csv")


if __name__ == "__main__":
    main()
