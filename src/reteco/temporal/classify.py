"""Query temporal-type classifier, trained on guidance_train.jsonl labels only.

Target: is the query anchored to calendar time? Positive when the guidance
says `is_temporal_query` and `expected_granularity` is calendar-like (year,
decade, century, date, month, era, period, ...); negative for no time,
durations, block heights, epochs/slots, versions, hours/minutes, etc. (Note
`is_temporal_query` alone is useless: it is true for 96% of train queries.)

Features: TF-IDF word 1-2 grams of the HTML-stripped query plus counts of
extracted time-expression kinds. Model: logistic regression (seed 13).

Train-split gate values are 5-fold out-of-fold predictions, so tuning the
gate on train does not see a query's own label. Dev-split gate values come
from a model fit on all of train. guidance_dev.jsonl is never read.
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np

from .. import SEED
from ..data import DOMAINS, data_dir, load_queries
from .doctimes import CACHE_DIR
from .extract import clean_text, extract

KINDS = ["date", "year", "range", "decade", "century", "millennium", "era", "relative", "present"]
_POS = ("year", "decade", "centur", "date", "month", "era", "millenni", "period", "day", "week",
        "quarter", "event")
_NEG = ("block", "epoch", "slot", "hour", "minute", "second", "duration", "version", "conceptual",
        "process", "transaction", "functionality", "location", "lineage", "document", "textual",
        "individual", "technical", "real-time", "city")


def label_from_guidance(g: dict) -> int:
    if not g.get("is_temporal_query"):
        return 0
    gran = (g.get("expected_granularity") or "").lower()
    return int(any(p in gran for p in _POS) and not any(n in gran for n in _NEG))


def load_train_labels(root=None) -> dict[str, int]:
    """qid -> label, from guidance_train.jsonl (TRAIN ONLY)."""
    out = {}
    for d in DOMAINS:
        with open(data_dir(root) / d / "guidance_train.jsonl", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    out[str(r["id"])] = label_from_guidance(r["query_guidance"])
    return out


def _texts_and_feats(qtexts: list[str], domains: list[str]):
    texts = [clean_text(t) for t in qtexts]
    feats = []
    for t, dom in zip(texts, domains):
        c = Counter(e.kind for e in extract(t, present=True))
        feats.append([np.log1p(c[k]) for k in KINDS] + [np.log1p(len(t.split()))]
                     + [float(dom == d) for d in DOMAINS])
    return texts, np.asarray(feats)


def _fit_predict(train_texts, train_feats, y, test_texts, test_feats, C):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from scipy.sparse import csr_matrix, hstack

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=50000)
    sc = StandardScaler()
    Xtr = hstack([vec.fit_transform(train_texts), csr_matrix(sc.fit_transform(train_feats))]).tocsr()
    Xte = hstack([vec.transform(test_texts), csr_matrix(sc.transform(test_feats))]).tocsr()
    lr = LogisticRegression(C=C, max_iter=3000, random_state=SEED, class_weight="balanced")
    lr.fit(Xtr, y)
    return lr.predict_proba(Xte)[:, 1]


def train_queries(root=None):
    qids, texts, doms = [], [], []
    for d in DOMAINS:
        for q in load_queries(d, "train", root):
            qids.append(q.id)
            texts.append(q.query)
            doms.append(d)
    return qids, texts, doms


def oof_predictions(C: float = 1.0, folds: int = 5, root=None) -> tuple[dict[str, float], dict]:
    """5-fold out-of-fold probabilities on train + CV metrics."""
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    labels = load_train_labels(root)
    qids, raw, doms = train_queries(root)
    texts, feats = _texts_and_feats(raw, doms)
    y = np.asarray([labels[q] for q in qids])
    probs = np.zeros(len(qids))
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    for tr, te in skf.split(texts, y):
        probs[te] = _fit_predict([texts[i] for i in tr], feats[tr], y[tr],
                                 [texts[i] for i in te], feats[te], C)
    # baseline: "query has any extracted non-present time expression"
    rule = np.asarray([float(any(k > 0 for k in f[:8])) for f in feats])
    metrics = {"n": len(y), "positive_rate": float(y.mean()),
               "auc": float(roc_auc_score(y, probs)), "acc@0.5": float(accuracy_score(y, probs >= 0.5)),
               "rule_acc": float(accuracy_score(y, rule)), "C": C}
    return dict(zip(qids, probs.tolist())), metrics


def dev_predictions(C: float = 1.0, root=None) -> dict[str, float]:
    labels = load_train_labels(root)
    qids, raw, doms = train_queries(root)
    texts, feats = _texts_and_feats(raw, doms)
    y = np.asarray([labels[q] for q in qids])
    dq, draw, ddoms = [], [], []
    for d in DOMAINS:
        for q in load_queries(d, "dev", root):
            dq.append(q.id)
            draw.append(q.query)
            ddoms.append(d)
    dtexts, dfeats = _texts_and_feats(draw, ddoms)
    return dict(zip(dq, _fit_predict(texts, feats, y, dtexts, dfeats, C).tolist()))


def gate_probs(split: str, C: float = 1.0) -> dict[str, float]:
    path = CACHE_DIR / "gate" / f"{split}_C{C}.json"
    if path.exists():
        return json.loads(path.read_text())
    path.parent.mkdir(parents=True, exist_ok=True)
    if split == "train":
        probs, metrics = oof_predictions(C)
        (CACHE_DIR / "gate" / f"cv_metrics_C{C}.json").write_text(json.dumps(metrics, indent=2))
    else:
        probs = dev_predictions(C)
    path.write_text(json.dumps(probs))
    return probs


def predict_gate(table, split: str, gcfg: dict) -> np.ndarray:
    probs = gate_probs(split, gcfg.get("C", 1.0))
    return np.asarray([probs[q] for q in table.qids])


if __name__ == "__main__":
    for C in (0.3, 1.0, 3.0, 10.0):
        _, m = oof_predictions(C)
        print(json.dumps(m))
