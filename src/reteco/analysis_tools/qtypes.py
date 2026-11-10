"""Query temporal types: guidance-derived train labels, rules, and a classifier.

Seven types describe what a query asks about time:

    specific_date      when did X happen / what year / did X happen at time T
    era_period         what was X like during a period (medieval, WWII, 1920s)
    ordering_sequence  before/after, which came first, timelines, sequences
    duration           how long, for how long, how many days/years
    change_over_time   origins, evolution, trends, how X changed
    present_vs_past    is X still true / current status vs. the past
    no_time            no temporal intent

Train labels come from `guidance_train.jsonl` (`temporal_intent`, mapped below).
guidance_dev.jsonl is never read: dev queries get types from the classifier,
trained on train labels, using only the dev query text.

A second, orthogonal facet comes from the temporal extractor alone: whether the
query states an explicit time anchor (any extracted expression other than
"present").
"""
from __future__ import annotations

import json
import re
from collections import Counter

import numpy as np

from .. import SEED
from ..data import DOMAINS, data_dir, load_queries
from ..temporal.extract import clean_text, extract

TYPES = ["specific_date", "era_period", "ordering_sequence", "duration",
         "change_over_time", "present_vs_past", "no_time"]

TYPE_LABELS = {
    "specific_date": "Specific date",
    "era_period": "Era / period",
    "ordering_sequence": "Ordering / sequence",
    "duration": "Duration",
    "change_over_time": "Change over time",
    "present_vs_past": "Present vs. past",
    "no_time": "No temporal intent",
}

# guidance `temporal_intent` -> type. The 18 raw intents include a long tail of
# rare values; each goes to the closest of the seven types.
INTENT_MAP = {
    "when": "specific_date",
    "event_verification_and_authenticity": "specific_date",
    "event_analysis_and_localization": "specific_date",
    "period_definition": "era_period",
    "during": "era_period",
    "historical_attribution_and_context": "era_period",
    "timeline": "ordering_sequence",
    "order": "ordering_sequence",
    "before_after": "ordering_sequence",
    "after": "ordering_sequence",
    "since": "ordering_sequence",
    "duration": "duration",
    "origins_evolution_comparative_analysis": "change_over_time",
    "trends_changes_and_cross_period": "change_over_time",
    "ongoing_status": "present_vs_past",
    "none": "no_time",
    "why": "no_time",
    "causation_analysis": "no_time",
}

EXTRACT_KINDS = ["date", "year", "range", "decade", "century", "millennium", "era", "relative",
                 "present"]


def type_from_guidance(g: dict) -> str:
    if not g.get("is_temporal_query", True):
        return "no_time"
    return INTENT_MAP.get(g.get("temporal_intent") or "none", "no_time")


def load_train_labels(root=None) -> dict[str, dict]:
    """qid -> {type, intent, granularity}, from guidance_train.jsonl only."""
    out = {}
    for d in DOMAINS:
        path = data_dir(root) / d / "guidance_train.jsonl"   # TRAIN ONLY, never dev
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    g = r["query_guidance"]
                    out[str(r["id"])] = {"type": type_from_guidance(g),
                                         "intent": g.get("temporal_intent") or "",
                                         "granularity": g.get("expected_granularity") or ""}
    return out


# ------------------------------------------------------------------ text --
_WS = re.compile(r"\s+")


def split_query(raw: str) -> tuple[str, str]:
    """(title, full cleaned text). Queries are a title line followed by an HTML body."""
    title = clean_text(raw.split("\n", 1)[0])
    full = clean_text(raw)
    return _WS.sub(" ", title).strip(), _WS.sub(" ", full).strip()


# Cue patterns. Each is a rule for `rule_type` and a binary classifier feature,
# matched separately on the title and on the full text.
CUES = {
    "duration": re.compile(
        r"\bhow long\b|\bfor how long\b|\bduration\b|\bhow many (?:years|months|weeks|days|hours|"
        r"minutes|seconds|blocks|confirmations)\b|\btake[sn]? (?:to|for)\b|\bexpir\w*|\blifetime\b|"
        r"\btime ?out\b|\bdeadline\b|\bwait\w*\b|\blast(?:ed|s)? (?:for|until)\b", re.I),
    "present": re.compile(
        r"\bstill\b|\bnowadays\b|\bcurrently\b|\bcurrent\b|\btoday\b|\bthese days\b|\bany ?more\b|"
        r"\bmodern\b|\bat present\b|\bas of now\b|\bsurviv\w*|\bremain\w*|\bstill exist\w*|"
        r"\bno longer\b|\bused to\b", re.I),
    "when": re.compile(
        r"^\s*when\b|\bwhen (?:did|was|were|is|does|do|will|has|had)\b|\bwhat (?:year|date|day|"
        r"month|time|century|decade)\b|\bwhich (?:year|date|century|decade)\b|\bdat(?:e|ed|ing)\b",
        re.I),
    "verify": re.compile(r"^\s*(?:did|was|were|is|has|had)\b|\breally\b|\bactually\b|\btrue\b|"
                         r"\bauthentic\w*|\bmyth\b", re.I),
    "order": re.compile(
        r"\bbefore\b|\bafter\b|\bfirst\b|\bearliest\b|\blatest\b|\bprior to\b|\bprecede\w*|"
        r"\bsequence\b|\btimeline\b|\bchronolog\w*|\bin what order\b|\bsince\b|\buntil\b|"
        r"\bpre-\w+|\bpost-\w+|\bthen\b|\bfollow\w*\b|\blater\b|\bearlier\b", re.I),
    "change": re.compile(
        r"\bevolv\w*|\bevolution\b|\bchang(?:e|ed|es|ing)\b|\bdevelop\w*|\bhistory of\b|"
        r"\borigin\w*|\btrend\w*|\bover time\b|\bdeclin\w*|\brise of\b|\bgrow\w*|\bbec[oa]me\b|"
        r"\bshift\w*|\bhow did .* (?:start|begin|come)\b|\bwhy did\b", re.I),
    "period": re.compile(
        r"\bduring\b|\b(?:era|period|age|epoch|dynasty|reign|century|centuries|decade)\b|"
        r"\bmedieval\b|\bancient\b|\bclassical\b|\bwar\b|\bin the \d{4}s\b|\bin \d{3,4}\b", re.I),
}
CUE_NAMES = list(CUES)


def query_facets(raw: str) -> dict:
    """Extractor facets of one query (no labels involved)."""
    title, full = split_query(raw)
    exprs = extract(full, present=True)
    kinds = Counter(e.kind for e in exprs)
    anchored = any(e.kind != "present" for e in exprs)
    return {"title": title, "text": full, "kinds": kinds, "anchored": anchored,
            "present_expr": kinds["present"] > 0,
            "cues_title": {c: bool(p.search(title)) for c, p in CUES.items()},
            "cues_text": {c: bool(p.search(full)) for c, p in CUES.items()},
            "n_words": len(full.split())}


def rule_type(f: dict) -> str:
    """Hand-written rules on the title cues, the extractor, then the body cues."""
    t, b = f["cues_title"], f["cues_text"]
    if t["duration"]:
        return "duration"
    if t["present"] or f["present_expr"]:
        return "present_vs_past"
    if t["when"]:
        return "specific_date"
    if t["change"]:
        return "change_over_time"
    if t["order"]:
        return "ordering_sequence"
    if f["anchored"] or t["period"]:
        return "era_period"
    for cue, typ in (("duration", "duration"), ("when", "specific_date"),
                     ("change", "change_over_time"), ("order", "ordering_sequence")):
        if b[cue]:
            return typ
    return "no_time"


# ------------------------------------------------------------ classifier --
def _dense_feats(facets: list[dict], domains: list[str]) -> np.ndarray:
    rows = []
    for f, d in zip(facets, domains):
        row = [float(f["cues_title"][c]) for c in CUE_NAMES]
        row += [float(f["cues_text"][c]) for c in CUE_NAMES]
        row += [np.log1p(f["kinds"][k]) for k in EXTRACT_KINDS]
        row += [float(f["anchored"]), np.log1p(f["n_words"])]
        row += [float(d == dom) for dom in DOMAINS]
        rows.append(row)
    return np.asarray(rows)


class TypeClassifier:
    """TF-IDF (title and full text) + cue/extractor/domain features, logistic regression."""

    def __init__(self, C: float = 1.0):
        self.C = C

    def fit(self, facets, domains, y):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        self.v_title = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                       lowercase=True)
        self.v_text = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.5,
                                      sublinear_tf=True, max_features=50000)
        X = self._X(facets, domains, fit=True)
        self.model = LogisticRegression(C=self.C, class_weight="balanced", max_iter=5000,
                                        random_state=SEED)
        self.model.fit(X, y)
        return self

    def _X(self, facets, domains, fit=False):
        import scipy.sparse as sp
        titles = [f["title"] for f in facets]
        texts = [f["text"] for f in facets]
        if fit:
            A, B = self.v_title.fit_transform(titles), self.v_text.fit_transform(texts)
        else:
            A, B = self.v_title.transform(titles), self.v_text.transform(texts)
        return sp.hstack([A, B, sp.csr_matrix(_dense_feats(facets, domains))]).tocsr()

    def predict(self, facets, domains):
        return list(self.model.predict(self._X(facets, domains)))

    def predict_proba(self, facets, domains):
        return self.model.predict_proba(self._X(facets, domains)), list(self.model.classes_)


def load_split(split: str, root=None) -> list[dict]:
    """Queries (id + query text only) with facets, in a fixed order."""
    out = []
    for d in DOMAINS:
        for q in load_queries(d, split, root):
            out.append({"qid": q.id, "domain": d, **query_facets(q.query)})
    return out


def cross_validate(items: list[dict], y: list[str], Cs=(0.3, 1.0, 3.0, 10.0), folds: int = 5) -> dict:
    """Stratified CV on train for the classifier (each C), the rules and the majority class."""
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import StratifiedKFold
    y = np.asarray(y)
    doms = [it["domain"] for it in items]
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(y)), y))
    res = {}
    for C in Cs:
        pred = np.empty(len(y), dtype=object)
        for tr, te in splits:
            clf = TypeClassifier(C).fit([items[i] for i in tr], [doms[i] for i in tr], y[tr])
            pred[te] = clf.predict([items[i] for i in te], [doms[i] for i in te])
        res[f"classifier_C{C:g}"] = pred
    res["rules"] = np.asarray([rule_type(it) for it in items], dtype=object)
    major = Counter(y).most_common(1)[0][0]
    res["majority"] = np.asarray([major] * len(y), dtype=object)
    summary = {k: {"accuracy": accuracy_score(y, p),
                   "macro_f1": f1_score(y, p, labels=TYPES, average="macro", zero_division=0),
                   "pred": p}
               for k, p in res.items()}
    return summary
