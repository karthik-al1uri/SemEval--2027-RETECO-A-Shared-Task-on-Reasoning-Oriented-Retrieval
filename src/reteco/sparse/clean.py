"""Corpus and query cleaning.

Corpus variants (index is rebuilt per variant, see `reteco.sparse.index`):

    raw     documents as given
    markup  strip markup and known boilerplate patterns from each document:
            HTML tags and entities, BBCode ([url=...], [b]), Markdown emphasis,
            image and link syntax, URLs and e-mail addresses, software license
            headers, copyright lines, disclaimers, cookie / newsletter / nav
            text, "Retrieved <date>" / "Archived from the original" citation
            metadata
    clean   markup + drop "repeated footer" sentences: any sentence of >= 5
            words that occurs (after `markup` cleaning) in >= 20 distinct
            documents of the same domain. The sentence list is computed per
            domain from the documents alone (no labels) and cached under
            cache/sparse/boilerplate/.

Query cleaning (StackExchange posts: title + HTML body). Steps, applied in order:

    code       drop <pre> code blocks (inline <code> text is kept)
    html       drop tags (incl. attributes such as href / src) and unescape entities
    urls       drop URLs left in the visible text
    signature  drop sign-offs and pleasantries ("Thanks in advance", "Any help
               appreciated", "Regards, ...", "EDIT:" markers)
"""
from __future__ import annotations

import html
import json
import re
from collections import Counter
from multiprocessing import Pool

CORPUS_VARIANTS = ("raw", "markup", "clean")
QUERY_STEPS = ("code", "html", "urls", "signature")
REPEAT_MIN_DOCS = 20
REPEAT_MIN_WORDS = 5

# ---------------------------------------------------------------- markup --
_URL = re.compile(r"(?:https?|ftp)://[^\s<>\]\[\"')]+|www\.[^\s<>\]\[\"')]+\.[a-z]{2,}[^\s<>\]\[\"')]*", re.I)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_HTML_TAG = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]*(?:\s[^<>]{0,500})?/?>")
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
_ORPHAN_ENTITY = re.compile(r"(?<![&\w])(amp|nbsp|quot|lt|gt|#\d{2,5});")
_BBCODE = re.compile(r"\[/?(?:url|b|i|u|s|img|quote|color|size|font|center|code|link|email)(?:=[^\]]*)?\]", re.I)
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_MD_EMPH = re.compile(r"\*\*|__|`+")

# Sentences dropped whole when they match one of these.
_BOILER_SENTENCE = re.compile("|".join([
    # software licenses
    r"permission is hereby granted",
    r"provided \W*as is\W*,? without warranty",
    r"in no event shall the\b.{0,40}\b(authors|copyright holders)",
    r"use of this source code is governed by",
    r"licensed under the (apache|mit|bsd|gnu)",
    r"spdx-license-identifier",
    r"redistribution and use in source and binary forms",
    r"this (program|library|file) is free software",
    # open-access / copyright footers
    r"distributed under the terms (and conditions )?of the creative commons",
    r"all rights reserved",
    r"(©|\(c\)|copyright)\s*(\d{4}|by\b)",
    # disclaimers
    r"(does not|doesn.t|is not|not meant to|not intended to) (constitute|serve as|be construed as)?\s*"
    r"(financial|investment|legal|tax) advice",
    r"for (general )?informational purposes only",
    r"consult (your|a) (own )?(financial|legal|tax|investment) (advisor|adviser|professional)",
    r"^\W*disclaimer\b",
    r"solely at your own risk",
    # web chrome
    r"\b(we|this (site|website)) uses? cookies",
    r"\bprivacy policy\b",
    r"\bterms (of (use|service)|(and|&) conditions)\b",
    r"subscribe to (our|the) newsletter|sign up for (our|the) newsletter|join (our|the) mailing list",
    r"follow us on (twitter|facebook|linkedin|instagram|x\b)",
    r"press enter or click to view image",
    r"^\W*(return to main menu|skip to (main )?content|share this( article| post)?|read more)\W*$",
    r"full text not available from this repository",
    r"a high-res(olution)? (photo|version|image) (can be found|is available) here",
]), re.I)

# Spans removed in place (citation metadata, nav fragments).
_BOILER_SPAN = re.compile("|".join([
    r"archived from the original on [^.;,)]{0,40}",
    r"retrieved (on )?\d{1,2} [A-Z][a-z]+ \d{4}",
    r"retrieved (on )?[A-Z][a-z]+ \d{1,2},? \d{4}",
    r"\bclick here\b",
    r"©[^.]{0,80}?\b(1[89]|20)\d\d\b",
    r"\[citation needed\]",
]), re.I)

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"“(\[]?[A-Z0-9])|\n+|\s+[-|•·]\s+")
_WS = re.compile(r"\s+")


def strip_markup(text: str) -> str:
    text = _HTML_COMMENT.sub(" ", text)
    text = _MD_IMAGE.sub(" ", text)
    text = _MD_LINK.sub(r" \1 ", text)
    text = _BBCODE.sub(" ", text)
    text = _HTML_TAG.sub(" ", text)
    text = html.unescape(text)
    text = _ORPHAN_ENTITY.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _EMAIL.sub(" ", text)
    text = _MD_EMPH.sub(" ", text)
    return text


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s and s.strip()]


def norm_sentence(s: str) -> str:
    return _WS.sub(" ", s.lower()).strip()


def clean_markup(text: str) -> str:
    """`markup` corpus variant for one document."""
    text = strip_markup(text)
    text = _BOILER_SPAN.sub(" ", text)
    kept = [s for s in split_sentences(text) if not _BOILER_SENTENCE.search(s)]
    return _WS.sub(" ", " ".join(kept)).strip()


def clean_document(text: str, repeated: frozenset[str] | set[str] = frozenset()) -> str:
    """`markup` cleaning, then drop sentences listed in `repeated` (the `clean` variant)."""
    text = clean_markup(text)
    if not repeated:
        return text
    kept = [s for s in split_sentences(text) if norm_sentence(s) not in repeated]
    return " ".join(kept)


# ---------------------------------------------------- repeated sentences --
def _doc_sentences(texts: list[str]) -> Counter:
    c = Counter()
    for t in texts:
        c.update({norm_sentence(s) for s in split_sentences(clean_markup(t))
                  if len(s.split()) >= REPEAT_MIN_WORDS})
    return c


def repeated_sentences(domain: str, workers: int = 8, min_docs: int = REPEAT_MIN_DOCS) -> set[str]:
    """Sentences (normalised) that occur in >= min_docs documents of a domain. Cached."""
    from ..data import iter_corpus
    from .index import CACHE_DIR, _chunks

    path = CACHE_DIR / "boilerplate" / f"{domain}_min{min_docs}.json"
    if path.exists():
        return set(json.loads(path.read_text()))
    total = Counter()
    with Pool(workers) as pool:
        for c in pool.imap_unordered(_doc_sentences, (txt for _, txt in _chunks(iter_corpus(domain), 1000))):
            total.update(c)
    rep = sorted(s for s, n in total.items() if n >= min_docs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rep, indent=0, ensure_ascii=False))
    return set(rep)


# ---------------------------------------------------------------- queries --
_PRE = re.compile(r"<pre\b[^>]*>.*?</pre>", re.S | re.I)
_SIGNATURE = re.compile("|".join([
    r"\b(many |any )?thanks?( you)?( (so|very) much)?( in advance| for (your|any) (help|time|answers?|input|reading))?\s*[!.:)]*\s*$",
    r"\bthanks? (you )?in advance\b[^.!?]*[.!?]*",
    r"\b(any|all) (help|advice|ideas|suggestions|pointers|input|insights?)( on this)?( (would be|is|are|will be) (greatly |much |very much |really )?appreciated)\s*[!.]*",
    r"\b(i would|i'd) (really |greatly )?appreciate (any|your) (help|advice|input|thoughts)\b[^.!?]*[.!?]*",
    r"\b(best |kind |warm )?regards\b,?[^.!?]{0,40}$",
    r"\bcheers\b[!.,]*\s*\w{0,20}\s*$",
    r"\b(edit|update)\s*\d*\s*:",
]), re.I | re.M)


def clean_query(text: str, steps=QUERY_STEPS) -> str:
    steps = set(steps)
    unknown = steps - set(QUERY_STEPS)
    if unknown:
        raise ValueError(f"unknown query cleaning steps {sorted(unknown)}")
    if "code" in steps:
        text = _PRE.sub(" ", text)
    if "html" in steps:
        text = _HTML_COMMENT.sub(" ", text)
        text = _HTML_TAG.sub(" ", text)
        text = html.unescape(text)
    if "urls" in steps:
        text = _URL.sub(" ", text)
    if "signature" in steps:
        text = _SIGNATURE.sub(" ", text)
    return _WS.sub(" ", text).strip()
