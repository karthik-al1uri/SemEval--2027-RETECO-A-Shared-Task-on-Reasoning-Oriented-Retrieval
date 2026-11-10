"""Rule-based time-expression extraction for queries and documents.

Every expression is mapped to a closed integer year interval [start, end]
(negative = BCE; there is no year 0, so 1 BCE = 0 internally is close enough
for distance computations). Expression kinds:

    date      explicit calendar dates ("5 March 1943", "1943-03-05"),
              normalised with dateparser
    year      standalone year 1000-2029, or any year with an era marker
              ("44 BC", "AD 79", "10,000 BCE")
    range     "1914-1918", "1914–18", "from 1914 to 1918", "between 2000 and 2010"
    decade    "1940s", "the '60s", "the sixties", "early 1950s"
    century   "19th century", "nineteenth-century", "late 1800s", "3rd century BC"
    millennium "2nd millennium BC"
    era       named eras and events from a gazetteer ("WWII", "Victorian era")
    relative  "before 1600", "pre-1600", "since 1945", "after WWII"
    present   "nowadays", "currently", "today" (only with present=True)

Matching runs in priority order over the text; a matched span is masked so
lower-priority patterns (e.g. bare years) do not match inside it again.
Plain 4-digit numbers are filtered with simple context rules (no currency
sign before, no unit such as "BTC" or "blocks" after) because domains such
as bitcoin are full of quantities in the year range.

Speed: pure `re`, ~1.3 ms per bitcoin document and ~3.5 ms per (long)
history document on one core. Named-era patterns only run when one of their
required literals occurs in the text. dateparser is only called on the short
date spans the regexes find (results memoised), never on whole texts:
`dateparser.search.search_dates` costs about as much per document and only
returns point dates (no ranges, centuries, eras or relative expressions).
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass
from functools import lru_cache

EXTRACTOR_VERSION = "v2"

MIN_YEAR = -12000
PRESENT_YEAR = 2026
PLAIN_YEAR_MIN, PLAIN_YEAR_MAX = 1000, 2029


@dataclass(frozen=True)
class TimeExpr:
    start: int
    end: int
    kind: str
    text: str


# ----------------------------------------------------------------- helpers --
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Strip HTML tags and entities (queries are StackExchange HTML)."""
    if "<" in text:
        text = _TAG_RE.sub(" ", text)
    if "&" in text:
        text = html.unescape(text)
    return text


_ORD_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
    "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11,
    "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15,
    "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19,
    "twentieth": 20, "twenty-first": 21, "twenty first": 21,
}
_DECADE_WORDS = {
    "twenties": 20, "thirties": 30, "forties": 40, "fifties": 50, "sixties": 60,
    "seventies": 70, "eighties": 80, "nineties": 90,
}
_MONTHS = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|"
           r"Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?")

# Loose markers (any case, optional dots/spaces) are only used right after
# "century"/"millennium"; bare years need strict uppercase markers so that
# "25 ADA", "AD1" (aggregate demand) or hex fragments like "8C E" do not match.
_BC = r"(?:B\.?\s?C\.?(?:\s?E\.?)?|BCE)"
_AD = r"(?:A\.?\s?D\.?|C\.?\s?E\.?)"
_BC_STRICT = r"(?-i:BCE|BC|B\.C\.E\.|B\.C\.)"
_AD_STRICT = r"(?-i:AD|A\.D\.|CE|C\.E\.)"
_END = r"(?![A-Za-z0-9])"
_NUM_BC = r"(\d{1,3}(?:[ ,]\d{3})+|\d{1,5})"  # "10 000", "10,000", "300"

_UNIT_AFTER = re.compile(
    r"\s*(?:%|percent|BTC|XMR|ADA|IOTA|MIOTA|USD|EUR|GBP|sats?\b|satoshis?\b|blocks?\b|bytes?\b|"
    r"[kmgt]i?b\b|px\b|ms\b|km\b|kms\b|miles?\b|m\b|kg\b|lbs?\b|g\b|dollars?\b|euros?\b|pounds?\b|"
    r"people\b|persons?\b|men\b|women\b|soldiers\b|troops\b|ships\b|words\b|pages?\b|feet\b|ft\b|"
    r"meters?\b|metres?\b|hours?\b|hrs?\b|minutes?\b|mins?\b|seconds?\b|secs?\b|calories\b|rpm\b|"
    r"[mg]hz\b|tx\b|txs\b|transactions?\b|nodes?\b|coins?\b|tokens?\b|points?\b|votes?\b|tons?\b|"
    r"tonnes?\b|acres?\b|km2\b|sq\b|square\b|inputs?\b|outputs?\b|confirmations?\b|peers?\b|"
    r"employees?\b|students?\b|deaths?\b|casualties\b|killed\b|copies\b|units?\b|rows?\b|"
    r"times\b|x\b|bp\b|bps\b|ADA\b|lovelace\b|epochs?\b|slots?\b|ppm\b|mph\b|kph\b|cc\b|ml\b|"
    r"l\b|mm\b|cm\b|inch(?:es)?\b|yards?\b|horses?\b|cattle\b|guns?\b|tanks?\b|aircraft\b|planes?\b)",
    re.I)
_PREFIX_BEFORE = re.compile(
    r"(?:[$£€¥#@]|\bblock\s*|\bheight\s*|\bport\s*|\bversion\s*|\bv\.?\s*|\bno\.\s*|\bnumber\s*|"
    r"\bpage\s*|\bp\.\s*|\bpp\.\s*|\bline\s*|\bitem\s*|\bBIP\s*-?|\bCIP\s*-?|\bissue\s*|\bPR\s*|"
    r"\berror\s*|\bcode\s*|\broute\s*|\bflight\s*|\bbus\s*|\broom\s*|\bArticle\s*|\bsection\s*|"
    r"\b§\s*|\bUSD\s*|\bEUR\s*|\bBTC\s*)$", re.I)


@lru_cache(maxsize=200_000)
def _parse_date_year(span: str) -> int | None:
    """Normalise an explicit date span with dateparser; returns the year."""
    import dateparser

    dt = dateparser.parse(span, languages=["en"],
                          settings={"REQUIRE_PARTS": ["year"], "PREFER_DAY_OF_MONTH": "first"})
    if dt is None:
        return None
    return dt.year


# ----------------------------------------------------------------- gazetteer --
# (pattern, start, end). Patterns are case-insensitive unless prefixed with (?-i:).
# Ranges are conventional textbook dates; precision is a few years at best,
# which is fine because the temporal-fit score uses a distance kernel.
_ERAS: list[tuple[str, int, int]] = [
    # Wars and conflicts
    (r"\bWW\s?II\b|\bWW2\b|\bworld\s+war\s+(?:II|2|two)\b|\bsecond\s+world\s+war\b", 1939, 1945),
    (r"\bWW\s?I\b|\bWW1\b|\bworld\s+war\s+(?:I|1|one)\b|\bfirst\s+world\s+war\b|\bgreat\s+war\b", 1914, 1918),
    (r"\bcold\s+war\b", 1947, 1991),
    (r"\bkorean\s+war\b", 1950, 1953),
    (r"\bvietnam\s+war\b", 1955, 1975),
    (r"\b(?:first\s+)?gulf\s+war\b", 1990, 1991),
    (r"\biraq\s+war\b", 2003, 2011),
    (r"\bwar\s+in\s+afghanistan\b", 2001, 2021),
    (r"\bamerican\s+civil\s+war\b|\b(?-i:U\.?S\.? Civil War)\b|\bwar\s+between\s+the\s+states\b", 1861, 1865),
    (r"\benglish\s+civil\s+war\b", 1642, 1651),
    (r"\bspanish\s+civil\s+war\b", 1936, 1939),
    (r"\brussian\s+civil\s+war\b", 1917, 1922),
    (r"\bamerican\s+revolution(?:ary\s+war)?\b|\brevolutionary\s+war\b|\bwar\s+of\s+independence\b", 1765, 1783),
    (r"\bfrench\s+revolution(?:ary)?\b", 1789, 1799),
    (r"\brussian\s+revolution\b|\boctober\s+revolution\b|\bbolshevik\s+revolution\b", 1917, 1917),
    (r"\bnapoleonic\s+wars?\b|\bnapoleonic\s+era\b", 1799, 1815),
    (r"\bwar\s+of\s+1812\b", 1812, 1815),
    (r"\bseven\s+years'?\s*war\b", 1756, 1763),
    (r"\bthirty\s+years'?\s*war\b", 1618, 1648),
    (r"\bhundred\s+years'?\s*war\b", 1337, 1453),
    (r"\bwars?\s+of\s+the\s+roses\b", 1455, 1487),
    (r"\bcrimean\s+war\b", 1853, 1856),
    (r"\bboer\s+war\b", 1899, 1902),
    (r"\bfranco[- ]prussian\s+war\b", 1870, 1871),
    (r"\bopium\s+wars?\b", 1839, 1860),
    (r"\bpeloponnesian\s+war\b", -431, -404),
    (r"\bpunic\s+wars?\b", -264, -146),
    (r"\bpersian\s+wars?\b|\bgreco[- ]persian\s+wars?\b", -499, -449),
    (r"\b(?:the\s+)?crusades?\b", 1096, 1291),
    (r"\breconquista\b", 718, 1492),
    (r"\bmexican[- ]american\s+war\b", 1846, 1848),
    (r"\bspanish[- ]american\s+war\b", 1898, 1898),
    (r"\bfalklands\s+war\b", 1982, 1982),
    (r"\bsix[- ]day\s+war\b", 1967, 1967),
    (r"\byom\s+kippur\s+war\b", 1973, 1973),
    (r"\bholocaust\b", 1941, 1945),
    (r"\binterwar\b|\binter-war\b|\bbetween\s+the\s+wars\b", 1918, 1939),
    (r"\bpost[- ]war\b|\bpostwar\b", 1945, 1960),
    (r"\bpre[- ]war\b|\bprewar\b", 1919, 1939),
    # Periods (Europe / general)
    (r"\bstone\s+age\b|\bpal(?:a)?eolithic\b", -50000, -10000),
    (r"\bmesolithic\b", -10000, -6000),
    (r"\bneolithic\b", -10000, -3000),
    (r"\bchalcolithic\b|\bcopper\s+age\b", -4500, -3300),
    (r"\bbronze\s+age\b", -3300, -1200),
    (r"\biron\s+age\b", -1200, -500),
    (r"\bclassical\s+antiquity\b|\bantiquity\b|\bancient\s+world\b", -800, 500),
    (r"\bancient\s+(?:greece|greek|greeks)\b|\bclassical\s+greece\b", -800, -146),
    (r"\bhellenistic\b", -323, -31),
    (r"\bancient\s+egypt(?:ian)?\b|\bpharaonic\b", -3100, -30),
    (r"\bold\s+kingdom\b", -2686, -2181),
    (r"\bnew\s+kingdom\b", -1550, -1070),
    (r"\broman\s+republic\b", -509, -27),
    (r"\broman\s+empire\b|\bimperial\s+rome\b|\bancient\s+rome\b|\bancient\s+romans?\b", -27, 476),
    (r"\bfall\s+of\s+(?:the\s+)?(?:western\s+)?roman\s+empire\b|\bfall\s+of\s+rome\b", 400, 476),
    (r"\blate\s+antiquity\b", 284, 700),
    (r"\bbyzantine\b|\beastern\s+roman\s+empire\b", 330, 1453),
    (r"\bfall\s+of\s+constantinople\b", 1453, 1453),
    (r"\bdark\s+ages\b|\bearly\s+middle\s+ages\b|\bearly\s+medieval\b", 476, 1000),
    (r"\bhigh\s+middle\s+ages\b|\bhigh\s+medieval\b", 1000, 1250),
    (r"\blate\s+middle\s+ages\b|\blate\s+medieval\b", 1250, 1500),
    (r"\bmiddle\s+ages\b|\bmedieval\b|\bmediaeval\b", 476, 1500),
    (r"\bviking\s+age\b|\bvikings?\b", 793, 1066),
    (r"\bnorman\s+conquest\b", 1066, 1066),
    (r"\banglo[- ]saxons?\b", 450, 1066),
    (r"\bcarolingian\b|\bcharlemagne\b", 751, 888),
    (r"\bblack\s+death\b", 1346, 1353),
    (r"\brenaissance\b", 1400, 1600),
    (r"\bage\s+of\s+discovery\b|\bage\s+of\s+exploration\b", 1415, 1650),
    (r"\b(?:protestant\s+)?reformation\b", 1517, 1648),
    (r"\bcounter[- ]reformation\b", 1545, 1648),
    (r"\bearly\s+modern\b", 1500, 1800),
    (r"\b(?:age\s+of\s+)?enlightenment\b", 1685, 1815),
    (r"\bindustrial\s+revolution\b", 1760, 1840),
    (r"\btudors?\b|\btudor\s+(?:era|period|england)\b", 1485, 1603),
    (r"\belizabethan\b", 1558, 1603),
    (r"\bjacobean\b", 1603, 1625),
    (r"\bstuart\s+(?:era|period|england)\b", 1603, 1714),
    (r"\bgeorgian\s+(?:era|period|england|britain)\b", 1714, 1837),
    (r"\bregency\s+(?:era|period|england)\b", 1811, 1820),
    (r"\bvictorian\b", 1837, 1901),
    (r"\bedwardian\b", 1901, 1910),
    (r"\bbelle\s+[ée]poque\b", 1871, 1914),
    (r"\bgilded\s+age\b", 1870, 1900),
    (r"\bprogressive\s+era\b", 1896, 1916),
    (r"\bantebellum\b", 1812, 1861),
    (r"\breconstruction\s+era\b|\b(?-i:Reconstruction)\b", 1865, 1877),
    (r"\bwild\s+west\b|\bold\s+west\b", 1865, 1895),
    (r"\bprohibition\b", 1920, 1933),
    (r"\broaring\s+twenties\b|\bjazz\s+age\b", 1920, 1929),
    (r"\bgreat\s+depression\b", 1929, 1939),
    (r"\bnew\s+deal\b", 1933, 1939),
    (r"\bweimar\b", 1918, 1933),
    (r"\bnazi\s+(?:germany|era|regime)\b|\bthird\s+reich\b", 1933, 1945),
    (r"\bmccarthy(?:ism|\s+era)\b", 1947, 1957),
    (r"\bcivil\s+rights\s+(?:movement|era)\b", 1954, 1968),
    (r"\bspace\s+race\b", 1955, 1975),
    (r"\bsoviet\s+union\b|\bussr\b|\bsoviet\s+era\b", 1922, 1991),
    (r"\bcollapse\s+of\s+the\s+soviet\s+union\b|\bfall\s+of\s+the\s+soviet\s+union\b|\bdissolution\s+of\s+the\s+soviet\s+union\b", 1991, 1991),
    (r"\bfall\s+of\s+the\s+berlin\s+wall\b", 1989, 1989),
    (r"\btsarist\b|\bimperial\s+russia\b|\brussian\s+empire\b", 1721, 1917),
    (r"\bottoman\s+empire\b|\bottomans?\b", 1299, 1922),
    (r"\bholy\s+roman\s+empire\b", 800, 1806),
    (r"\bmughal\b", 1526, 1857),
    (r"\bbritish\s+raj\b", 1858, 1947),
    (r"\bpartition\s+of\s+india\b", 1947, 1947),
    (r"\bcolonial\s+(?:era|period|america)\b", 1607, 1776),
    (r"\bpre[- ]columbian\b", -3000, 1492),
    (r"\baztecs?\b", 1300, 1521),
    (r"\bincas?\b|\binca\s+empire\b", 1438, 1533),
    (r"\bmayan?\s+(?:civilization|civilisation|empire)\b|\bclassic\s+maya\b", 250, 900),
    # East Asia
    (r"\bhan\s+dynasty\b", -206, 220),
    (r"\btang\s+dynasty\b", 618, 907),
    (r"\bsong\s+dynasty\b", 960, 1279),
    (r"\byuan\s+dynasty\b", 1271, 1368),
    (r"\bming\s+dynasty\b|\bming\s+(?:china|era|period)\b", 1368, 1644),
    (r"\bqing\s+dynasty\b|\bqing\s+(?:china|era|period)\b", 1644, 1912),
    (r"\bwarring\s+states\b", -475, -221),
    (r"\bqin\s+dynasty\b", -221, -206),
    (r"\bthree\s+kingdoms\b", 220, 280),
    (r"\bcultural\s+revolution\b", 1966, 1976),
    (r"\bgreat\s+leap\s+forward\b", 1958, 1962),
    (r"\bedo\s+(?:period|era|japan)\b|\btokugawa\s+shogunate\b", 1603, 1868),
    (r"\bmeiji\b", 1868, 1912),
    (r"\bsengoku\b", 1467, 1615),
    (r"\bheian\b", 794, 1185),
    (r"\bkamakura\b", 1185, 1333),
    (r"\bmongol\s+empire\b|\bgenghis\s+khan\b", 1206, 1368),
    # Recent events
    (r"\b9/11\b|\bseptember\s+11(?:th)?\s+attacks?\b", 2001, 2001),
    (r"\bdot[- ]?com\s+(?:bubble|boom|crash)\b", 1995, 2001),
    (r"\bgreat\s+recession\b|\b(?:2007|2008)[-–](?:08|09|2008|2009)\s+financial\s+crisis\b|\bglobal\s+financial\s+crisis\b|\bsubprime\s+(?:mortgage\s+)?crisis\b", 2007, 2009),
    (r"\beuro(?:zone)?\s+(?:debt\s+)?crisis\b", 2009, 2012),
    (r"\barab\s+spring\b", 2010, 2012),
    (r"\bbrexit\b", 2016, 2020),
    (r"\bcovid(?:-?19)?\b|\bcoronavirus\s+pandemic\b|\bsars-cov-2\b", 2020, 2022),
    (r"\bfinancial\s+crisis\s+of\s+2008\b", 2008, 2008),
    # Crypto milestones (bitcoin / cardano / monero / iota domains)
    (r"\bgenesis\s+block\b|\bsatoshi\s+nakamoto\b", 2008, 2009),
    (r"\bmt\.?\s?gox\b", 2010, 2014),
    (r"\bsegwit\b|\bsegregated\s+witness\b", 2015, 2017),
    (r"\btaproot\b", 2018, 2021),
    (r"\bblock\s+size\s+wars?\b", 2015, 2017),
    (r"\bbitcoin\s+cash\b", 2017, 2017),
    (r"\b(?-i:Byron)\s+(?:era|mainnet|wallet)\b", 2017, 2020),
    (r"\b(?-i:Shelley)\s+(?:era|hard\s+fork|mainnet)\b", 2020, 2021),
    (r"\b(?-i:Allegra)\s+(?:era|hard\s+fork)\b|\b(?-i:Mary)\s+(?:era|hard\s+fork)\b", 2020, 2021),
    (r"\b(?-i:Alonzo)\b", 2021, 2022),
    (r"\b(?-i:Vasil)\b", 2022, 2022),
    (r"\b(?-i:Chang)\s+(?:hard\s+fork|upgrade)\b", 2024, 2025),
    (r"\bringct\b|\bring\s+confidential\s+transactions\b", 2017, 2017),
    (r"\brandomx\b", 2019, 2019),
    (r"\bbulletproofs\b", 2018, 2018),
    (r"\bchrysalis\b", 2021, 2021),
]


def _split_top(pat: str, sep: str = "|") -> list[str]:
    """Split a regex on top-level alternation bars."""
    parts, depth, cls, buf, i = [], 0, False, [], 0
    while i < len(pat):
        c = pat[i]
        if c == "\\":
            buf.append(pat[i:i + 2])
            i += 2
            continue
        if cls:
            cls = c != "]"
        elif c == "[":
            cls = True
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c == sep and depth == 0:
            parts.append("".join(buf))
            buf = []
            i += 1
            continue
        buf.append(c)
        i += 1
    parts.append("".join(buf))
    return parts


def _required_literal(alt: str) -> str | None:
    """Longest literal letter run that every match of `alt` must contain (lowercased)."""
    out, i = [], 0
    while i < len(alt):
        c = alt[i]
        if c == "\\":
            out.append(" ")
            i += 2
            continue
        if c == "[":
            j = alt.index("]", i + 1)
            out.append(" ")
            i = j + 1
            continue
        if c == "(":
            # find matching paren
            depth, j = 0, i
            while j < len(alt):
                if alt[j] == "\\":
                    j += 2
                    continue
                depth += alt[j] == "("
                depth -= alt[j] == ")"
                if depth == 0:
                    break
                j += 1
            inner = alt[i + 1:j]
            optional = j + 1 < len(alt) and alt[j + 1] in "?*"
            if inner.startswith("?-i:"):
                inner = inner[4:]
            elif inner.startswith("?:"):
                inner = inner[2:]
            if optional or len(_split_top(inner)) > 1:
                out.append(" ")
            else:
                out.append(" " + inner + " ")
            i = j + 1
            continue
        if i + 1 < len(alt) and alt[i + 1] in "?*":
            out.append(" ")
            i += 1
            continue
        out.append(c if c.isalpha() or c.isdigit() or c == "/" else " ")
        i += 1
    flat = "".join(out)
    if "(" in flat or "\\" in flat:  # nested group survived: recurse once
        return _required_literal(flat)
    runs = re.findall(r"[a-z0-9/]{2,}", flat.lower())
    return max(runs, key=len) if runs else None


def _triggers(pat: str) -> tuple[str, ...] | None:
    lits = [_required_literal(a) for a in _split_top(pat)]
    return None if any(t is None for t in lits) else tuple(set(lits))


# Each era regex only runs when one of its required literals occurs in the
# lowercased text; a plain substring test is ~100x cheaper than the regex scan.
_ERA_RE = [(re.compile(p, re.I), _triggers(p), s, e) for p, s, e in _ERAS]

# ------------------------------------------------------------------ patterns --
_YEAR4 = r"(1\d{3}|20[0-2]\d)"
_SEP = r"\s*(?:-|–|—|to|until|till|through|thru)\s*"

_DATE_RE = re.compile(
    rf"\b(?:\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?{_MONTHS},?\s+{_YEAR4}"
    rf"|{_MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+{_YEAR4}"
    rf"|{_YEAR4}-\d{{2}}-\d{{2}}"
    rf"|\d{{1,2}}[/.]\d{{1,2}}[/.]{_YEAR4})\b", re.I)
_MONTH_YEAR_RE = re.compile(rf"\b{_MONTHS}\s+(?:of\s+)?{_YEAR4}\b(?!\s*[,.]?\d)", re.I)

_RANGE_RE = re.compile(
    rf"(?:\b(?:from|between)\s+)?(?<![\d$£€#.,])\b{_YEAR4}"
    rf"(?:{_SEP}|\s+and\s+)(?:{_YEAR4}|(\d{{2}}))\b(?![\d.,]\d|-\d)", re.I)
_BC_RANGE_RE = re.compile(rf"\b{_NUM_BC}\s*(?:{_BC_STRICT})?{_SEP}{_NUM_BC}\s*{_BC_STRICT}{_END}", re.I)

_BC_YEAR_RE = re.compile(rf"\b{_NUM_BC}\s*{_BC_STRICT}{_END}")
_AD_YEAR_RE = re.compile(rf"(?:\b{_AD_STRICT}\s+(\d{{1,4}})\b(?![.,]\d)|\b(\d{{1,4}})\s*{_AD_STRICT}{_END})")

_CENT_MOD = r"(?:(early|mid|middle|late|first\s+half\s+of\s+the|second\s+half\s+of\s+the|turn\s+of\s+the|end\s+of\s+the|beginning\s+of\s+the|start\s+of\s+the)[\s-]+)?"
_ORD = r"(\d{1,2})(?:st|nd|rd|th)|(" + "|".join(sorted(_ORD_WORDS, key=len, reverse=True)) + ")"
_CENTURY_RE = re.compile(
    rf"\b{_CENT_MOD}(?:the\s+)?(?:{_ORD})[\s-]+centur(?:y|ies)(?:\s*({_BC}|{_AD}))?", re.I)
_MILLENNIUM_RE = re.compile(rf"\b(?:{_ORD})[\s-]+millenni(?:um|a)(?:\s*({_BC}|{_AD}))?", re.I)

_DECADE_RE = re.compile(
    rf"\b{_CENT_MOD}(?:the\s+)?(?:(1\d|20)(\d)0'?s\b|'(\d)0s\b|(?<!\d)(\d)0s\b"
    rf"|({'|'.join(_DECADE_WORDS)})\b)", re.I)

_REL_BEFORE = r"(?:before|prior\s+to|pre-|pre\s|until|till|up\s+to|earlier\s+than|by)"
_REL_AFTER = r"(?:after|since|post-|post\s|later\s+than|from)"
_REL_RE = re.compile(rf"\b(?:({_REL_BEFORE})|({_REL_AFTER}))\s*(?:the\s+)?(?=\S)", re.I)

_PRESENT_RE = re.compile(
    r"\b(?:nowadays|currently|these\s+days|at\s+present|at\s+the\s+moment|as\s+of\s+(?:now|today)|"
    r"today|right\s+now|in\s+recent\s+years|most\s+recent|up[- ]to[- ]date|still\s+(?:valid|true|the\s+case|used|relevant))\b",
    re.I)


def _century_bounds(n: int, bc: bool) -> tuple[int, int]:
    if bc:
        return -n * 100, -(n - 1) * 100 - 1
    return (n - 1) * 100, n * 100 - 1


def _apply_mod(mod: str | None, s: int, e: int) -> tuple[int, int]:
    if not mod:
        return s, e
    mod = " ".join(mod.lower().split())
    span = e - s + 1
    third = span // 3
    if mod == "early" or mod.startswith(("first half", "beginning", "start")):
        return s, s + (span // 2 if "half" in mod else third) - 1
    if mod in ("mid", "middle"):
        return s + third, e - third
    if mod == "late" or mod.startswith(("second half", "end")):
        return e - (span // 2 if "half" in mod else third) + 1, e
    if mod.startswith("turn"):
        return e - span // 10, e + span // 10
    return s, e


def _to_int(num: str) -> int:
    return int(re.sub(r"[ ,]", "", num))


class _Masker:
    def __init__(self, text: str):
        self.text = text
        self.taken: list[tuple[int, int]] = []

    def free(self, a: int, b: int) -> bool:
        return all(b <= x or a >= y for x, y in self.taken)

    def take(self, a: int, b: int) -> None:
        self.taken.append((a, b))


def _plain_year_ok(text: str, a: int, b: int) -> bool:
    before = text[max(0, a - 12):a]
    if _PREFIX_BEFORE.search(before):
        return False
    after = text[b:b + 16]
    if _UNIT_AFTER.match(after):
        return False
    # Thousands groups / decimals / IDs: 1,500 or 1.500 or 1500.25 or 0x1500
    if a > 0 and text[a - 1] in ",.:/_-+*x" and a > 1 and text[a - 2].isdigit():
        return False
    if b < len(text) and text[b] in ",." and b + 1 < len(text) and text[b + 1].isdigit():
        return False
    if b < len(text) and (text[b].isalpha() and text[b:b + 2] not in ("s ", "s,", "s.") or text[b] in "_"):
        return False
    if a > 0 and (text[a - 1].isalpha() or text[a - 1] == "_"):
        return False
    return True


def extract(text: str, present: bool = False, max_exprs: int = 300) -> list[TimeExpr]:
    """Extract time expressions from raw (possibly HTML) text."""
    text = clean_text(text)
    low = text.lower()
    m = _Masker(text)
    out: list[TimeExpr] = []
    pos: list[int] = []

    def add(a, b, s, e, kind):
        if s > e:
            s, e = e, s
        hi = PRESENT_YEAR + 5
        s, e = min(max(s, MIN_YEAR), hi), min(max(e, MIN_YEAR), hi)
        m.take(a, b)
        out.append(TimeExpr(s, e, kind, text[a:b]))
        pos.append(a)

    # 1. explicit dates (dateparser normalises the span)
    for mt in _DATE_RE.finditer(text):
        y = _parse_date_year(mt.group(0))
        if y is not None and m.free(*mt.span()):
            add(*mt.span(), y, y, "date")
    for mt in _MONTH_YEAR_RE.finditer(text):
        if m.free(*mt.span()):
            y = int(mt.group(1))
            add(*mt.span(), y, y, "date")

    # 2. ranges
    for mt in _BC_RANGE_RE.finditer(text):
        if m.free(*mt.span()):
            s, e = -_to_int(mt.group(1)), -_to_int(mt.group(2))
            if abs(e - s) <= 3000:
                add(*mt.span(), s, e, "range")
    for mt in _RANGE_RE.finditer(text):
        if not m.free(*mt.span()):
            continue
        s = int(mt.group(1))
        if mt.group(2):
            e = int(mt.group(2))
        else:
            e = s // 100 * 100 + int(mt.group(3))
        if s < e <= s + 300:
            add(*mt.span(), s, e, "range")

    # 3. centuries / millennia
    for mt in (_CENTURY_RE.finditer(text) if "centur" in low else ()):
        if not m.free(*mt.span()):
            continue
        n = int(mt.group(2)) if mt.group(2) else _ORD_WORDS[" ".join(mt.group(3).lower().split())]
        if not 1 <= n <= 21:
            continue
        bc = bool(mt.group(4) and re.match(_BC, mt.group(4), re.I))
        s, e = _apply_mod(mt.group(1), *_century_bounds(n, bc))
        add(*mt.span(), s, e, "century")
    for mt in (_MILLENNIUM_RE.finditer(text) if "millenni" in low else ()):
        if not m.free(*mt.span()):
            continue
        n = int(mt.group(1)) if mt.group(1) else _ORD_WORDS[" ".join(mt.group(2).lower().split())]
        if not 1 <= n <= 12:
            continue
        bc = bool(mt.group(3) and re.match(_BC, mt.group(3), re.I))
        s, e = (-n * 1000, -(n - 1) * 1000 - 1) if bc else ((n - 1) * 1000, n * 1000 - 1)
        add(*mt.span(), s, e, "millennium")

    # 4. decades ("1800s" is a century, "1840s" a decade)
    has_decade = "0s" in low or "0's" in low or "ties" in low
    for mt in (_DECADE_RE.finditer(text) if has_decade else ()):
        if not m.free(*mt.span()):
            continue
        mod = mt.group(1)
        if mt.group(2):
            c, d = int(mt.group(2)), int(mt.group(3))
            base = c * 100 + d * 10
            if d == 0:
                s, e = c * 100, c * 100 + 99
                kind = "century"
            else:
                s, e, kind = base, base + 9, "decade"
        else:
            dd = mt.group(4) or mt.group(5)
            if dd is not None:
                base = 1900 + int(dd) * 10
            else:
                base = 1900 + _DECADE_WORDS[mt.group(6).lower()]
            s, e, kind = base, base + 9, "decade"
        s, e = _apply_mod(mod, s, e)
        add(*mt.span(), s, e, kind)

    # 5. era-marked years
    for mt in _BC_YEAR_RE.finditer(text):
        if m.free(*mt.span()):
            y = -_to_int(mt.group(1))
            after = text[mt.end():mt.end() + 2]
            before = text[max(0, mt.start() - 8):mt.start()].lower()
            if "million" in before or "billion" in before or after.startswith(("-", "–")):
                continue
            add(*mt.span(), y, y, "year")
    for mt in _AD_YEAR_RE.finditer(text):
        if m.free(*mt.span()):
            y = int(mt.group(1) or mt.group(2))
            if 1 <= y <= PRESENT_YEAR:
                add(*mt.span(), y, y, "year")

    # 6. named eras and events
    for rx, trig, s, e in _ERA_RE:
        if trig is not None and not any(t in low for t in trig):
            continue
        for mt in rx.finditer(text):
            if m.free(*mt.span()):
                add(*mt.span(), s, e, "era")

    # 7. plain years
    for mt in re.finditer(rf"(?<!\d){_YEAR4}(?!\d)", text):
        a, b = mt.span()
        if m.free(a, b) and _plain_year_ok(text, a, b):
            y = int(mt.group(1))
            if PLAIN_YEAR_MIN <= y <= PLAIN_YEAR_MAX:
                add(a, b, y, y, "year")

    # 8. relative expressions: turn the expression right after "before"/"since" into an open interval
    if out:
        starts = sorted((p, i) for i, p in enumerate(pos))
        for mt in _REL_RE.finditer(text):
            nxt = mt.end()
            for p, i in starts:
                if nxt <= p <= nxt + 2:
                    t = out[i]
                    word = (mt.group(1) or mt.group(2)).lower().strip()
                    if mt.group(1) and word != "by":
                        out[i] = TimeExpr(MIN_YEAR, t.start, "relative", mt.group(0) + t.text)
                    elif mt.group(2) and word != "from" and t.kind not in ("range",):
                        out[i] = TimeExpr(t.end, PRESENT_YEAR, "relative", mt.group(0) + t.text)
                    break

    if present:
        for mt in _PRESENT_RE.finditer(text):
            if m.free(*mt.span()):
                add(*mt.span(), PRESENT_YEAR - 4, PRESENT_YEAR, "present")

    return out[:max_exprs]


def intervals(exprs: list[TimeExpr]) -> list[tuple[int, int]]:
    return [(t.start, t.end) for t in exprs]
