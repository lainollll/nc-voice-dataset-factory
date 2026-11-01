from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from difflib import SequenceMatcher

from .storage import canonical_text

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_THRESHOLD = 0.90
_LEN_RATIO = 0.90


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(canonical_text(text))


def _similarity(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return 1.0 if a == b else 0.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


class _DSU:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def _validate(records: list[dict], validation_fraction: float) -> None:
    if not isinstance(records, list) or not records:
        raise ValueError("records must be a non-empty list")
    if not (isinstance(validation_fraction, (int, float)) and 0.0 <= validation_fraction <= 0.5):
        raise ValueError("validation_fraction must be in [0, 0.5]")
    seen: set[str] = set()
    for rec in records:
        rid = rec.get("id")
        lang = rec.get("language")
        text = rec.get("training_text")
        if not isinstance(rid, str) or not rid:
            raise ValueError("record id must be a non-empty string")
        if rid in seen:
            raise ValueError(f"duplicate id: {rid!r}")
        seen.add(rid)
        if lang not in ("en", "sv"):
            raise ValueError(f"unsupported language: {lang!r}")
        if not isinstance(text, str):
            raise ValueError(f"training_text must be a string for id: {rid!r}")


def _groups(records: list[dict]) -> list[list[str]]:
    records = sorted(records, key=lambda r: r["id"])
    n = len(records)
    toks = [_tokens(r["training_text"]) for r in records]
    lens = [len(t) for t in toks]
    dsu = _DSU(n)

    # Exact grouping per language using normalized token join.
    exact: dict[tuple[str, tuple[str, ...]], int] = {}
    for i, rec in enumerate(records):
        key = (rec["language"], tuple(toks[i]))
        if key in exact:
            dsu.union(exact[key], i)
        else:
            exact[key] = i

    # Build token postings per language for necessary-bound candidate generation.
    # A pair whose SequenceMatcher token ratio >= 0.90 must share at least
    # ceil(0.90 * (lenA + lenB) / 2) multiset token mass in common.
    postings: dict[str, dict[str, list[int]]] = {"en": defaultdict(list), "sv": defaultdict(list)}
    counts: list[dict[str, int]] = []
    for i, rec in enumerate(records):
        c: dict[str, int] = defaultdict(int)
        for t in toks[i]:
            c[t] += 1
        counts.append(c)
        for t in c:
            postings[rec["language"]][t].append(i)

    for i, rec in enumerate(records):
        li = lens[i]
        if li == 0:
            continue
        # candidate accumulation: shared token mass lower bound uses min counts.
        shared: dict[int, int] = defaultdict(int)
        for tok, ci in counts[i].items():
            for j in postings[rec["language"]][tok]:
                if j == i:
                    continue
                if records[j]["language"] != rec["language"]:
                    continue
                shared[j] += min(ci, counts[j][tok])
        for j, common in shared.items():
            if j <= i:
                continue
            lj = lens[j]
            if lj == 0:
                continue
            short, long_ = (li, lj) if li <= lj else (lj, li)
            if 2 * short / (short + long_) < _THRESHOLD:
                continue
            need = math.ceil(_THRESHOLD * (li + lj) / 2.0)
            if common < need:
                continue
            if dsu.find(i) == dsu.find(j):
                continue
            if _similarity(toks[i], toks[j]) >= _THRESHOLD:
                dsu.union(i, j)

    buckets: dict[int, list[str]] = defaultdict(list)
    for i, rec in enumerate(records):
        buckets[dsu.find(i)].append(rec["id"])
    return list(buckets.values())


def _rank(seed: int, key: str) -> str:
    return hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()


def grouped_split(records: list[dict], validation_fraction: float = 0.1, seed: int = 0) -> dict[str, str]:
    """Group near-identical transcripts (same language) and assign train/validation.

    Determinism: groups are formed by union of exact and >=0.90 SequenceMatcher
    token-similar pairs within the same language; canonical text comes from
    ``.storage.canonical_text``. Groups are then ordered by sha256(seed + sorted
    canonical member ids). If only one group exists, all records are 'train'
    (callers requesting validation must handle the no-groups case).

    Complexity: token postings give a shared-token-mass necessary bound for
    the similarity threshold; length-ratio bound also prunes. Worst case is
    O(n * tokens) candidate postings for repeated templates; exact SequenceMatcher
    comparisons are limited to pairs passing both necessary bounds and not
    already in the same DSU group.
    """
    _validate(records, validation_fraction)
    groups = _groups(records)
    groups.sort(key=lambda ids: _rank(seed, ",".join(sorted(ids))))
    result: dict[str, str] = {}
    if len(groups) <= 1:
        for ids in groups:
            for rid in ids:
                result[rid] = "train"
        return result

    total = len(records)
    target = validation_fraction * total
    if validation_fraction <= 0.0:
        for ids in groups:
            for rid in ids:
                result[rid] = "train"
        return result

    # Greedy: pick seeded-rank groups whose sizes bring us closest to target,
    # keep at least one group train.
    chosen: set[int] = set()
    running = 0
    for gi, ids in enumerate(groups):
        if len(chosen) < len(groups) - 1 and abs(running + len(ids) - target) < abs(running - target):
            chosen.add(gi)
            running += len(ids)
    if not chosen:
        # Whole-group assignment can only approximate the requested fraction.
        chosen.add(min(range(len(groups)), key=lambda gi: (len(groups[gi]), gi)))

    for gi, ids in enumerate(groups):
        label = "validation" if gi in chosen else "train"
        for rid in ids:
            result[rid] = label
    return result
