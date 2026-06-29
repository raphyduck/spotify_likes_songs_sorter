"""Genre root-family normalization for similarity and ordering.

Album genres are composite, fine-grained tag lists (e.g. ``Punk, Pop Punk,
Emo`` vs ``Punk, Skate Punk``). The album-similarity step already compares
*sets of tags*, but two albums from the same broad family can still look
distant when their specific sub-tags differ, which fragments the final order
(a family reappears at non-contiguous positions).

This module maps each tag to a coarse **root family** (data-driven, see
``genre_roots.json``) and builds a weighted feature space where the root tokens
carry extra weight (``root_weight``). Albums that share a root then sit closer
together, smoothing the ordering, while the original composite labels are still
kept for display in the CSV.
"""

import json
import os

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_ROOTS_FILE = os.path.join(os.path.dirname(__file__), "genre_roots.json")


def load_genre_roots(path=None):
    """Load the ordered list of root rules; returns ``[]`` on any problem."""
    path = path or DEFAULT_ROOTS_FILE
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        rules = data.get("rules", []) if isinstance(data, dict) else []
        return [
            {"root": r["root"], "keywords": [k.lower() for k in r.get("keywords", [])]}
            for r in rules
            if isinstance(r, dict) and r.get("root")
        ]
    except (OSError, ValueError, KeyError):
        return []


def _norm(tag):
    return " ".join(str(tag or "").strip().lower().split())


def root_of(tag, rules):
    """Map a single genre tag to its root family.

    First matching rule wins; unmatched tags fall back to their first word.
    """
    t = _norm(tag)
    if not t:
        return "unknown"
    for rule in rules:
        if any(kw in t for kw in rule["keywords"]):
            return rule["root"]
    return t.split()[0]


def primary_root(genre_list, rules):
    """Root family of an album, taken from its most prominent (first) tag."""
    for tag in genre_list or []:
        if _norm(tag):
            return root_of(tag, rules)
    return "unknown"


def infer_root(genre_list, rules):
    """Infer an album's root family from its *whole* tag set.

    Scans rules in priority order (file order) and returns the root of the first
    rule whose keyword matches ANY tag — so a strong signal anywhere in the list
    (e.g. ``post-rock`` among ``ambient, post-rock``) wins over tag position.
    Unmatched tag sets fall back to the most specific tag (the last one, since
    tags are sorted broad->niche), never a bare ``"unknown"``.
    """
    norm_tags = [_norm(t) for t in (genre_list or []) if _norm(t)]
    if not norm_tags:
        return "unknown"
    for rule in rules:
        for tag in norm_tags:
            if any(kw in tag for kw in rule["keywords"]):
                return rule["root"]
    return norm_tags[-1]


def display_root(root):
    return root.replace("_", " ").title()


def genre_similarity_matrix(genre_lists, rules, root_weight):
    """Cosine similarity over root-weighted tag vectors.

    Each album contributes weight 1 per tag plus ``root_weight`` per (deduped)
    root family of its tags, so shared families dominate the similarity.
    """
    rows = []
    for sub in genre_lists:
        feats = {}
        roots_seen = set()
        for tag in sub:
            t = _norm(tag)
            if not t:
                continue
            feats["tag:" + t] = feats.get("tag:" + t, 0.0) + 1.0
            roots_seen.add(root_of(t, rules))
        for r in roots_seen:
            feats["root:" + r] = feats.get("root:" + r, 0.0) + float(root_weight)
        rows.append(feats)

    vocab = sorted({token for feats in rows for token in feats})
    if not vocab:
        return np.zeros((len(rows), len(rows)))
    index = {token: i for i, token in enumerate(vocab)}
    matrix = np.zeros((len(rows), len(vocab)))
    for i, feats in enumerate(rows):
        for token, value in feats.items():
            matrix[i, index[token]] = value
    return cosine_similarity(matrix)


# -----------------------------
#  Ordering-quality metrics
# -----------------------------
def avg_adjacent_overlap(ordered_tag_sets):
    """Mean Jaccard overlap between consecutive albums in an ordering."""
    if len(ordered_tag_sets) < 2:
        return 0.0
    scores = []
    for a, b in zip(ordered_tag_sets, ordered_tag_sets[1:]):
        if not a and not b:
            scores.append(0.0)
            continue
        union = a | b
        scores.append(len(a & b) / len(union) if union else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def count_fragmented_roots(ordered_roots):
    """Number of root families that appear in more than one contiguous run."""
    runs = {}
    prev = None
    for root in ordered_roots:
        if root != prev:
            runs[root] = runs.get(root, 0) + 1
        prev = root
    return sum(1 for count in runs.values() if count > 1)
