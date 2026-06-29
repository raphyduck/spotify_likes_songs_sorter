"""Manual genre overrides — the deterministic guarantee layer.

Automatic genre resolution is inherently fuzzy. Overrides let you pin the genre
tags and/or root family for a specific artist (or artist+album) with absolute
priority over every provider and over root inference. Matching is
case/accent-insensitive.

``genre_overrides.json`` format::

    {
      "overrides": [
        {"match": "Sigur Rós",            "tags": ["Post-Rock", "Ambient"], "root": "rock"},
        {"match": "Juanes|Mi Sangre",     "tags": ["Latin Pop"],            "root": "latin"}
      ]
    }

``match`` is either ``"Artist"`` or ``"Artist|Album"``; the more specific
``Artist|Album`` key wins when both are present.
"""

import json
import os
import unicodedata

DEFAULT_OVERRIDES_FILE = os.path.join(os.path.dirname(__file__), "genre_overrides.json")


def _strip_accents(value):
    return "".join(
        c for c in unicodedata.normalize("NFKD", value) if not unicodedata.combining(c)
    )


def _norm_key(value):
    return " ".join(_strip_accents(str(value or "")).casefold().split())


def _norm_match(value):
    return "|".join(_norm_key(part) for part in str(value or "").split("|"))


def load_overrides(path=None):
    """Load overrides into a dict keyed by normalized match string.

    Returns ``{}`` on any problem (missing file, bad JSON), so overrides are
    simply inactive rather than fatal.
    """
    path = path or DEFAULT_OVERRIDES_FILE
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    entries = data.get("overrides", []) if isinstance(data, dict) else []
    table = {}
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("match"):
            continue
        key = _norm_match(entry["match"])
        if not key:
            continue
        table[key] = {
            "tags": list(entry.get("tags") or []),
            "root": (entry.get("root") or "").strip().lower() or None,
        }
    return table


def lookup_override(overrides, artist, album):
    """Return the override entry for an artist/album, or ``None``.

    The ``Artist|Album`` key takes precedence over the ``Artist`` key.
    """
    if not overrides:
        return None
    artist_key = _norm_key(artist)
    album_key = _norm_key(album)
    for key in (f"{artist_key}|{album_key}", artist_key):
        if key in overrides:
            return overrides[key]
    return None
