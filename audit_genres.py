#!/usr/bin/env python3
"""Audit a sorted CSV for likely genre-classification errors.

Reads a CSV produced by ``sorter.py`` and flags albums that are probably
mis-classified, so you can add corrections to ``genre_overrides.json``. An album
is a suspect when its root family is ``Unknown``, its genre came from a
low-reliability single source, or its root differs from the majority root of the
same artist's other albums.

Usage:
    python audit_genres.py tidal_favorite_tracks_sorted_2026-06-29.csv
"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict

LOW_RELIABILITY_SOURCES = {"Spotify Artist", "iTunes", "LastFM Track", "None"}


def find_suspects(albums, low_sources=LOW_RELIABILITY_SOURCES):
    """Return suspect albums with the reasons they were flagged.

    ``albums`` is a list of dicts with keys ``artist``, ``album``, ``root``,
    ``source`` (one entry per unique album).
    """
    by_artist = defaultdict(list)
    for album in albums:
        by_artist[album["artist"]].append(album)
    majority = {}
    for artist, items in by_artist.items():
        counts = Counter(a["root"] for a in items)
        majority[artist] = counts.most_common(1)[0][0]

    suspects = []
    for album in albums:
        reasons = []
        if (album.get("root") or "").strip().lower() in ("", "unknown"):
            reasons.append("unknown root")
        if album.get("source") in low_sources:
            reasons.append(f"low-reliability source ({album.get('source')})")
        siblings = by_artist[album["artist"]]
        if len(siblings) >= 2 and album["root"] != majority[album["artist"]]:
            reasons.append(f"differs from artist majority root ({majority[album['artist']]})")
        if reasons:
            suspects.append({**album, "reasons": reasons})
    return suspects


def _albums_from_csv(path):
    albums = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = row.get("Unique Album") or f"{row.get('Album')}|{row.get('Artist')}"
            if key in albums:
                continue
            albums[key] = {
                "artist": row.get("Artist", ""),
                "album": row.get("Album", ""),
                "root": row.get("Root Genre", ""),
                "source": row.get("source", ""),
                "genres": row.get("Album Genre", ""),
            }
    return list(albums.values())


def _override_skeleton(suspects):
    return {
        "overrides": [
            {"match": s["artist"], "tags": [t.strip() for t in s["genres"].split(",") if t.strip()],
             "root": s["root"].lower() or "CHANGE_ME"}
            for s in suspects
        ]
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Audit a sorted CSV for genre errors.")
    parser.add_argument("csv_path", help="Path to a *_sorted_*.csv file")
    parser.add_argument("--json", action="store_true",
                        help="Emit a genre_overrides.json skeleton for the suspects.")
    args = parser.parse_args(argv)

    try:
        albums = _albums_from_csv(args.csv_path)
    except OSError as exc:
        print(f"Could not read {args.csv_path}: {exc}", file=sys.stderr)
        return 1

    suspects = find_suspects(albums)
    if not suspects:
        print(f"✅ No suspect albums found among {len(albums)} albums.")
        return 0

    print(f"⚠️  {len(suspects)} suspect album(s) of {len(albums)} — candidates for genre_overrides.json:\n")
    for s in suspects:
        print(f"  • {s['artist']} — {s['album']}  [root: {s['root'] or 'Unknown'}]")
        print(f"      {'; '.join(s['reasons'])}")
    if args.json:
        print("\n--- genre_overrides.json skeleton (edit before use) ---")
        print(json.dumps(_override_skeleton(suspects), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
