#!/usr/bin/env python3
"""One-off: star in Navidrome every track matching an exported favorites list.

Navidrome starts with nothing starred. This script reads an exported list of
favorites (the fav.json generated from Tidal, or a two-column Artist,Title
CSV), matches them against the whole Navidrome library by normalized
artist+title, and stars the matches through the Subsonic API.

Usage:
    python star_favorites.py --fav-json /tmp/fav.json --dry-run   # check match rate
    python star_favorites.py --fav-json /tmp/fav.json             # apply
    python star_favorites.py --fav-csv favorites.csv
"""
import argparse
import configparser
import csv
import hashlib
import json
import secrets
import sys
import unicodedata

import requests

SEARCH_PAGE = 500
STAR_CHUNK = 50


def normalize(s):
    """Lowercase, strip accents and punctuation so near-identical spellings match."""
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    return "".join(c.lower() for c in s if c.isalnum())


def auth_params(username, password, client_name, api_version):
    salt = secrets.token_hex(6)
    token = hashlib.md5((password + salt).encode("utf-8")).hexdigest()
    return {"u": username, "t": token, "s": salt, "v": api_version,
            "c": client_name, "f": "json"}


def subsonic_get(base_url, endpoint, params, extra=None):
    resp = requests.get(
        f"{base_url}/rest/{endpoint}",
        params=list(params.items()) + list(extra or []),
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json().get("subsonic-response", {})
    if payload.get("status") != "ok":
        err = payload.get("error", {})
        raise RuntimeError(f"Subsonic error {err.get('code')}: {err.get('message')}")
    return payload


def _entry_field(entry, *names):
    for name in names:
        value = entry.get(name)
        if value:
            return value
    return ""


def load_favorites(fav_json=None, fav_csv=None):
    """Return a list of ``(artist, title)`` pairs from a JSON or CSV export."""
    pairs = []
    if fav_json:
        with open(fav_json, encoding="utf-8") as fh:
            favorites = json.load(fh)
        for entry in favorites:
            artist = _entry_field(entry, "artist", "Artist")
            title = _entry_field(entry, "title", "Title", "song", "Song", "name")
            if artist or title:
                pairs.append((artist, title))
    else:
        with open(fav_csv, encoding="utf-8", newline="") as fh:
            for row in csv.reader(fh):
                if len(row) < 2:
                    continue
                artist, title = row[0].strip(), row[1].strip()
                # Skip an eventual header line.
                if (artist.lower(), title.lower()) in (
                    ("artist", "title"), ("artiste", "titre"),
                ):
                    continue
                if artist or title:
                    pairs.append((artist, title))
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Star tracks in Navidrome from an exported favorites list."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--fav-json", help="Path to fav.json (Tidal export)")
    source.add_argument("--fav-csv", help="Path to a two-column Artist,Title CSV")
    parser.add_argument("--config", default="settings.ini", help="Path to settings.ini")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only report the match rate, do not star anything.")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)
    base_url = config.get("NAVIDROME", "url").rstrip("/")
    username = config.get("NAVIDROME", "username")
    password = config.get("NAVIDROME", "password")
    client_name = config.get("NAVIDROME", "client_name", fallback="likes_songs_sorter")
    api_version = config.get("NAVIDROME", "api_version", fallback="1.16.1")

    favorites = load_favorites(args.fav_json, args.fav_csv)
    if not favorites:
        print("No favorites found in the export. Nothing to do.", file=sys.stderr)
        sys.exit(1)
    wanted = {(normalize(artist), normalize(title)) for artist, title in favorites}

    # Index the whole Navidrome library by normalized (artist, title). An empty
    # search3 query pages through every song (OpenSubsonic behaviour).
    index = {}
    offset = 0
    while True:
        params = auth_params(username, password, client_name, api_version)
        params.update({"query": "", "songCount": SEARCH_PAGE, "songOffset": offset})
        payload = subsonic_get(base_url, "search3", params)
        songs = payload.get("searchResult3", {}).get("song", [])
        if not songs:
            break
        for s in songs:
            index.setdefault(
                (normalize(s.get("artist")), normalize(s.get("title"))), s["id"]
            )
        offset += SEARCH_PAGE
    print(f"Indexed {len(index)} unique (artist, title) pairs from Navidrome.")

    matched = [index[k] for k in sorted(wanted) if k in index]
    missing = sorted(k for k in wanted if k not in index)
    print(f"{len(matched)}/{len(wanted)} favorites found locally in Navidrome.")
    if missing:
        print("Sample of unmatched favorites (normalized):")
        for artist, title in missing[:10]:
            print(f"   • {artist} — {title}")
        if len(missing) > 10:
            print(f"   … and {len(missing) - 10} more.")

    if args.dry_run:
        print("Dry run: nothing starred.")
        return

    for i in range(0, len(matched), STAR_CHUNK):
        chunk = matched[i:i + STAR_CHUNK]
        params = auth_params(username, password, client_name, api_version)
        subsonic_get(base_url, "star", params, extra=[("id", sid) for sid in chunk])
    print(f"{len(matched)} tracks marked as starred.")


if __name__ == "__main__":
    main()
