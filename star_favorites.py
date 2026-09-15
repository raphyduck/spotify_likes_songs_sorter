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
import difflib
import hashlib
import json
import secrets
import sys
import unicodedata
from collections import defaultdict

import requests

SEARCH_PAGE = 500
STAR_CHUNK = 50


def normalize(s):
    """Lowercase, strip accents and punctuation so near-identical spellings match."""
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    return "".join(c.lower() for c in s if c.isalnum())


def bigrams(s):
    """Set of adjacent character pairs, used to pre-filter fuzzy artist candidates."""
    return {s[i:i + 2] for i in range(len(s) - 1)}


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
    parser.add_argument(
        "--fuzzy-threshold", type=float, default=0.85,
        help="Title similarity (0-1) above which a non-exact favorite is matched "
             "by fuzzy fallback. Set to 1.0 to disable fuzzy matching (exact only).",
    )
    parser.add_argument(
        "--fuzzy-artist-threshold", type=float, default=0.85,
        help="Similarity (0-1) for matching a differently-spelled artist. "
             "Set >1 to disable artist fuzzy matching.",
    )
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
    # Keep an original spelling per normalized favorite for readable reporting.
    fav_orig = {}
    for artist, title in favorites:
        fav_orig.setdefault((normalize(artist), normalize(title)), (artist, title))

    # Index the whole Navidrome library by normalized (artist, title). An empty
    # search3 query pages through every song (OpenSubsonic behaviour). A second
    # index groups titles by normalized artist so the fuzzy fallback only ever
    # compares a favorite to titles by the *same* artist.
    index = {}
    by_artist = defaultdict(list)  # artist_n -> list of (title_n, song_id, title_orig)
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
            by_artist[normalize(s.get("artist"))].append(
                (normalize(s.get("title")), s["id"], s.get("title") or "")
            )
        offset += SEARCH_PAGE
    print(f"Indexed {len(index)} unique (artist, title) pairs from Navidrome.")

    # Pre-compute an artist bigram index once so the fuzzy artist fallback can
    # cheaply pre-filter candidates instead of comparing against every artist.
    artist_bigrams = {a: bigrams(a) for a in by_artist}

    def candidate_artists(fav_artist_n, min_overlap=2):
        fb = bigrams(fav_artist_n)
        if not fb:
            return []
        return [a for a, ab in artist_bigrams.items() if len(fb & ab) >= min_overlap]

    matched = [index[k] for k in sorted(wanted) if k in index]
    missing = sorted(k for k in wanted if k not in index)

    # Fuzzy fallback for favorites not matched exactly. The title must ALWAYS
    # clear fuzzy_threshold — the safety guarantee. Artist fuzzy matching only
    # widens the set of artists examined (for differently-spelled acts such as
    # "Katrina & The Waves" vs "Katrina and the Waves"); it never lets a weak
    # title through.
    title_fuzzy_on = args.fuzzy_threshold < 1.0
    artist_fuzzy_on = args.fuzzy_artist_threshold <= 1.0

    def best_title_in(candidates, title_n):
        best_id, best_sim, best_title = None, 0.0, None
        for cand_title_n, song_id, cand_title_orig in candidates:
            sim = difflib.SequenceMatcher(None, title_n, cand_title_n).ratio()
            if sim > best_sim:
                best_id, best_sim, best_title = song_id, sim, cand_title_orig
        return best_id, best_sim, best_title

    fuzzy_title = []   # (song_id, artist_n, fav_title_n, lib_title, title_sim)
    fuzzy_artist = []  # (song_id, fav_artist_n, fav_title_n, lib_artist_n, lib_title, a_sim, t_sim)
    still_missing = []
    for artist_n, title_n in missing:
        matched_here = False

        # 1. Exact artist + fuzzy title.
        if title_fuzzy_on and artist_n in by_artist:
            bid, bsim, btitle = best_title_in(by_artist[artist_n], title_n)
            if bsim >= args.fuzzy_threshold:
                fuzzy_title.append((bid, artist_n, title_n, btitle, bsim))
                matched_here = True

        # 2. Otherwise, fuzzy artist + fuzzy title.
        if not matched_here and title_fuzzy_on and artist_fuzzy_on:
            best = None  # (title_sim, artist_sim, song_id, lib_artist_n, lib_title)
            for cand_artist in candidate_artists(artist_n):
                if cand_artist == artist_n:
                    continue  # already tried as the exact artist in step 1
                a_sim = difflib.SequenceMatcher(None, artist_n, cand_artist).ratio()
                if a_sim < args.fuzzy_artist_threshold:
                    continue
                bid, bsim, btitle = best_title_in(by_artist[cand_artist], title_n)
                if bsim >= args.fuzzy_threshold and (
                    best is None or (bsim, a_sim) > (best[0], best[1])
                ):
                    best = (bsim, a_sim, bid, cand_artist, btitle)
            if best is not None:
                t_sim, a_sim, bid, lib_artist_n, lib_title = best
                fuzzy_artist.append(
                    (bid, artist_n, title_n, lib_artist_n, lib_title, a_sim, t_sim)
                )
                matched_here = True

        if not matched_here:
            still_missing.append((artist_n, title_n))

    print(f"{len(matched)}/{len(wanted)} exact matches.")
    if fuzzy_title:
        print(f"+{len(fuzzy_title)} fuzzy title matches "
              f"(artist exact, title >= {args.fuzzy_threshold}):")
        for song_id, artist_n, fav_title_n, lib_title, sim in sorted(
            fuzzy_title, key=lambda x: x[4], reverse=True
        ):
            print(f"   [{sim:.2f}] {artist_n}: {fav_title_n!r} ~ {lib_title!r}")
    if fuzzy_artist:
        print(f"+{len(fuzzy_artist)} fuzzy artist matches "
              f"(artist >= {args.fuzzy_artist_threshold} AND title >= {args.fuzzy_threshold}):")
        for (song_id, fav_artist_n, fav_title_n, lib_artist_n, lib_title,
             a_sim, t_sim) in sorted(fuzzy_artist, key=lambda x: (x[6], x[5]), reverse=True):
            fav = fav_orig.get((fav_artist_n, fav_title_n), (fav_artist_n, fav_title_n))
            print(f"   [artist_sim={a_sim:.2f} title_sim={t_sim:.2f}] "
                  f"{fav[0] + ' - ' + fav[1]!r} ~ lib {lib_artist_n!r} / {lib_title!r}")
    print(f"{len(still_missing)} favorites genuinely not found locally.")

    if args.dry_run:
        print("Dry run: nothing starred.")
        return

    # Star exact + fuzzy-title + fuzzy-artist, de-duplicated by song id.
    seen = set()
    to_star = []
    for song_id in (matched + [r[0] for r in fuzzy_title] + [r[0] for r in fuzzy_artist]):
        if song_id not in seen:
            seen.add(song_id)
            to_star.append(song_id)
    for i in range(0, len(to_star), STAR_CHUNK):
        chunk = to_star[i:i + STAR_CHUNK]
        params = auth_params(username, password, client_name, api_version)
        subsonic_get(base_url, "star", params, extra=[("id", sid) for sid in chunk])
    print(f"{len(to_star)} tracks marked as starred "
          f"({len(matched)} exact + {len(fuzzy_title)} fuzzy-title "
          f"+ {len(fuzzy_artist)} fuzzy-artist).")


if __name__ == "__main__":
    main()
