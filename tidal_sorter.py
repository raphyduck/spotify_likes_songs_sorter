__version__ = "2.0.0"

import os
import sys
import time
import configparser
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import tidalapi
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from scipy.sparse.csgraph import minimum_spanning_tree
from tqdm import tqdm
from genre_helpers import (
    clean_album_name,
    get_discogs_album_info,
    get_itunes_album_info,
    get_lastfm_album_info,
    get_musicbrainz_album_info,
    get_lastfm_track_info,
    get_wikipedia_album_info,
    normalize_and_sort_genres)

# -----------------------------
#  Load credentials from settings.ini
# -----------------------------
config = configparser.ConfigParser()
config.read("settings.ini")

DISCOGS_API_KEY = config["DISCOGS"]["API_KEY"]
LASTFM_API_KEY  = config["LASTFM"]["API_KEY"]
SEGMENTATION_STRENGTH = float(config.get("CLUSTERING", "segmentation_strength", fallback="0.6"))
MAX_CLUSTERS          = int(config.get("CLUSTERING", "max_clusters", fallback="10"))

# Tidal's OAuth device flow does not require a developer client id/secret; it
# uses the built-in TV/device client. The resulting session is cached here so
# you only have to authorize once.
SESSION_FILE = Path(
    config.get("TIDAL", "session_file", fallback=None)
    or os.path.join(os.path.expanduser("~"), ".tidal_sorter_session.json")
)

# -----------------------------
#  Authenticate with Tidal
# -----------------------------
print("\n🔄 Authenticating with Tidal...")

def get_tidal_session(session_file: Path) -> tidalapi.Session:
    """
    Console-only OAuth using Tidal's device authorization flow.

    `login_session_file` reuses a cached session when present (refreshing the
    access token automatically), and otherwise prints a link.tidal.com URL for
    you to open and authorize before polling for completion. No local web
    server or redirect URI is required.
    """
    session = tidalapi.Session()
    print("\n=== Tidal OAuth (console mode) ===")
    print("If no cached session is found, open the link printed below in a")
    print("browser, log in, and authorize the device. Authorization is then")
    print("detected automatically.\n")
    session.login_session_file(session_file)
    if not session.check_login():
        print("❌ Tidal authentication failed. Please retry.", file=sys.stderr)
        sys.exit(1)
    return session

sp = get_tidal_session(SESSION_FILE)
print("✅ Authentication successful!\n")

# -----------------------------
#  Map Tidal tracks to flat rows
# -----------------------------
def map_track_to_row(track):
    if not track:
        return None
    try:
        if getattr(track, "artist", None) and track.artist.name:
            artist_name = track.artist.name
        elif getattr(track, "artists", None):
            artist_name = track.artists[0].name
        else:
            artist_name = "Unknown Artist"
    except Exception:
        artist_name = "Unknown Artist"

    album = getattr(track, "album", None)
    album_name = getattr(album, "name", None) or "Unknown Album"
    album_id = getattr(album, "id", None)
    return {
        "Song": getattr(track, "name", None) or "Unknown Song",
        "Artist": artist_name,
        "Album": album_name,
        "Album ID": str(album_id) if album_id is not None else None,
        "Track Number": getattr(track, "track_num", None),
        "Disc Number":  getattr(track, "volume_num", None),
        "Tidal Track ID": str(track.id) if getattr(track, "id", None) is not None else None,
    }

def track_dedupe_key(row):
    return (
        row.get("Tidal Track ID")
        or f"{row.get('Song')}|{row.get('Artist')}|{row.get('Album')}|{row.get('Track Number')}|{row.get('Disc Number')}"
    )

def dedupe_rows(rows):
    seen = set()
    deduped = []
    for row in rows:
        key = track_dedupe_key(row)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped

def select_input_source():
    print("Select source for sorting:")
    print("  [1] Favorite tracks")
    print("  [2] Playlist(s)")
    print("  [3] Favorite tracks + one playlist")
    while True:
        choice = input("> Choice (1/2/3): ").strip()
        if choice in {"1", "2", "3"}:
            return choice
        print("Invalid choice. Please enter 1, 2, or 3.")

# -----------------------------
#  Fetch tracks from Tidal
# -----------------------------
PAGE_LIMIT = 100

def get_favorite_tracks():
    rows = []
    favorites = sp.user.favorites
    try:
        total = favorites.get_tracks_count()
    except Exception:
        total = 0
    print("🎵 Fetching favorite tracks from Tidal...")

    offset = 0
    with tqdm(total=total or None, desc="Favorite tracks", unit="track") as pbar:
        while True:
            batch = favorites.tracks(limit=PAGE_LIMIT, offset=offset)
            if not batch:
                break
            for track in batch:
                row = map_track_to_row(track)
                if row:
                    rows.append(row)
            pbar.update(len(batch))
            if len(batch) < PAGE_LIMIT:
                break
            offset += PAGE_LIMIT
            time.sleep(0.3)

    rows = dedupe_rows(rows)
    print(f"🎉 Retrieved {len(rows)} favorite tracks!\n")
    return rows

def get_user_playlists():
    return sp.user.playlists()

def _playlist_name(playlist):
    return getattr(playlist, "name", None) or "Untitled"

def _playlist_track_total(playlist):
    return getattr(playlist, "num_tracks", None) or 0

def choose_playlists(playlists):
    if not playlists:
        print("No playlists found on your account.")
        sys.exit(1)

    print("\nAvailable playlists:")
    for idx, playlist in enumerate(playlists, start=1):
        print(f"  [{idx}] {_playlist_name(playlist)} ({_playlist_track_total(playlist)} tracks)")

    while True:
        raw = input("> Select one or more playlists (e.g. 1,3,5): ").strip()
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if not parts or any(not p.isdigit() for p in parts):
            print("Invalid input. Enter playlist numbers separated by commas.")
            continue
        indexes = sorted(set(int(p) for p in parts))
        if indexes[0] < 1 or indexes[-1] > len(playlists):
            print("Invalid selection. One or more numbers are out of range.")
            continue
        return [playlists[i - 1] for i in indexes]

def choose_one_playlist(playlists):
    return choose_playlists(playlists)[:1]

def get_playlist_tracks(selected_playlists):
    rows = []
    print("🎵 Fetching tracks from selected playlist(s)...")
    total = sum(_playlist_track_total(p) for p in selected_playlists)
    with tqdm(total=total or None, desc="Playlist tracks", unit="track") as pbar:
        for playlist in selected_playlists:
            offset = 0
            while True:
                batch = playlist.tracks(limit=PAGE_LIMIT, offset=offset)
                if not batch:
                    break
                for track in batch:
                    row = map_track_to_row(track)
                    if not row:
                        continue
                    rows.append(row)
                pbar.update(len(batch))
                if len(batch) < PAGE_LIMIT:
                    break
                offset += PAGE_LIMIT
                time.sleep(0.2)

    rows = dedupe_rows(rows)
    print(f"🎉 Retrieved {len(rows)} unique songs from selected playlists!\n")
    return rows

# -----------------------------
#  Determine best genre using shared helpers
# -----------------------------
album_genre_cache = {}

def get_best_genre(song_name, artist_name, album_name, album_id=None):
    cache_key = (album_id or album_name, artist_name)  # album id+artist avoids same-title clashes
    if cache_key in album_genre_cache:
        return album_genre_cache[cache_key]

    clean_name = clean_album_name(album_name)
    providers = [
        ("Discogs", lambda: get_discogs_album_info(clean_name, artist_name, DISCOGS_API_KEY)),
        ("LastFM Album", lambda: get_lastfm_album_info(clean_name, artist_name, LASTFM_API_KEY)),
        ("MusicBrainz", lambda: get_musicbrainz_album_info(clean_name, artist_name)),
        ("LastFM Track", lambda: get_lastfm_track_info(song_name, artist_name, LASTFM_API_KEY)),
        ("Wikipedia", lambda: get_wikipedia_album_info(clean_name, artist_name)),
        ("iTunes", lambda: get_itunes_album_info(clean_name, artist_name)),
    ]
    for source, lookup in providers:
        genres = lookup()
        if genres:
            result = (genres, source)
            album_genre_cache[cache_key] = result
            return result
    return [], "None"

# -----------------------------
# Main Processing
# -----------------------------
source_choice = select_input_source()
if source_choice == "1":
    source_slug = "favorite_tracks"
    source_label = "Favorite tracks"
    songs_data = get_favorite_tracks()
elif source_choice == "2":
    source_slug = "selected_playlists"
    playlists = get_user_playlists()
    selected_playlists = choose_playlists(playlists)
    source_label = "Playlists: " + ", ".join(_playlist_name(p) for p in selected_playlists)
    songs_data = get_playlist_tracks(selected_playlists)
else:
    source_slug = "favorites_plus_playlist"
    playlists = get_user_playlists()
    selected_playlist = choose_one_playlist(playlists)[0]
    source_label = f"Favorite tracks + {_playlist_name(selected_playlist)}"
    songs_data = dedupe_rows(get_favorite_tracks() + get_playlist_tracks([selected_playlist]))
    print(f"🎉 Combined source contains {len(songs_data)} unique songs.\n")

df = pd.DataFrame(songs_data)
if df.empty:
    print("No tracks found for the selected source. Nothing to sort.")
    sys.exit(0)

print("🔎 Fetching genres for songs (with shared helpers)...")
album_genres = []
album_genre_sources = []
for row in tqdm(df.to_dict("records"), total=len(df), desc="Genres", unit="track"):
    genres, source = get_best_genre(
        row.get("Song"),
        row.get("Artist"),
        row.get("Album"),
        row.get("Album ID"),
    )
    album_genres.append(genres)
    album_genre_sources.append(source)
df["Album Genre"] = album_genres
df["source"] = album_genre_sources

# Unique identifier - use Album ID when available, fallback when missing
df["Unique Album"] = df["Album ID"].fillna(df["Album"] + " - " + df["Artist"])

# -----------------------------
#  MST‑based clustering + greedy chaining
# -----------------------------
unique_albums_df = df.drop_duplicates(subset=["Unique Album"]).copy()
raw_lists = [g if isinstance(g, list) else [] for g in unique_albums_df["Album Genre"]]
genre_sorted = normalize_and_sort_genres(raw_lists)
unique_albums_df["Sorted Genres"] = [", ".join(sub) for sub in genre_sorted]
genre_onehot = MultiLabelBinarizer().fit_transform(genre_sorted)

sim = cosine_similarity(genre_onehot)
dist = 1.0 - sim
mst = minimum_spanning_tree(dist).toarray()
G = nx.from_numpy_array(mst)

edges = sorted(G.edges(data=True), key=lambda x: x[2]["weight"], reverse=True)
weights = np.array([w["weight"] for *_, w in edges]) if edges else np.array([0.0])
strength = float(np.clip(SEGMENTATION_STRENGTH, 0.0, 1.0))

def _components_for_k(k):
    g = G.copy()
    for u, v, _ in edges[: k - 1]:
        g.remove_edge(u, v)
    return list(nx.connected_components(g))

labels_best, best_score = None, -1.0
max_k = min(MAX_CLUSTERS, len(edges) + 1)
if len(unique_albums_df) > 2 and max_k >= 2:
    for k in range(2, max_k + 1):
        comps = _components_for_k(k)
        labels = [-1] * len(unique_albums_df)
        for lbl, comp in enumerate(comps):
            for idx in comp:
                labels[idx] = lbl
        try:
            score = silhouette_score(dist, labels, metric="precomputed")
        except ValueError:
            continue
        if score > best_score:
            best_score, labels_best = score, labels

if labels_best is None:
    cutoff = np.quantile(weights, 0.55 + 0.35 * strength)
    g = G.copy()
    for u, v, w in edges:
        if w["weight"] >= cutoff:
            g.remove_edge(u, v)
    components = list(nx.connected_components(g))
else:
    clusters = {}
    for idx, lbl in enumerate(labels_best):
        clusters.setdefault(lbl, set()).add(idx)
    components = list(clusters.values())

min_size = 3
large_comps = [c for c in components if len(c) >= min_size]
small_comps = [c for c in components if len(c) < min_size]
final_comps = [set(c) for c in large_comps] or [set(c) for c in components] or [set(range(len(unique_albums_df)))]
for small in small_comps:
    for idx in small:
        best_i = max(range(len(final_comps)), key=lambda i: np.mean([sim[idx, j] for j in final_comps[i]]))
        final_comps[best_i].add(idx)

reset_factor = float(np.quantile(sim[np.triu_indices_from(sim, k=1)], 0.35 + 0.3 * strength)) if len(sim) > 1 else 0.5

def greedy_chain(albums, sim_df, threshold_ratio):
    chain, prev_sim = [], 1.0
    sub = sim_df.loc[albums, albums]
    start = sub.mean(axis=1).idxmax(); chain.append(start)
    remaining = set(albums) - {start}
    while remaining:
        last = chain[-1]
        sims = sim_df.loc[last, list(remaining)]
        best, val = sims.idxmax(), sims.max()
        if val < prev_sim * threshold_ratio:
            nxt = remaining.pop(); chain.append(nxt); prev_sim = 1.0
        else:
            chain.append(best); remaining.remove(best); prev_sim = val
    return chain

album_similarity = pd.DataFrame(sim, index=unique_albums_df["Unique Album"], columns=unique_albums_df["Unique Album"])
sorted_albums = []
for comp in final_comps:
    names = [unique_albums_df["Unique Album"].iloc[i] for i in comp]
    sorted_albums.extend(greedy_chain(names, album_similarity, reset_factor))

unique_albums_df["Sort Order"] = unique_albums_df["Unique Album"].apply(lambda x: sorted_albums.index(x))
unique_albums_df = unique_albums_df.sort_values("Sort Order")

# Merge back, including Sorted Genres
final_df = (
    pd.merge(
        df,
        unique_albums_df[["Unique Album", "Sort Order", "Sorted Genres"]],
        on="Unique Album", how="left"
    )
    .sort_values(["Sort Order", "Disc Number", "Track Number"])
)
# Overwrite Album Genre column
final_df["Album Genre"] = final_df["Sorted Genres"]
final_df.drop(columns=["Sorted Genres"], inplace=True)

# -----------------------------
# Create playlist & save CSV
# -----------------------------
current_date = datetime.today().strftime('%Y-%m-%d')
playlist_name = f"liked songs sorted {current_date}"
playlist_description = f"Playlist created by Tidal Sorter from {source_label.lower()} using album genre similarity."

playlist = sp.user.create_playlist(playlist_name, playlist_description)
playlist_id = getattr(playlist, "id", None)
print(f"\n🎯 Created playlist: {playlist_name} (ID: {playlist_id})")

track_ids = [
    str(tid)
    for tid in final_df["Tidal Track ID"]
    if isinstance(tid, str) and tid
]
chunks = [track_ids[pos:pos+100] for pos in range(0, len(track_ids), 100)]
for chunk in tqdm(chunks, desc=f"Uploading {playlist_name}", unit="chunk"):
    playlist.add(chunk)
    time.sleep(0.5)

csv_filename = f"{source_slug}_sorted_{current_date}.csv"
final_df.to_csv(csv_filename, index=False)
print(f"\n📁 Sorted songs saved to CSV: {csv_filename}")
print(f"\n✅ Playlist '{playlist_name}' created successfully with {len(track_ids)} tracks!")
