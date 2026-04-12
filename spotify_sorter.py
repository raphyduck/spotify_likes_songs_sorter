__version__ = "1.4.1"

import os
import sys
import time
import configparser
from datetime import datetime

import pandas as pd
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from scipy.sparse.csgraph import minimum_spanning_tree
from urllib.parse import urlparse, parse_qs
from tqdm import tqdm
from genre_helpers import (
    clean_album_name,
    get_discogs_album_info,
    get_itunes_album_info,
    get_lastfm_album_info,
    get_musicbrainz_album_info,
    get_lastfm_track_info,
    get_wikipedia_album_info,
    get_spotify_album_info,
    get_spotify_artist_genres,
    get_spotify_track_artist_genres,
    normalize_and_sort_genres)

# -----------------------------
#  Load credentials from settings.ini
# -----------------------------
config = configparser.ConfigParser()
config.read("settings.ini")

CLIENT_ID       = config["SPOTIFY"]["CLIENT_ID"]
CLIENT_SECRET   = config["SPOTIFY"]["CLIENT_SECRET"]
REDIRECT_URI    = config["SPOTIFY"]["REDIRECT_URI"]
EXPECTED_REDIRECT_URI = "http://127.0.0.1:8080/"
if REDIRECT_URI.rstrip("/") + "/" != EXPECTED_REDIRECT_URI:
    print(
        "ERROR: The console authorization flow requires the redirect URI to be set to"
        f" {EXPECTED_REDIRECT_URI}. Please update settings.ini and your Spotify"
        " Developer Dashboard to match."
    )
    sys.exit(1)
SCOPES           = [
"user-library-read",
"user-read-private",
"playlist-read-private",
"playlist-modify-private"
]
SCOPE = " ".join(SCOPES)
DISCOGS_API_KEY = config["DISCOGS"]["API_KEY"]
LASTFM_API_KEY  = config["LASTFM"]["API_KEY"]
GOOGLE_API_KEY  = config.get("GOOGLE_CSE", "API_KEY", fallback=None)
CSE_ID           = config.get("GOOGLE_CSE", "CSE_ID", fallback=None)
SEGMENTATION_STRENGTH = float(config.get("CLUSTERING", "segmentation_strength", fallback="0.6"))
MAX_CLUSTERS          = int(config.get("CLUSTERING", "max_clusters", fallback="10"))

CACHE_PATH = os.path.join(os.path.expanduser("~"), ".spotify_cache")

# -----------------------------
#  Authenticate with Spotify
# -----------------------------
print("\n🔄 Authenticating with Spotify...")
# --- auth_console.py style helper, à coller dans spotify_sorter.py ---

def get_spotify_client_console(scope: str,
                               client_id: str,
                               client_secret: str,
                               cache_path: str = ".cache") -> spotipy.Spotify:
    """
    Auth console-only (pas de serveur local, pas de port). 
    Nécessite que 'http://127.0.0.1:8080/' soit ajouté dans les Redirect URIs du dashboard Spotify.
    """
    oauth = SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=REDIRECT_URI,
        scope=scope,
        open_browser=False,                  # n’essaie pas d’ouvrir un navigateur
        cache_path=cache_path,
    )

    # 1) Tente d’utiliser le token en cache si existant
    token_info = oauth.validate_token(oauth.cache_handler.get_cached_token())
    if token_info:
        return spotipy.Spotify(auth_manager=oauth)

    # 2) Sinon, on lance un flow manuel
    auth_url = oauth.get_authorize_url()
    redirect_in_url = parse_qs(urlparse(auth_url).query).get("redirect_uri", [""])[0]
    if redirect_in_url.rstrip("/") + "/" != EXPECTED_REDIRECT_URI:
        print(
            "ERROR: Generated authorize URL does not match the expected redirect"
            f" URI ({EXPECTED_REDIRECT_URI}), which can lead to INVALID_CLIENT."
            " Confirm your settings.ini and Spotify app redirect URI both use this"
            " exact value."
        )
        sys.exit(1)
    print("\n=== Spotify OAuth (mode console) ===")
    print("1) Ouvre cette URL dans un navigateur (copie/colle) :\n")
    print(auth_url)
    print("\n2) Connecte-toi, autorise l’app, puis COPIE/COLLE ICI l’URL complète de redirection (celle qui commence par http://127.0.0.1:8080/ ...):\n")
    try:
        redirected_url = input("> URL de redirection: ").strip()
    except EOFError:
        print("Entrée manquante. Relance le script et colle l’URL de redirection.", file=sys.stderr)
        sys.exit(1)

    # 3) Extraction du ?code=...
    parsed = urlparse(redirected_url)
    code_list = parse_qs(parsed.query).get("code")
    if not code_list:
        print("Aucun 'code' trouvé dans l’URL. Vérifie que tu as bien collé l’URL complète.", file=sys.stderr)
        sys.exit(1)
    code = code_list[0]

    # 4) Échange code -> access_token
    token_info = oauth.get_access_token(code, as_dict=True)
    if not token_info or "access_token" not in token_info:
        print("Impossible d’obtenir un access_token.", file=sys.stderr)
        sys.exit(1)

    return spotipy.Spotify(auth_manager=oauth)

sp = get_spotify_client_console(
    scope=SCOPE,
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    cache_path=".cache-spotify-sorter"
)
print("✅ Authentication successful!\n")

# -----------------------------
#  Fetch all liked songs from Spotify
# -----------------------------
def map_track_to_row(track):
    if not track:
        return None
    artists = track.get("artists") or []
    album = track.get("album") or {}
    return {
        "Song": track.get("name") or "Unknown Song",
        "Artist": artists[0].get("name") if artists else "Unknown Artist",
        "Album": album.get("name") or "Unknown Album",
        "Album ID": album.get("id"),
        "Track Number": track.get("track_number"),
        "Disc Number":  track.get("disc_number"),
        "Spotify Track ID": track.get("id"),
        "Spotify URI": track.get("uri"),
        "Is Local": bool(track.get("is_local", False)),
    }

def print_local_tracks_log(rows, source_label):
    local_rows = [r for r in rows if r.get("Is Local")]
    if not local_rows:
        return
    print(f"📁 Local tracks found in {source_label}: {len(local_rows)}")
    for row in local_rows[:10]:
        print(f"   • {row['Artist']} — {row['Song']} ({row['Album']})")
    if len(local_rows) > 10:
        print(f"   … and {len(local_rows) - 10} more local tracks.")
    print()

def track_dedupe_key(row):
    return (
        row.get("Spotify URI")
        or row.get("Spotify Track ID")
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
    print("  [1] Liked songs")
    print("  [2] Playlist(s)")
    print("  [3] Liked songs + one playlist")
    while True:
        choice = input("> Choice (1/2/3): ").strip()
        if choice in {"1", "2", "3"}:
            return choice
        print("Invalid choice. Please enter 1, 2, or 3.")

def get_liked_songs():
    rows = []
    results = sp.current_user_saved_tracks(limit=50)
    total = results.get("total", 0)
    print("🎵 Fetching liked songs from Spotify...")

    with tqdm(total=total, desc="Liked songs", unit="track") as pbar:
        while results:
            for item in results["items"]:
                row = map_track_to_row(item.get("track"))
                if row:
                    rows.append(row)
            pbar.update(len(results.get("items", [])))
            results = sp.next(results) if results.get("next") else None
            time.sleep(0.5)

    print(f"🎉 Retrieved {len(rows)} songs!\n")
    print_local_tracks_log(rows, "liked songs")
    return rows

def get_user_playlists():
    playlists = []
    results = sp.current_user_playlists(limit=50)
    while results:
        playlists.extend(results.get("items", []))
        results = sp.next(results) if results.get("next") else None
        time.sleep(0.2)
    return playlists

def choose_playlists(playlists):
    if not playlists:
        print("No playlists found on your account.")
        sys.exit(1)

    print("\nAvailable playlists:")
    for idx, playlist in enumerate(playlists, start=1):
        print(f"  [{idx}] {playlist.get('name', 'Untitled')} ({playlist.get('tracks', {}).get('total', 0)} tracks)")

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
    total = sum((p.get("tracks") or {}).get("total", 0) for p in selected_playlists)
    with tqdm(total=total, desc="Playlist tracks", unit="track") as pbar:
        for playlist in selected_playlists:
            results = sp.playlist_items(playlist["id"], limit=100)
            while results:
                items = results.get("items", [])
                for item in items:
                    row = map_track_to_row(item.get("track"))
                    if not row:
                        continue
                    rows.append(row)
                pbar.update(len(items))
                results = sp.next(results) if results.get("next") else None
                time.sleep(0.2)

    rows = dedupe_rows(rows)
    print(f"🎉 Retrieved {len(rows)} unique songs from selected playlists!\n")
    print_local_tracks_log(rows, "selected playlists")
    return rows

# -----------------------------
#  Determine best genre using shared helpers
# -----------------------------
album_genre_cache = {}

def get_best_genre(song_name, artist_name, album_name, album_id, track_id):
    cache_key = (album_id or album_name, artist_name)  # album id+artist avoids same-title clashes
    if cache_key in album_genre_cache:
        return album_genre_cache[cache_key]

    clean_name = clean_album_name(album_name)
    providers = [
        ("Discogs", lambda: get_discogs_album_info(clean_name, artist_name, DISCOGS_API_KEY)),
        *((("Spotify Album", lambda: get_spotify_album_info(sp, album_id)),) if album_id else ()),
        *((("Spotify Track Artist", lambda: get_spotify_track_artist_genres(sp, track_id)),) if track_id else ()),
        ("LastFM Album", lambda: get_lastfm_album_info(clean_name, artist_name, LASTFM_API_KEY)),
        ("MusicBrainz", lambda: get_musicbrainz_album_info(clean_name, artist_name)),
        ("LastFM Track", lambda: get_lastfm_track_info(song_name, artist_name, LASTFM_API_KEY)),
        ("Spotify Artist", lambda: get_spotify_artist_genres(sp, artist_name)),
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
    source_slug = "liked_songs"
    source_label = "Liked songs"
    songs_data = get_liked_songs()
elif source_choice == "2":
    source_slug = "selected_playlists"
    playlists = get_user_playlists()
    selected_playlists = choose_playlists(playlists)
    source_label = "Playlists: " + ", ".join(p.get("name", "Untitled") for p in selected_playlists)
    songs_data = get_playlist_tracks(selected_playlists)
else:
    source_slug = "liked_plus_playlist"
    playlists = get_user_playlists()
    selected_playlist = choose_one_playlist(playlists)[0]
    source_label = f"Liked songs + {selected_playlist.get('name', 'Untitled')}"
    songs_data = dedupe_rows(get_liked_songs() + get_playlist_tracks([selected_playlist]))
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
        row.get("Spotify Track ID"),
    )
    album_genres.append(genres)
    album_genre_sources.append(source)
df["Album Genre"] = album_genres
df["source"] = album_genre_sources

# Unique identifier - use Album ID when available, fallback for local tracks
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
current_user = sp.current_user()
user_id = current_user["id"]

current_date = datetime.today().strftime('%Y-%m-%d')
playlist_name = f"liked songs sorted {current_date}"
playlist_description = f"Playlist created by Spotify Sorter from {source_label.lower()} using album genre similarity."

playlist = sp.user_playlist_create(user=user_id, name=playlist_name, public=False, description=playlist_description)
playlist_id = playlist["id"]
print(f"\n🎯 Created playlist: {playlist_name} (ID: {playlist_id})")

track_uris = [
    f"spotify:track:{tid}"
    for tid in final_df["Spotify Track ID"]
    if isinstance(tid, str) and tid
]
local_count = int(final_df["Is Local"].sum()) if "Is Local" in final_df else 0
chunks = [track_uris[pos:pos+100] for pos in range(0, len(track_uris), 100)]
for chunk in tqdm(chunks, desc=f"Uploading {playlist_name}", unit="chunk"):
    sp.playlist_add_items(playlist_id, chunk)
    time.sleep(0.5)

csv_filename = f"{source_slug}_sorted_{current_date}.csv"
final_df.to_csv(csv_filename, index=False)
print(f"\n📁 Sorted songs saved to CSV: {csv_filename}")
if local_count:
    print(
        f"\n⚠️ {local_count} local track(s) were kept in the CSV/sorting output "
        "but could not be added to the playlist through the Spotify Web API."
    )
print(f"\n✅ Playlist '{playlist_name}' created successfully with {len(track_uris)} tracks!")
