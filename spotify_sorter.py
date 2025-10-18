__version__ = "1.4.1"

import os
import time
import configparser
from datetime import datetime

import pandas as pd
import numpy as np
import requests
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from scipy.sparse.csgraph import minimum_spanning_tree
from urllib.parse import urlparse, parse_qs
from genre_helpers import (
    clean_album_name,
    get_discogs_album_info,
    get_lastfm_album_info,
    get_musicbrainz_album_info,
    get_lastfm_track_info,
    get_wikipedia_album_info,
    get_spotify_album_info,
    get_spotify_artist_genres,
    normalize_and_sort_genres)

# -----------------------------
#  Load credentials from settings.ini
# -----------------------------
config = configparser.ConfigParser()
config.read("settings.ini")

CLIENT_ID       = config["SPOTIFY"]["CLIENT_ID"]
CLIENT_SECRET   = config["SPOTIFY"]["CLIENT_SECRET"]
REDIRECT_URI    = config["SPOTIFY"]["REDIRECT_URI"]
SCOPES           = [
"user-library-read",
"user-read-private",
"playlist-modify-private"
]
SCOPE = " ".join(SCOPES)
DISCOGS_API_KEY = config["DISCOGS"]["API_KEY"]
LASTFM_API_KEY  = config["LASTFM"]["API_KEY"]
GOOGLE_API_KEY  = config.get("GOOGLE_CSE", "API_KEY", fallback=None)
CSE_ID           = config.get("GOOGLE_CSE", "CSE_ID", fallback=None)

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
    Nécessite que 'http://localhost/' soit ajouté dans les Redirect URIs du dashboard Spotify.
    """
    oauth = SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri="http://localhost/",   # IMPORTANT: ajouter exactement cette URI dans le dashboard
        scope=scope,
        open_browser=False,                  # n’essaie pas d’ouvrir un navigateur
        cache_path=cache_path,
    )

    # 1) Tente d’utiliser le token en cache si existant
    token_info = oauth.get_cached_token()
    if token_info and token_info.get("access_token"):
        return spotipy.Spotify(auth=token_info["access_token"])

    # 2) Sinon, on lance un flow manuel
    auth_url = oauth.get_authorize_url()
    print("\n=== Spotify OAuth (mode console) ===")
    print("1) Ouvre cette URL dans un navigateur (copie/colle) :\n")
    print(auth_url)
    print("\n2) Connecte-toi, autorise l’app, puis COPIE/COLLE ICI l’URL complète de redirection (celle qui commence par http://localhost/ ...):\n")
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

    return spotipy.Spotify(auth=token_info["access_token"])

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
def get_liked_songs():
    liked_songs = []
    results = sp.current_user_saved_tracks(limit=50)
    print("🎵 Fetching liked songs from Spotify...")

    while results:
        for item in results["items"]:
            track = item["track"]
            liked_songs.append({
                "Song": track["name"],
                "Artist": track["artists"][0]["name"],
                "Album": track["album"]["name"],
                "Album ID": track["album"]["id"],
                "Track Number": track["track_number"],
                "Disc Number":  track["disc_number"],
                "Spotify Track ID": track["id"]
            })
        results = sp.next(results) if results["next"] else None
        time.sleep(0.5)

    print(f"🎉 Retrieved {len(liked_songs)} songs!\n")
    return liked_songs

# -----------------------------
#  Determine best genre using shared helpers
# -----------------------------
album_genre_cache = {}

def get_best_genre(song_name, artist_name, album_name, album_id):
    if album_name in album_genre_cache:
        return album_genre_cache[album_name]

    clean_name = clean_album_name(album_name)
    # 1) Discogs
    g = get_discogs_album_info(clean_name, artist_name, DISCOGS_API_KEY)
    if g:
        album_genre_cache[album_name] = g
        return g
    # 2) Last.fm album
    g = get_lastfm_album_info(clean_name, artist_name, LASTFM_API_KEY)
    if g:
        album_genre_cache[album_name] = g
        return g
    # 3) MusicBrainz
    g = get_musicbrainz_album_info(clean_name, artist_name)
    if g:
        album_genre_cache[album_name] = g
        return g
    # 4) Last.fm track
    g = get_lastfm_track_info(song_name, artist_name, LASTFM_API_KEY)
    if g:
        album_genre_cache[album_name] = g
        return g
    # 5) Spotify album
    g = get_spotify_album_info(sp, album_id)
    if g:
        album_genre_cache[album_name] = g
        return g
    # 6) Wikipedia
    g = get_wikipedia_album_info(clean_name, artist_name)
    if g:
        album_genre_cache[album_name] = g
        return g
    # 7) Spotify artist
    g = get_spotify_artist_genres(sp, artist_name)
    if g:
        album_genre_cache[album_name] = g
        return g
    return []

# -----------------------------
# Main Processing
# -----------------------------
songs_data = get_liked_songs()
df = pd.DataFrame(songs_data)

print("🔎 Fetching genres for songs (with shared helpers)...")
df["Album Genre"] = df.apply(
    lambda row: get_best_genre(
        row["Song"], row["Artist"], row["Album"], row["Album ID"]
    ), axis=1
)

# Unique identifier
df["Unique Album"] = df["Album"] + " - " + df["Artist"]

# -----------------------------
#  MST‑based clustering + greedy chaining
# -----------------------------
unique_albums_df = df.drop_duplicates(subset=["Unique Album"]).copy()
raw_lists = [g if isinstance(g, list) else [] for g in unique_albums_df["Album Genre"]]
# Normalize and sort genre tags by frequency
genre_sorted = normalize_and_sort_genres(raw_lists)
# Store sorted genre strings
unique_albums_df["Sorted Genres"] = [", ".join(sub) for sub in genre_sorted]
# One‑hot encode
genre_onehot = MultiLabelBinarizer().fit_transform(genre_sorted)

sim = cosine_similarity(genre_onehot)
dist = 1.0 - sim
mst = minimum_spanning_tree(dist).toarray()
G = nx.from_numpy_array(mst)

k = 5
edges = sorted(G.edges(data=True), key=lambda x: x[2]["weight"], reverse=True)
for u, v, w in edges[: k - 1]: G.remove_edge(u, v)
components = list(nx.connected_components(G))

# Reassign tiny outliers
min_size = 3
large_comps = [c for c in components if len(c) >= min_size]
small_comps = [c for c in components if len(c) < min_size]
final_comps = [set(c) for c in large_comps]
for small in small_comps:
    for idx in small:
        best_i = max(range(len(final_comps)), key=lambda i: np.mean([sim[idx, j] for j in final_comps[i]]))
        final_comps[best_i].add(idx)

# Greedy chaining
def greedy_chain(albums, sim_df, threshold_ratio=0.5):
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
    sorted_albums.extend(greedy_chain(names, album_similarity))

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
playlist_name = f"Liked songs sorted {current_date}"
playlist_description = "Playlist created by Spotify Sorter using album genre similarity."

playlist = sp.user_playlist_create(user=user_id, name=playlist_name, public=False, description=playlist_description)
playlist_id = playlist["id"]
print(f"\n🎯 Created playlist: {playlist_name} (ID: {playlist_id})")

track_uris = ["spotify:track:" + tid for tid in final_df["Spotify Track ID"]]
for chunk in (track_uris[pos:pos+100] for pos in range(0, len(track_uris), 100)):
    sp.playlist_add_items(playlist_id, chunk)
    time.sleep(0.5)

csv_filename = f"liked_songs_sorted_{current_date}.csv"
final_df.to_csv(csv_filename, index=False)
print(f"\n📁 Sorted songs saved to CSV: {csv_filename}")
print(f"\n✅ Playlist '{playlist_name}' created successfully with {len(track_uris)} tracks!")
