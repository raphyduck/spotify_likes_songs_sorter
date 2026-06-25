"""Service-agnostic sorting pipeline shared by every backend.

Given an authenticated :class:`~backends.Backend`, this module fetches the
chosen source, enriches each album with genre metadata, clusters albums by
genre similarity (MST cut + silhouette search + greedy chaining), then creates
the ordered playlist and writes a CSV export.
"""

import sys
import time
from datetime import datetime

import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.csgraph import minimum_spanning_tree
from tqdm import tqdm

from genre_helpers import clean_album_name, normalize_and_sort_genres


# -----------------------------
#  Source selection helpers
# -----------------------------
def _track_dedupe_key(row, backend):
    return (
        row.get(backend.track_id_col)
        or f"{row.get('Song')}|{row.get('Artist')}|{row.get('Album')}|"
           f"{row.get('Track Number')}|{row.get('Disc Number')}"
    )


def dedupe_rows(rows, backend):
    seen = set()
    deduped = []
    for row in rows:
        key = _track_dedupe_key(row, backend)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _print_local_tracks_log(rows, source_label):
    local_rows = [r for r in rows if r.get("Is Local")]
    if not local_rows:
        return
    print(f"📁 Local tracks found in {source_label}: {len(local_rows)}")
    for row in local_rows[:10]:
        print(f"   • {row['Artist']} — {row['Song']} ({row['Album']})")
    if len(local_rows) > 10:
        print(f"   … and {len(local_rows) - 10} more local tracks.")
    print()


def _select_input_source(backend):
    print(f"\nSelect source for sorting ({backend.display_name}):")
    print(f"  [1] {backend.liked_label}")
    print("  [2] Playlist(s)")
    print(f"  [3] {backend.liked_label} + one playlist")
    while True:
        choice = input("> Choice (1/2/3): ").strip()
        if choice in {"1", "2", "3"}:
            return choice
        print("Invalid choice. Please enter 1, 2, or 3.")


def _choose_playlists(backend, playlists):
    if not playlists:
        print("No playlists found on your account.")
        sys.exit(1)
    print("\nAvailable playlists:")
    for idx, playlist in enumerate(playlists, start=1):
        name, total = backend.playlist_display(playlist)
        print(f"  [{idx}] {name} ({total} tracks)")
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


def _collect_source(backend):
    """Run the interactive source menu and return ``(slug, label, rows)``."""
    choice = _select_input_source(backend)
    if choice == "1":
        rows = backend.get_liked_songs()
        if backend.supports_local:
            _print_local_tracks_log(rows, backend.liked_label.lower())
        return backend.liked_slug, backend.liked_label, dedupe_rows(rows, backend)

    if choice == "2":
        playlists = backend.get_user_playlists()
        selected = _choose_playlists(backend, playlists)
        rows = dedupe_rows(backend.get_playlist_tracks(selected), backend)
        if backend.supports_local:
            _print_local_tracks_log(rows, "selected playlists")
        print(f"🎉 Retrieved {len(rows)} unique songs from selected playlists!\n")
        label = "Playlists: " + ", ".join(backend.playlist_display(p)[0] for p in selected)
        return "selected_playlists", label, rows

    playlists = backend.get_user_playlists()
    selected = _choose_playlists(backend, playlists)[:1]
    selected_playlist = selected[0]
    liked = backend.get_liked_songs()
    if backend.supports_local:
        _print_local_tracks_log(liked, backend.liked_label.lower())
    playlist_rows = backend.get_playlist_tracks([selected_playlist])
    rows = dedupe_rows(liked + playlist_rows, backend)
    print(f"🎉 Combined source contains {len(rows)} unique songs.\n")
    label = f"{backend.liked_label} + {backend.playlist_display(selected_playlist)[0]}"
    return f"{backend.liked_slug}_plus_playlist", label, rows


# -----------------------------
#  Genre enrichment
# -----------------------------
def _make_genre_resolver(backend, config):
    cache = {}

    def get_best_genre(song_name, artist_name, album_name, album_id, track_id):
        cache_key = (album_id or album_name, artist_name)
        if cache_key in cache:
            return cache[cache_key]
        clean_name = clean_album_name(album_name or "")
        providers = backend.get_genre_providers(
            song_name, artist_name, album_name, clean_name, album_id, track_id, config
        )
        for source, lookup in providers:
            genres = lookup()
            if genres:
                result = (genres, source)
                cache[cache_key] = result
                return result
        return [], "None"

    return get_best_genre


# -----------------------------
#  Clustering + ordering
# -----------------------------
def _order_albums(df, segmentation_strength, max_clusters):
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
    strength = float(np.clip(segmentation_strength, 0.0, 1.0))

    def _components_for_k(k):
        g = G.copy()
        for u, v, _ in edges[: k - 1]:
            g.remove_edge(u, v)
        return list(nx.connected_components(g))

    labels_best, best_score = None, -1.0
    max_k = min(max_clusters, len(edges) + 1)
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
    final_comps = (
        [set(c) for c in large_comps]
        or [set(c) for c in components]
        or [set(range(len(unique_albums_df)))]
    )
    for small in small_comps:
        for idx in small:
            best_i = max(
                range(len(final_comps)),
                key=lambda i: np.mean([sim[idx, j] for j in final_comps[i]]),
            )
            final_comps[best_i].add(idx)

    reset_factor = (
        float(np.quantile(sim[np.triu_indices_from(sim, k=1)], 0.35 + 0.3 * strength))
        if len(sim) > 1 else 0.5
    )

    def greedy_chain(albums, sim_df, threshold_ratio):
        chain, prev_sim = [], 1.0
        sub = sim_df.loc[albums, albums]
        start = sub.mean(axis=1).idxmax()
        chain.append(start)
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

    album_similarity = pd.DataFrame(
        sim, index=unique_albums_df["Unique Album"], columns=unique_albums_df["Unique Album"]
    )
    sorted_albums = []
    for comp in final_comps:
        names = [unique_albums_df["Unique Album"].iloc[i] for i in comp]
        sorted_albums.extend(greedy_chain(names, album_similarity, reset_factor))

    unique_albums_df["Sort Order"] = unique_albums_df["Unique Album"].apply(sorted_albums.index)
    unique_albums_df = unique_albums_df.sort_values("Sort Order")
    return unique_albums_df[["Unique Album", "Sort Order", "Sorted Genres"]]


# -----------------------------
#  Top-level entry point
# -----------------------------
def run(backend, config):
    """Run the full pipeline for an authenticated backend."""
    source_slug, source_label, songs_data = _collect_source(backend)

    df = pd.DataFrame(songs_data)
    if df.empty:
        print("No tracks found for the selected source. Nothing to sort.")
        sys.exit(0)

    print("🔎 Fetching genres for songs (with shared helpers)...")
    get_best_genre = _make_genre_resolver(backend, config)
    album_genres, album_genre_sources = [], []
    for row in tqdm(df.to_dict("records"), total=len(df), desc="Genres", unit="track"):
        genres, source = get_best_genre(
            row.get("Song"),
            row.get("Artist"),
            row.get("Album"),
            row.get("Album ID"),
            row.get(backend.track_id_col),
        )
        album_genres.append(genres)
        album_genre_sources.append(source)
    df["Album Genre"] = album_genres
    df["source"] = album_genre_sources

    # Unique identifier - use Album ID when available, fallback otherwise
    df["Unique Album"] = df["Album ID"].fillna(df["Album"] + " - " + df["Artist"])

    segmentation_strength = float(config.get("CLUSTERING", "segmentation_strength", fallback="0.6"))
    max_clusters = int(config.get("CLUSTERING", "max_clusters", fallback="10"))
    ordering = _order_albums(df, segmentation_strength, max_clusters)

    final_df = (
        pd.merge(df, ordering, on="Unique Album", how="left")
        .sort_values(["Sort Order", "Disc Number", "Track Number"])
    )
    final_df["Album Genre"] = final_df["Sorted Genres"]
    final_df.drop(columns=["Sorted Genres"], inplace=True)

    # -----------------------------
    #  Create playlist & save CSV
    # -----------------------------
    current_date = datetime.today().strftime('%Y-%m-%d')
    playlist_name = f"liked songs sorted {current_date}"
    playlist_description = (
        f"Playlist created by {backend.display_name} Sorter from "
        f"{source_label.lower()} using album genre similarity."
    )

    handle = backend.create_playlist(playlist_name, playlist_description)
    print(f"\n🎯 Created playlist: {playlist_name}")

    uploaded, local_count = backend.add_tracks(handle, final_df.to_dict("records"))

    csv_filename = f"{backend.key}_{source_slug}_sorted_{current_date}.csv"
    final_df.to_csv(csv_filename, index=False)
    print(f"\n📁 Sorted songs saved to CSV: {csv_filename}")
    if local_count:
        print(
            f"\n⚠️ {local_count} local track(s) were kept in the CSV/sorting output "
            f"but could not be added to the playlist through the {backend.display_name} API."
        )
    print(f"\n✅ Playlist '{playlist_name}' created successfully with {uploaded} tracks!")
