"""Service-agnostic sorting pipeline shared by every backend.

Given an authenticated :class:`~backends.Backend`, this module fetches the
chosen source, enriches each album with genre metadata, clusters albums by
genre similarity (MST cut + silhouette search + greedy chaining), then creates
the ordered playlist and writes a CSV export.
"""

import sys
import time
from collections import OrderedDict
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
from genre_cache import build_cache_from_config, make_key
from genre_normalization import (
    load_genre_roots,
    infer_root,
    display_root,
    genre_similarity_matrix,
    avg_adjacent_overlap,
    count_fragmented_roots,
)
from genre_overrides import load_overrides, lookup_override


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
def _make_genre_resolver(backend, config, cache, overrides=None):
    def get_best_genre(song_name, artist_name, album_name, album_id, track_id):
        # Manual overrides win over everything (providers and cache).
        override = lookup_override(overrides, artist_name, album_name)
        if override and override.get("tags"):
            return override["tags"], "Override"
        cache_key = make_key(album_id, album_name, artist_name)
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
        clean_name = clean_album_name(album_name or "")
        providers = backend.get_genre_providers(
            song_name, artist_name, album_name, clean_name, album_id, track_id, config
        )
        for source, lookup in providers:
            genres = lookup()
            if genres:
                cache.set(cache_key, genres, source)
                return genres, source
        # Cache the negative result too (short TTL) so it is retried before long.
        cache.set(cache_key, [], "None")
        return [], "None"

    return get_best_genre


# -----------------------------
#  Clustering + ordering
# -----------------------------
def _order_from_similarity(names, sim, segmentation_strength, max_clusters):
    """Cluster (MST + silhouette) and greedily chain albums into one order.

    ``names`` are the album identifiers aligned to the rows/cols of ``sim``;
    returns the album names in their final order.
    """
    n = len(names)
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
    if n > 2 and max_k >= 2:
        for k in range(2, max_k + 1):
            comps = _components_for_k(k)
            labels = [-1] * n
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
        or [set(range(n))]
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

    album_similarity = pd.DataFrame(sim, index=names, columns=names)
    sorted_albums = []
    for comp in final_comps:
        comp_names = [names[i] for i in comp]
        sorted_albums.extend(greedy_chain(comp_names, album_similarity, reset_factor))
    return sorted_albums


def _nearest_neighbor_order(labels, sim):
    """Greedy single chain over ``labels`` using similarity matrix ``sim``.

    Starts from the most central label (highest mean similarity) and repeatedly
    appends the most similar unused label. Deterministic (ties -> lower index).
    """
    n = len(labels)
    if n <= 1:
        return list(labels)
    remaining = set(range(n))
    start = int(np.argmax(sim.mean(axis=1)))
    order = [start]
    remaining.discard(start)
    while remaining:
        last = order[-1]
        best = max(remaining, key=lambda j: (sim[last, j], -j))
        order.append(best)
        remaining.discard(best)
    return [labels[i] for i in order]


def _two_level_order(names, M, sim_tags, tag_sets, roots,
                     segmentation_strength, max_clusters):
    """Order albums macro-by-root, micro-by-tags, with block orientation.

    1. Micro: within each root family, order albums by full-tag similarity using
       the existing chain/cluster logic.
    2. Macro: order the root families by the cosine similarity of their tag
       centroids (singleton/``unknown`` roots pushed to the tail).
    3. Stitch the per-root blocks in macro order, flipping each block to
       minimise the genre jump at the seam (max Jaccard across the boundary).
    """
    n = len(names)
    if n <= 1:
        return list(names)

    groups = OrderedDict()
    for i, root in enumerate(roots):
        groups.setdefault(root, []).append(i)
    root_labels = list(groups)

    # Macro ordering of root families via tag-centroid similarity.
    if M.shape[1] > 0:
        centroids = np.array([M[groups[r]].mean(axis=0) for r in root_labels])
        root_sim = cosine_similarity(centroids)
    else:
        root_sim = np.zeros((len(root_labels), len(root_labels)))
    main = [k for k, r in enumerate(root_labels) if len(groups[r]) >= 2 and r != "unknown"]
    tail = [k for k in range(len(root_labels)) if k not in main]
    if main:
        main_order = _nearest_neighbor_order(
            [root_labels[k] for k in main], root_sim[np.ix_(main, main)]
        )
    else:
        main_order = []
    tail_order = sorted((root_labels[k] for k in tail), key=lambda r: (-len(groups[r]), r))
    macro = main_order + tail_order

    # Micro ordering inside each root family (full-tag similarity).
    micro = {}
    for root in root_labels:
        idx = groups[root]
        if len(idx) == 1:
            micro[root] = [names[idx[0]]]
        else:
            sub_sim = sim_tags[np.ix_(idx, idx)]
            sub_names = [names[i] for i in idx]
            micro[root] = _order_from_similarity(
                sub_names, sub_sim, segmentation_strength, max_clusters
            )

    name_to_set = dict(zip(names, tag_sets))

    def jac(a, b):
        sa, sb = name_to_set[a], name_to_set[b]
        if not sa and not sb:
            return 0.0
        union = sa | sb
        return len(sa & sb) / len(union) if union else 0.0

    blocks = [micro[r] for r in macro if micro.get(r)]
    result = []
    for bi, block in enumerate(blocks):
        if not result:
            if bi + 1 < len(blocks):
                nxt = blocks[bi + 1]
                normal = max(jac(block[-1], nxt[0]), jac(block[-1], nxt[-1]))
                flipped = max(jac(block[0], nxt[0]), jac(block[0], nxt[-1]))
                if flipped > normal:
                    block = list(reversed(block))
            result.extend(block)
            continue
        last = result[-1]
        if jac(last, block[-1]) > jac(last, block[0]):
            block = list(reversed(block))
        result.extend(block)
    return result


def _ordering_metric(order, tag_sets_by_name, root_by_name):
    ordered_tag_sets = [tag_sets_by_name[name] for name in order]
    ordered_roots = [root_by_name[name] for name in order]
    return {
        "overlap": avg_adjacent_overlap(ordered_tag_sets),
        "fragmented": count_fragmented_roots(ordered_roots),
    }


def _album_root(genre_list, artist, album, rules, overrides):
    override = lookup_override(overrides, artist, album)
    if override and override.get("root"):
        return override["root"]
    return infer_root(genre_list, rules)


ORDERING_MODES = ("legacy", "roots", "two_level")


def _order_albums(df, segmentation_strength, max_clusters, root_weight, rules,
                  overrides=None, ordering_mode="two_level"):
    unique_albums_df = df.drop_duplicates(subset=["Unique Album"]).copy()
    raw_lists = [g if isinstance(g, list) else [] for g in unique_albums_df["Album Genre"]]
    genre_sorted = normalize_and_sort_genres(raw_lists)
    unique_albums_df["Sorted Genres"] = [", ".join(sub) for sub in genre_sorted]

    names = list(unique_albums_df["Unique Album"])
    artists = list(unique_albums_df["Artist"])
    albums = list(unique_albums_df["Album"])
    tag_sets = [{t.strip().lower() for t in sub if t.strip()} for sub in genre_sorted]
    roots = [
        _album_root(genre_sorted[i], artists[i], albums[i], rules, overrides)
        for i in range(len(genre_sorted))
    ]
    unique_albums_df["Root Genre"] = [display_root(r) for r in roots]
    tag_sets_by_name = dict(zip(names, tag_sets))
    root_by_name = dict(zip(names, roots))

    # Per-tag one-hot (shared by legacy ordering and the two-level micro/macro
    # steps) and the root-weighted similarity used by the single-pass "roots".
    M = MultiLabelBinarizer().fit_transform(genre_sorted)
    sim_tags = cosine_similarity(M) if M.shape[1] else np.zeros((len(names), len(names)))
    sim_roots = genre_similarity_matrix(genre_sorted, rules, root_weight)

    orders = {
        "legacy": _order_from_similarity(names, sim_tags, segmentation_strength, max_clusters),
        "roots": _order_from_similarity(names, sim_roots, segmentation_strength, max_clusters),
        "two_level": _two_level_order(
            names, M, sim_tags, tag_sets, roots, segmentation_strength, max_clusters
        ),
    }
    metrics = {m: _ordering_metric(o, tag_sets_by_name, root_by_name) for m, o in orders.items()}

    if ordering_mode not in orders:
        ordering_mode = "two_level"
    chosen = orders[ordering_mode]

    sort_index = {name: i for i, name in enumerate(chosen)}
    unique_albums_df["Sort Order"] = unique_albums_df["Unique Album"].map(sort_index)
    unique_albums_df = unique_albums_df.sort_values("Sort Order")
    ordering = unique_albums_df[["Unique Album", "Sort Order", "Sorted Genres", "Root Genre"]]
    return ordering, metrics


# -----------------------------
#  Top-level entry point
# -----------------------------
def run(backend, config, refresh_cache=False, no_cache=False):
    """Run the full pipeline for an authenticated backend."""
    source_slug, source_label, songs_data = _collect_source(backend)

    df = pd.DataFrame(songs_data)
    if df.empty:
        print("No tracks found for the selected source. Nothing to sort.")
        sys.exit(0)

    overrides_file = config.get("GENRE", "overrides_file", fallback=None) or None
    overrides = load_overrides(overrides_file)
    if overrides:
        print(f"🛠️  Loaded {len(overrides)} manual genre override(s).")

    cache = build_cache_from_config(config, refresh=refresh_cache, disabled=no_cache)
    if cache.enabled:
        mode = " (refresh)" if refresh_cache else ""
        print(f"🗃️  Genre cache: {cache.backend}{mode}")
    print("🔎 Fetching genres for songs (with shared helpers)...")
    get_best_genre = _make_genre_resolver(backend, config, cache, overrides)
    album_genres, album_genre_sources = [], []
    try:
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
    finally:
        cache.close()
    df["Album Genre"] = album_genres
    df["source"] = album_genre_sources

    # Unique identifier - group by normalized album name + primary artist so
    # that multiple Tidal editions/IDs of the same album are treated as one
    # album. Keep the raw Album ID for various-artist releases (compilations /
    # soundtracks) so they are not split apart by their per-track artists.
    _clean_name = df["Album"].fillna("").map(clean_album_name).str.casefold().str.strip()
    _artist_key = df["Artist"].fillna("").str.casefold().str.strip()
    _name_key = _clean_name + " \u2014 " + _artist_key
    _artists_per_album = df.groupby("Album ID")["Artist"].transform("nunique")
    df["Unique Album"] = _name_key.where(
        df["Album ID"].isna() | (_artists_per_album <= 1),
        df["Album ID"].astype("string"),
    )

    segmentation_strength = float(config.get("CLUSTERING", "segmentation_strength", fallback="0.6"))
    max_clusters = int(config.get("CLUSTERING", "max_clusters", fallback="10"))
    root_weight = float(config.get("CLUSTERING", "genre_root_weight", fallback="2.0"))
    roots_file = config.get("CLUSTERING", "genre_roots_file", fallback=None) or None
    rules = load_genre_roots(roots_file)
    ordering_mode = config.get("CLUSTERING", "ordering_mode", fallback="two_level").strip().lower()
    if ordering_mode not in ORDERING_MODES:
        ordering_mode = "two_level"
    ordering, metrics = _order_albums(
        df, segmentation_strength, max_clusters, root_weight, rules, overrides, ordering_mode
    )

    legacy, roots, two = metrics["legacy"], metrics["roots"], metrics["two_level"]
    print("\n📊 Genre ordering (higher adjacent overlap / lower fragmentation is better):")
    print(f"   Adjacent tag overlap (Jaccard): legacy {legacy['overlap']:.3f}  "
          f"roots {roots['overlap']:.3f}  two_level {two['overlap']:.3f}")
    print(f"   Fragmented root families:       legacy {legacy['fragmented']}  "
          f"roots {roots['fragmented']}  two_level {two['fragmented']}")
    print(f"   Active ordering mode: {ordering_mode}")

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
