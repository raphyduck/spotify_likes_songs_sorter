# Liked Songs Sorter (Spotify & Tidal)

A small utility for **Spotify** and **Tidal** users who maintain a large collection of liked/favorite songs and want to keep them organized. At launch you pick which streaming service to act on; the sorter then enriches each track with album- and artist-level genre metadata so that it can group and order your collection by actual listening styles.

## Features

- **Pick your service at launch:** Spotify or Tidal — the same sorting workflow runs on either.
- Fetches input tracks from either your liked/favorite songs or one/multiple playlists.
- Aggregates genre information for every album from Discogs, Last.fm, MusicBrainz, Spotify (Spotify only), Wikipedia, and the iTunes Search API.
- Clusters albums by genre similarity and produces a smoothly ordered playlist plus a CSV export of the final ordering.
- Provides a helper script for inspecting the resolved genres and fine-tuning your configuration.

## Prerequisites

- Python 3.10 or newer.
- For **Spotify**: a Spotify Developer application (Client ID / Client Secret) configured for Web API access.
- For **Tidal**: an active Tidal subscription (no developer credentials required — see below).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/likes_songs_sorter.git
   cd likes_songs_sorter
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. Copy the sample configuration file and update it with your credentials:
   ```bash
   cp settings.ini.sample settings.ini
   ```
2. You only need to configure the service(s) you intend to use:
   - **Spotify** — provide your Client ID, Client Secret, and redirect URI. Register **exactly** `http://127.0.0.1:8080/` as a Redirect URI in your Spotify Developer Dashboard and mirror the same value in `settings.ini` to match the console authorization flow. The app requests scopes for liked songs, private profile, reading private playlists, and creating private playlists.
   - **Tidal** — no client id/secret needed. Authentication uses the built-in **device OAuth flow**: on first run the script prints a `link.tidal.com` URL — open it, log in, and authorize. The session is then cached (by default at `~/.tidal_sorter_session.json`) and reused/refreshed automatically. Override the cache path via `session_file` in the `[TIDAL]` section if desired.
3. Provide your Discogs and Last.fm API keys (used for genre lookups by both services).
4. Optional: the iTunes Search fallback does not need credentials and is used automatically when earlier providers cannot supply genres.

> Note: Tidal does not expose album/artist genre metadata through its API, so for Tidal
> genres are resolved from the name-based providers (Discogs, Last.fm, MusicBrainz,
> Wikipedia, iTunes). Spotify additionally uses Spotify's own genre data.

#### Spotify genre cross-lookup for Tidal (optional)

Since Tidal has no genre data of its own, you can let **Spotify help enrich your Tidal
tracks**. If you fill in the `[SPOTIFY]` `CLIENT_ID` / `CLIENT_SECRET` in `settings.ini`,
a Tidal run consults Spotify first, before the name-based providers:

1. **Spotify Album** — the album is matched on Spotify by name + artist, and its
   album-level genres are used (falling back to the album's artists' genres).
2. **Spotify Artist** — the artist's genres, as a broader fallback.

This uses Spotify's **Client Credentials** flow — no Spotify login, profile, or redirect URI
is required; the credentials are used only to read public genre data. Leave the Spotify
placeholders untouched to keep this disabled.

### Clustering and ordering

The sorter picks segmentation settings from the data: it tries several minimum-spanning-tree cuts and keeps the one with the best silhouette score, falling back to trimming the heaviest genre-distance edges. The greedy chaining step also adapts to the observed similarity distribution so resets happen only when similarities drop meaningfully.

You can tune how tight the grouping is without touching the code via `settings.ini`:

```ini
[CLUSTERING]
# 0.0 = looser, 1.0 = tighter; default 0.6
segmentation_strength = 0.6
# Upper bound for how many clusters the silhouette search will try (default 10)
max_clusters = 10
```

Lower `segmentation_strength` values keep broader groups (fewer cuts), while higher values favor more, smaller clusters and earlier resets in the chaining order. Increase `max_clusters` only if you have many distinct genre sets and want the silhouette search to consider finer splits.

### Genre cache (faster repeat runs)

Genre enrichment makes many third-party HTTP calls and can take 15-20 minutes on a large
library. Because genres are essentially fixed per album/artist, results are cached in a
**persistent two-level cache** (an in-memory layer plus a durable store), so a second run on
the same library finishes the enrichment step in seconds.

Configure it under `[CACHE]` in `settings.ini`:

```ini
[CACHE]
backend = redis                       # redis | file | none
redis_url = redis://localhost:6379/0
ttl_days = 90                         # TTL for found genres
negative_ttl_hours = 6                # short TTL so "not found" gets retried
# file_path = ~/.cache/likes_songs_sorter/genre_cache.json
```

- **`redis`** (default) uses a local Redis server. If Redis is unreachable the run
  **automatically falls back to a JSON file cache** (with a warning) instead of failing.
- **`file`** uses the JSON file directly; **`none`** disables persistence.
- The cached value keeps the resolved genres *and* their `source`, so the CSV's `source`
  column is identical across cached runs.
- CLI overrides: `--refresh-cache` re-fetches from providers and overwrites the cache;
  `--no-cache` disables the cache for that run.

## Usage

1. Ensure your virtual environment is active and your configuration file is set up.
2. Run the sorter:
   ```bash
   python sorter.py
   ```
   You can also skip the service prompt with `--service spotify` or `--service tidal`, and
   control the genre cache with `--refresh-cache` / `--no-cache`.
3. Follow the console prompts:
   - First choose the streaming service: `[1] Spotify` / `[2] Tidal`.
   - Authenticate (Spotify console paste flow, or Tidal device link — first run only).
   - Then choose your source:
     - `[1] Liked songs` (Spotify) / `Favorite tracks` (Tidal)
     - `[2] Playlist(s)` (multi-select with comma-separated numbers, e.g. `1,3,5`)
     - `[3] Liked/Favorite songs + one playlist`
4. The script then fetches tracks, sorts them by genre similarity, creates a playlist named `liked songs sorted YYYY-MM-DD`, and exports a CSV (`<service>_<source>_sorted_YYYY-MM-DD.csv`).

> Note: Spotify local files cannot be inserted into playlists through the Web API.
> The sorter still includes local tracks in sorting + CSV, logs them in the console, then
> uploads only tracks with valid Spotify IDs. Tidal has no local-file concept.

### Debugging Genres

If you want to inspect genre data for specific artists or tracks, use the helper script. It runs the shared, name-based genre providers (the ones common to both services):

```bash
python debug_genres.py \
  --artist "Radiohead" \
  --album "In Rainbows" \
  --song "Weird Fishes/Arpeggi"
```

Replace the sample values with the artist/album/track you want to inspect. The output prints the attempt order so you can quickly diagnose which providers were called and what each returned.

## Project layout

- `sorter.py` — entry point; chooses the service and runs the pipeline.
- `backends.py` — `SpotifyBackend` and `TidalBackend` (auth, fetching, per-service genre providers, playlist creation).
- `sorter_core.py` — service-agnostic pipeline (genre enrichment, clustering, ordering, CSV export).
- `genre_helpers.py` — individual genre-provider implementations.

## Development

- Run the test suite with `python -m unittest discover -s tests`.
- Formatting and linting are handled by standard Python tooling; feel free to use `black` or `ruff` as desired.
- Contributions are welcome! Please open an issue or submit a pull request with improvements or bug fixes.

## License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.
