# Tidal Favorite Songs Sorter

A small utility for Tidal users who maintain a large collection of favorite songs and want to keep them organized. The sorter enriches each track with album- and artist-level genre metadata so that it can group and order your collection by actual listening styles.

## Features

- Fetches input tracks from either your favorite tracks or one/multiple playlists.
- Aggregates genre information for every album from Discogs, Last.fm, MusicBrainz, Wikipedia, and the iTunes Search API.
- Clusters albums by genre similarity and produces a smoothly ordered playlist plus a CSV export of the final ordering.
- Provides helper scripts for inspecting the resolved genres and fine-tuning your configuration.

## Prerequisites

- Python 3.10 or newer.
- A Tidal subscription (the account whose favorites/playlists you want to sort).

> Note: Tidal does not expose album/artist genre metadata through its API, so genres
> are resolved exclusively from the name-based providers listed above. None of them
> require Tidal credentials.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/tidal_likes_songs_sorter.git
   cd tidal_likes_songs_sorter
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
2. Edit `settings.ini` and provide your Discogs and Last.fm API keys (used for genre lookups).
3. Tidal authentication uses the built-in **device OAuth flow** and needs no developer
   client id/secret or redirect URI. On first run the script prints a `link.tidal.com`
   URL — open it in a browser, log in, and authorize. The session is then cached (by
   default at `~/.tidal_sorter_session.json`) and reused/refreshed automatically on later
   runs. You can override the cache location via `session_file` in the `[TIDAL]` section.
4. Optional: the iTunes Search fallback does not need credentials and is used automatically
   when earlier providers cannot supply genres.

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

## Usage

1. Ensure your virtual environment is active and your configuration file is set up.
2. Run the sorter script:
   ```bash
   python tidal_sorter.py
   ```
3. Follow the console prompts to authenticate with Tidal (first run only), then choose your source:
   - `[1] Favorite tracks`
   - `[2] Playlist(s)` (multi-select with comma-separated numbers, e.g. `1,3,5`)
   - `[3] Favorite tracks + one playlist`
4. The script then fetches tracks, sorts them by genre similarity, creates a playlist named `liked songs sorted YYYY-MM-DD`, and exports a CSV.

### Debugging Genres

If you want to inspect genre data for specific artists or tracks, use the helper script:

```bash
python debug_genres.py \
  --artist "Radiohead" \
  --album "In Rainbows" \
  --song "Weird Fishes/Arpeggi"
```

Replace the sample values with the artist/album/track you want to inspect. The output prints the attempt order so you can quickly diagnose which providers were called and what each returned.

## Development

- Run the test suite with `python -m unittest discover -s tests`.
- Formatting and linting are handled by standard Python tooling; feel free to use `black` or `ruff` as desired.
- Contributions are welcome! Please open an issue or submit a pull request with improvements or bug fixes.

## License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.
