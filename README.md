# Spotify Liked Songs Sorter

A small utility for Spotify users who maintain a large collection of liked songs and want to keep them organized. The sorter enriches each liked track with album- and artist-level genre metadata so that it can group and order your collection by actual listening styles.

## Features

- Fetches input tracks from either your liked songs or one/multiple playlists.
- Keeps local tracks in the sorting/CSV output when Spotify returns them.
- Aggregates genre information for every album from Discogs, Last.fm, MusicBrainz, Spotify, Wikipedia, iTunes Search, and more.
- Clusters albums by genre similarity and produces a smoothly ordered playlist plus a CSV export of the final ordering.
- Provides helper scripts for inspecting the resolved genres and fine-tuning your configuration.

## Prerequisites

- Python 3.10 or newer.
- A Spotify Developer account with an application configured for Web API access.
- Spotify OAuth credentials (Client ID and Client Secret) and a redirect URI.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/spotify_likes_songs_sorter.git
   cd spotify_likes_songs_sorter
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
2. Edit `settings.ini` and provide your Spotify Client ID, Client Secret, and redirect URI. Register **exactly** `http://127.0.0.1:8080/` as a Redirect URI in your Spotify Developer Dashboard and mirror the same value in `settings.ini` to match the console authorization flow.
   The app requests scopes for liked songs, private profile, reading private playlists, and creating private playlists.
3. Optional: the iTunes Search fallback does not need credentials and will be used automatically when earlier providers cannot supply genres.

### Clustering and ordering

The sorter now picks segmentation settings from the data: it tries several minimum-spanning-tree cuts and keeps the one with the best silhouette score, falling back to trimming the heaviest genre-distance edges. The greedy chaining step also adapts to the observed similarity distribution so resets happen only when similarities drop meaningfully.

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
   python spotify_sorter.py
   ```
3. Follow the console prompts to authenticate with Spotify, then choose your source:
   - `[1] Liked songs`
   - `[2] Playlist(s)` (multi-select with comma-separated numbers, e.g. `1,3,5`)
   - `[3] Liked songs + one playlist`
4. The script then fetches tracks, sorts them by genre similarity, creates a private playlist, and exports a CSV.

> Note: Spotify local files cannot be inserted into playlists through the Web API.  
> The sorter still includes local tracks in sorting + CSV, logs them in the console, then uploads only tracks with valid Spotify IDs.

### Debugging Genres

If you want to inspect genre data for specific artists or tracks, use the helper script. It accepts optional Spotify IDs so it can follow the same provider order/conditions as the main sorter (`if album_id`, `if track_id`):

```bash
python debug_genres.py \
  --artist "Radiohead" \
  --album "In Rainbows" \
  --song "Weird Fishes/Arpeggi" \
  --album-id "5vkqYmiPBYLaalcmjujWxK" \
  --track-id "4wajJ1o7jWIg62YqpkHC7S"
```

Replace the sample values with the artist/album/track you want to inspect (IDs are available from the Spotify desktop or web client). The output prints attempt order so you can quickly diagnose which providers were called and what each returned.

## Development

- Formatting and linting are handled by standard Python tooling; feel free to use `black` or `ruff` as desired.
- Contributions are welcome! Please open an issue or submit a pull request with improvements or bug fixes.

## License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.
