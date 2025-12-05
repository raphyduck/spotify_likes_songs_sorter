# Spotify Liked Songs Sorter

A small utility for Spotify users who maintain a large collection of liked songs and want to keep them organized. The sorter enriches each liked track with album- and artist-level genre metadata so that it can group and order your collection by actual listening styles.

## Features

- Fetches your liked songs directly from Spotify using the Web API.
- Aggregates genre information for every album from Discogs, Last.fm, MusicBrainz, Spotify, Wikipedia, and more.
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

## Usage

1. Ensure your virtual environment is active and your configuration file is set up.
2. Run the sorter script:
   ```bash
   python spotify_sorter.py
   ```
3. Follow the console prompts to authenticate with Spotify. Once authenticated, the script will process your liked songs and create playlists based on your configuration.

### Debugging Genres

If you want to inspect genre data for specific artists or tracks, use the helper script. The script requires details about the artist, album, track, and Spotify album ID so that it can query the same metadata sources as the main sorter:

```bash
python debug_genres.py \
  --artist "Radiohead" \
  --album "In Rainbows" \
  --song "Weird Fishes/Arpeggi" \
  --album-id "5vkqYmiPBYLaalcmjujWxK"
```

Replace the sample values with the artist/album/track you want to inspect (the album ID is available from the Spotify desktop or web client). The script will output available genre information and help you refine the genre configuration used by the main sorter.

## Development

- Formatting and linting are handled by standard Python tooling; feel free to use `black` or `ruff` as desired.
- Contributions are welcome! Please open an issue or submit a pull request with improvements or bug fixes.

## License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.
