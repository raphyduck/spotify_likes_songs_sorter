# Spotify Liked Songs Sorter

A small utility for Spotify users who maintain a large collection of liked songs and want to keep them organized.
The sorter analyzes each track's audio features and genre information to build playlists that better reflect your listening moods.

## Features

- Fetches your liked songs directly from Spotify using the Web API.
- Groups tracks by configurable genres or by similarity in audio features.
- Writes playlists back to your Spotify account, allowing for quick access on any device.
- Provides helper scripts for experimenting with genre detection and configuration.

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
2. Edit `settings.ini` and provide your Spotify Client ID, Client Secret, and redirect URI. You can also customize genre groupings and playlist naming conventions in this file.

## Usage

1. Ensure your virtual environment is active and your configuration file is set up.
2. Run the sorter script:
   ```bash
   python spotify_sorter.py
   ```
3. Follow the console prompts to authenticate with Spotify. Once authenticated, the script will process your liked songs and create playlists based on your configuration.

### Debugging Genres

If you want to inspect genre data for specific artists or tracks, use the helper script:
```bash
python debug_genres.py
```
This script will output available genre information and help you refine the genre configuration used by the main sorter.

## Development

- Formatting and linting are handled by standard Python tooling; feel free to use `black` or `ruff` as desired.
- Contributions are welcome! Please open an issue or submit a pull request with improvements or bug fixes.

## License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.
