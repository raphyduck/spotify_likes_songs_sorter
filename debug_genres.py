#!/usr/bin/env python3
import argparse
import configparser
import requests
from genre_helpers import clean_album_name, lookup_genres

# Keep a reference to the original requests.get
_original_requests_get = requests.get

def main():
    parser = argparse.ArgumentParser(description="Debug genre lookup for a track")
    parser.add_argument("--artist",    required=True, help="Artist name")
    parser.add_argument("--album",     required=True, help="Album title")
    parser.add_argument("--song",      required=True, help="Song name")
    parser.add_argument("--album-id",  required=True, help="Spotify album ID")
    parser.add_argument("--config",    default="settings.ini", help="Path to settings.ini")
    parser.add_argument("--debug",     action="store_true", help="Enable HTTP request/response debug output")
    args = parser.parse_args()

    # If debug flag is set, monkey-patch requests.get to log URLs and responses
    if args.debug:
        def debug_get(url, *gargs, **gkwargs):
            params = gkwargs.get('params')
            print(f"DEBUG: GET {url} params={params}")
            try:
                resp = _original_requests_get(url, *gargs, **gkwargs)
                snippet = resp.text[:200].replace('\\n', ' ')
                print(f"DEBUG: Response {resp.status_code}; body snippet: '{snippet}'")
                return resp
            except Exception as e:
                print(f"DEBUG: HTTP error: {e}")
                raise
        requests.get = debug_get

    # Load configuration **inside** main, using args.config
    config = configparser.ConfigParser()
    config.read(args.config)

    # Run the lookup
    results = lookup_genres(
        args.artist,
        clean_album_name(args.album),
        args.song,
        args.album_id,
        {
            "SPOTIFY":  config["SPOTIFY"],
            "DISCOGS":  config["DISCOGS"],
            "LASTFM":   config["LASTFM"],
            **({"GOOGLE_CSE": config["GOOGLE_CSE"]} if "GOOGLE_CSE" in config else {})
        }
    )

    # Print results
    print("\nGenre lookup results:")
    for source, genres in results.items():
        if args.debug:
            print(f"[{source}] returned {genres}")
        else:
            print(f"{source}: {genres if genres else 'None found'}")

if __name__ == "__main__":
    main()
