#!/usr/bin/env python3
"""Liked / favorite songs sorter for Spotify and Tidal.

At launch you choose which streaming service to act on; the rest of the
workflow (genre enrichment, clustering, ordered-playlist creation, CSV export)
is identical for both.
"""

__version__ = "2.0.0"

import sys
import argparse
import configparser

from backends import BACKENDS


def choose_service(preselected=None):
    if preselected:
        key = preselected.strip().lower()
        if key in BACKENDS:
            return key
        print(f"Unknown service '{preselected}'. Expected one of: {', '.join(BACKENDS)}.")
        sys.exit(1)

    options = list(BACKENDS.items())  # [(key, cls), ...]
    print("Select the streaming service:")
    for idx, (key, cls) in enumerate(options, start=1):
        print(f"  [{idx}] {cls.display_name}")
    while True:
        raw = input(f"> Choice (1-{len(options)}): ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1][0]
        if raw.lower() in BACKENDS:
            return raw.lower()
        print("Invalid choice. Please try again.")


def main():
    parser = argparse.ArgumentParser(description="Sort liked/favorite songs by genre similarity.")
    parser.add_argument(
        "--service",
        choices=list(BACKENDS),
        help="Streaming service to use (skips the interactive prompt).",
    )
    parser.add_argument("--config", default="settings.ini", help="Path to settings.ini")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    service_key = choose_service(args.service)
    backend = BACKENDS[service_key]()

    backend.authenticate(config)

    # Import here so the heavy data-science stack only loads once a service is chosen.
    import sorter_core
    sorter_core.run(backend, config)


if __name__ == "__main__":
    main()
