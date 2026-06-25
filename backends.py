"""Streaming-service backends for the song sorter.

Each backend hides the service-specific bits (authentication, fetching liked
tracks/playlists, the per-service genre providers, and creating/filling the
output playlist) behind a small common interface consumed by ``sorter_core``.
Two backends are provided: :class:`SpotifyBackend` and :class:`TidalBackend`.
"""

import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from tqdm import tqdm

from genre_helpers import (
    get_discogs_album_info,
    get_itunes_album_info,
    get_lastfm_album_info,
    get_musicbrainz_album_info,
    get_lastfm_track_info,
    get_wikipedia_album_info,
    get_spotify_album_info,
    get_spotify_artist_genres,
    get_spotify_track_artist_genres,
)


class Backend:
    """Common interface implemented by every streaming-service backend."""

    key = ""               # short identifier, e.g. "spotify"
    display_name = ""      # human label, e.g. "Spotify"
    liked_label = ""       # label for the "saved tracks" source
    liked_slug = ""        # slug used in CSV file names
    track_id_col = ""      # DataFrame column holding the service track id
    supports_local = False # whether the service can return local files

    def authenticate(self, config):
        raise NotImplementedError

    def get_liked_songs(self):
        raise NotImplementedError

    def get_user_playlists(self):
        raise NotImplementedError

    def playlist_display(self, playlist):
        """Return ``(name, track_total)`` for the selection menu."""
        raise NotImplementedError

    def get_playlist_tracks(self, selected_playlists):
        raise NotImplementedError

    def get_genre_providers(self, song, artist, album, clean_album,
                            album_id, track_id, config):
        """Return an ordered list of ``(source_label, callable)`` providers."""
        raise NotImplementedError

    def create_playlist(self, name, description):
        raise NotImplementedError

    def add_tracks(self, handle, ordered_rows):
        """Upload the ordered rows. Return ``(uploaded, local_skipped)``."""
        raise NotImplementedError


# -----------------------------------------------------------------------------
#  Spotify
# -----------------------------------------------------------------------------
class SpotifyBackend(Backend):
    key = "spotify"
    display_name = "Spotify"
    liked_label = "Liked songs"
    liked_slug = "liked_songs"
    track_id_col = "Spotify Track ID"
    supports_local = True

    EXPECTED_REDIRECT_URI = "http://127.0.0.1:8080/"

    def __init__(self):
        self.sp = None
        self.user_id = None
        self._discogs_key = None
        self._lastfm_key = None

    # --- auth -----------------------------------------------------------------
    def authenticate(self, config):
        import spotipy
        from spotipy.oauth2 import SpotifyOAuth

        client_id = config["SPOTIFY"]["CLIENT_ID"]
        client_secret = config["SPOTIFY"]["CLIENT_SECRET"]
        redirect_uri = config["SPOTIFY"]["REDIRECT_URI"]
        self._discogs_key = config["DISCOGS"]["API_KEY"]
        self._lastfm_key = config["LASTFM"]["API_KEY"]

        if redirect_uri.rstrip("/") + "/" != self.EXPECTED_REDIRECT_URI:
            print(
                "ERROR: The console authorization flow requires the redirect URI to be"
                f" set to {self.EXPECTED_REDIRECT_URI}. Please update settings.ini and"
                " your Spotify Developer Dashboard to match."
            )
            sys.exit(1)

        scope = " ".join([
            "user-library-read",
            "user-read-private",
            "playlist-read-private",
            "playlist-modify-private",
        ])

        print("\n🔄 Authenticating with Spotify...")
        oauth = SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            scope=scope,
            open_browser=False,
            cache_path=".cache-spotify-sorter",
        )

        token_info = oauth.validate_token(oauth.cache_handler.get_cached_token())
        if not token_info:
            auth_url = oauth.get_authorize_url()
            redirect_in_url = parse_qs(urlparse(auth_url).query).get("redirect_uri", [""])[0]
            if redirect_in_url.rstrip("/") + "/" != self.EXPECTED_REDIRECT_URI:
                print(
                    "ERROR: Generated authorize URL does not match the expected redirect"
                    f" URI ({self.EXPECTED_REDIRECT_URI}), which can lead to INVALID_CLIENT."
                )
                sys.exit(1)
            print("\n=== Spotify OAuth (console mode) ===")
            print("1) Open this URL in a browser (copy/paste):\n")
            print(auth_url)
            print("\n2) Log in, authorize the app, then COPY/PASTE the full redirect URL"
                  " here (it starts with http://127.0.0.1:8080/ ...):\n")
            try:
                redirected_url = input("> Redirect URL: ").strip()
            except EOFError:
                print("Missing input. Re-run and paste the redirect URL.", file=sys.stderr)
                sys.exit(1)
            code_list = parse_qs(urlparse(redirected_url).query).get("code")
            if not code_list:
                print("No 'code' found in the URL. Make sure you pasted the full URL.",
                      file=sys.stderr)
                sys.exit(1)
            token_info = oauth.get_access_token(code_list[0], as_dict=True)
            if not token_info or "access_token" not in token_info:
                print("Could not obtain an access_token.", file=sys.stderr)
                sys.exit(1)

        self.sp = spotipy.Spotify(auth_manager=oauth)
        self.user_id = self.sp.current_user()["id"]
        print("✅ Authentication successful!\n")

    # --- mapping --------------------------------------------------------------
    @staticmethod
    def _map_track(track):
        if not track:
            return None
        artists = track.get("artists") or []
        album = track.get("album") or {}
        return {
            "Song": track.get("name") or "Unknown Song",
            "Artist": artists[0].get("name") if artists else "Unknown Artist",
            "Album": album.get("name") or "Unknown Album",
            "Album ID": album.get("id"),
            "Track Number": track.get("track_number"),
            "Disc Number": track.get("disc_number"),
            "Spotify Track ID": track.get("id"),
            "Spotify URI": track.get("uri"),
            "Is Local": bool(track.get("is_local", False)),
        }

    # --- fetching -------------------------------------------------------------
    def get_liked_songs(self):
        rows = []
        results = self.sp.current_user_saved_tracks(limit=50)
        total = results.get("total", 0)
        print("🎵 Fetching liked songs from Spotify...")
        with tqdm(total=total, desc="Liked songs", unit="track") as pbar:
            while results:
                for item in results["items"]:
                    row = self._map_track(item.get("track"))
                    if row:
                        rows.append(row)
                pbar.update(len(results.get("items", [])))
                results = self.sp.next(results) if results.get("next") else None
                time.sleep(0.5)
        print(f"🎉 Retrieved {len(rows)} songs!\n")
        return rows

    def get_user_playlists(self):
        playlists = []
        results = self.sp.current_user_playlists(limit=50)
        while results:
            playlists.extend(results.get("items", []))
            results = self.sp.next(results) if results.get("next") else None
            time.sleep(0.2)
        return playlists

    def playlist_display(self, playlist):
        return (
            playlist.get("name", "Untitled"),
            (playlist.get("tracks", {}) or {}).get("total", 0),
        )

    def get_playlist_tracks(self, selected_playlists):
        rows = []
        print("🎵 Fetching tracks from selected playlist(s)...")
        total = sum((p.get("tracks") or {}).get("total", 0) for p in selected_playlists)
        with tqdm(total=total, desc="Playlist tracks", unit="track") as pbar:
            for playlist in selected_playlists:
                results = self.sp.playlist_items(playlist["id"], limit=100)
                while results:
                    items = results.get("items", [])
                    for item in items:
                        row = self._map_track(item.get("track"))
                        if row:
                            rows.append(row)
                    pbar.update(len(items))
                    results = self.sp.next(results) if results.get("next") else None
                    time.sleep(0.2)
        return rows

    # --- genres ---------------------------------------------------------------
    def get_genre_providers(self, song, artist, album, clean_album,
                            album_id, track_id, config):
        sp = self.sp
        providers = [
            ("Discogs", lambda: get_discogs_album_info(clean_album, artist, self._discogs_key)),
        ]
        if album_id:
            providers.append(("Spotify Album", lambda: get_spotify_album_info(sp, album_id)))
        if track_id:
            providers.append(("Spotify Track Artist", lambda: get_spotify_track_artist_genres(sp, track_id)))
        providers.extend([
            ("LastFM Album", lambda: get_lastfm_album_info(clean_album, artist, self._lastfm_key)),
            ("MusicBrainz", lambda: get_musicbrainz_album_info(clean_album, artist)),
            ("LastFM Track", lambda: get_lastfm_track_info(song, artist, self._lastfm_key)),
            ("Spotify Artist", lambda: get_spotify_artist_genres(sp, artist)),
            ("Wikipedia", lambda: get_wikipedia_album_info(clean_album, artist)),
            ("iTunes", lambda: get_itunes_album_info(clean_album, artist)),
        ])
        return providers

    # --- output ---------------------------------------------------------------
    def create_playlist(self, name, description):
        playlist = self.sp.user_playlist_create(
            user=self.user_id, name=name, public=False, description=description
        )
        return playlist["id"]

    def add_tracks(self, playlist_id, ordered_rows):
        track_uris = [
            f"spotify:track:{row.get(self.track_id_col)}"
            for row in ordered_rows
            if isinstance(row.get(self.track_id_col), str) and row.get(self.track_id_col)
        ]
        local_count = sum(1 for row in ordered_rows if row.get("Is Local"))
        chunks = [track_uris[i:i + 100] for i in range(0, len(track_uris), 100)]
        for chunk in tqdm(chunks, desc="Uploading playlist", unit="chunk"):
            self.sp.playlist_add_items(playlist_id, chunk)
            time.sleep(0.5)
        return len(track_uris), local_count


# -----------------------------------------------------------------------------
#  Tidal
# -----------------------------------------------------------------------------
class TidalBackend(Backend):
    key = "tidal"
    display_name = "Tidal"
    liked_label = "Favorite tracks"
    liked_slug = "favorite_tracks"
    track_id_col = "Tidal Track ID"
    supports_local = False

    PAGE_LIMIT = 100

    def __init__(self):
        self.session = None
        self._discogs_key = None
        self._lastfm_key = None

    # --- auth -----------------------------------------------------------------
    def authenticate(self, config):
        import tidalapi

        self._discogs_key = config["DISCOGS"]["API_KEY"]
        self._lastfm_key = config["LASTFM"]["API_KEY"]
        session_file = Path(
            config.get("TIDAL", "session_file", fallback=None)
            or os.path.join(os.path.expanduser("~"), ".tidal_sorter_session.json")
        )

        print("\n🔄 Authenticating with Tidal...")
        print("\n=== Tidal OAuth (console mode) ===")
        print("If no cached session is found, open the link printed below in a")
        print("browser, log in, and authorize the device. Authorization is then")
        print("detected automatically.\n")
        session = tidalapi.Session()
        session.login_session_file(session_file)
        if not session.check_login():
            print("❌ Tidal authentication failed. Please retry.", file=sys.stderr)
            sys.exit(1)
        self.session = session
        print("✅ Authentication successful!\n")

    # --- mapping --------------------------------------------------------------
    @staticmethod
    def _map_track(track):
        if not track:
            return None
        try:
            if getattr(track, "artist", None) and track.artist.name:
                artist_name = track.artist.name
            elif getattr(track, "artists", None):
                artist_name = track.artists[0].name
            else:
                artist_name = "Unknown Artist"
        except Exception:
            artist_name = "Unknown Artist"
        album = getattr(track, "album", None)
        album_id = getattr(album, "id", None)
        return {
            "Song": getattr(track, "name", None) or "Unknown Song",
            "Artist": artist_name,
            "Album": getattr(album, "name", None) or "Unknown Album",
            "Album ID": str(album_id) if album_id is not None else None,
            "Track Number": getattr(track, "track_num", None),
            "Disc Number": getattr(track, "volume_num", None),
            "Tidal Track ID": str(track.id) if getattr(track, "id", None) is not None else None,
        }

    # --- fetching -------------------------------------------------------------
    def get_liked_songs(self):
        rows = []
        favorites = self.session.user.favorites
        try:
            total = favorites.get_tracks_count()
        except Exception:
            total = 0
        print("🎵 Fetching favorite tracks from Tidal...")
        offset = 0
        with tqdm(total=total or None, desc="Favorite tracks", unit="track") as pbar:
            while True:
                batch = favorites.tracks(limit=self.PAGE_LIMIT, offset=offset)
                if not batch:
                    break
                for track in batch:
                    row = self._map_track(track)
                    if row:
                        rows.append(row)
                pbar.update(len(batch))
                if len(batch) < self.PAGE_LIMIT:
                    break
                offset += self.PAGE_LIMIT
                time.sleep(0.3)
        print(f"🎉 Retrieved {len(rows)} favorite tracks!\n")
        return rows

    def get_user_playlists(self):
        return self.session.user.playlists()

    def playlist_display(self, playlist):
        return (
            getattr(playlist, "name", None) or "Untitled",
            getattr(playlist, "num_tracks", None) or 0,
        )

    def get_playlist_tracks(self, selected_playlists):
        rows = []
        print("🎵 Fetching tracks from selected playlist(s)...")
        total = sum(self.playlist_display(p)[1] for p in selected_playlists)
        with tqdm(total=total or None, desc="Playlist tracks", unit="track") as pbar:
            for playlist in selected_playlists:
                offset = 0
                while True:
                    batch = playlist.tracks(limit=self.PAGE_LIMIT, offset=offset)
                    if not batch:
                        break
                    for track in batch:
                        row = self._map_track(track)
                        if row:
                            rows.append(row)
                    pbar.update(len(batch))
                    if len(batch) < self.PAGE_LIMIT:
                        break
                    offset += self.PAGE_LIMIT
                    time.sleep(0.2)
        return rows

    # --- genres ---------------------------------------------------------------
    def get_genre_providers(self, song, artist, album, clean_album,
                            album_id, track_id, config):
        # Tidal exposes no genre metadata, so only name-based providers apply.
        return [
            ("Discogs", lambda: get_discogs_album_info(clean_album, artist, self._discogs_key)),
            ("LastFM Album", lambda: get_lastfm_album_info(clean_album, artist, self._lastfm_key)),
            ("MusicBrainz", lambda: get_musicbrainz_album_info(clean_album, artist)),
            ("LastFM Track", lambda: get_lastfm_track_info(song, artist, self._lastfm_key)),
            ("Wikipedia", lambda: get_wikipedia_album_info(clean_album, artist)),
            ("iTunes", lambda: get_itunes_album_info(clean_album, artist)),
        ]

    # --- output ---------------------------------------------------------------
    def create_playlist(self, name, description):
        return self.session.user.create_playlist(name, description)

    def add_tracks(self, playlist, ordered_rows):
        track_ids = [
            str(row.get(self.track_id_col))
            for row in ordered_rows
            if isinstance(row.get(self.track_id_col), str) and row.get(self.track_id_col)
        ]
        chunks = [track_ids[i:i + 100] for i in range(0, len(track_ids), 100)]
        for chunk in tqdm(chunks, desc="Uploading playlist", unit="chunk"):
            playlist.add(chunk)
            time.sleep(0.5)
        return len(track_ids), 0


BACKENDS = {
    "spotify": SpotifyBackend,
    "tidal": TidalBackend,
}
