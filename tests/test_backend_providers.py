import unittest
from unittest.mock import MagicMock

from backends import TidalBackend, SpotifyBackend


class TidalProviderOrderTest(unittest.TestCase):
    def test_name_based_only_without_spotify(self):
        backend = TidalBackend()  # no Spotify client configured
        providers = backend.get_genre_providers(
            "song", "artist", "album", "album", None, None, None
        )
        self.assertEqual(
            [name for name, _ in providers],
            ["Discogs", "LastFM Album", "MusicBrainz", "LastFM Track", "Wikipedia", "iTunes"],
        )

    def test_spotify_first_when_cross_lookup_enabled(self):
        backend = TidalBackend()
        backend._spotify = MagicMock()  # pretend Spotify cross-lookup is available
        providers = backend.get_genre_providers(
            "song", "artist", "album", "album", None, None, None
        )
        self.assertEqual(
            [name for name, _ in providers],
            ["Spotify Album", "Spotify Artist", "Discogs", "LastFM Album",
             "MusicBrainz", "LastFM Track", "Wikipedia", "iTunes"],
        )


class SpotifyProviderOrderTest(unittest.TestCase):
    def test_full_chain_with_ids(self):
        backend = SpotifyBackend()
        backend.sp = MagicMock()
        backend._discogs_key = "d"
        backend._lastfm_key = "l"
        providers = backend.get_genre_providers(
            "song", "artist", "album", "album", "album_id", "track_id", None
        )
        self.assertEqual(
            [name for name, _ in providers],
            ["Discogs", "Spotify Album", "Spotify Track Artist", "LastFM Album",
             "MusicBrainz", "LastFM Track", "Spotify Artist", "Wikipedia", "iTunes"],
        )


if __name__ == "__main__":
    unittest.main()
