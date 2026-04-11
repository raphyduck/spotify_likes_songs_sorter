import unittest
from unittest.mock import patch

from genre_helpers import lookup_genres


class LookupGenresOrderTest(unittest.TestCase):
    def setUp(self):
        self.cfg = {
            "SPOTIFY": {"CLIENT_ID": "x", "CLIENT_SECRET": "y"},
            "DISCOGS": {"API_KEY": "d"},
            "LASTFM": {"API_KEY": "l"},
        }

    def _run(self, album_id=None, track_id=None):
        calls = []

        def mark(name):
            def _fn(*_args, **_kwargs):
                calls.append(name)
                return [name]
            return _fn

        with patch("genre_helpers.spotipy.Spotify", return_value=object()), \
             patch("genre_helpers.SpotifyClientCredentials", return_value=object()), \
             patch("genre_helpers.get_discogs_album_info", side_effect=mark("Discogs")), \
             patch("genre_helpers.get_spotify_album_info", side_effect=mark("Spotify Album")), \
             patch("genre_helpers.get_spotify_track_artist_genres", side_effect=mark("Spotify Track Artist")), \
             patch("genre_helpers.get_lastfm_album_info", side_effect=mark("LastFM Album")), \
             patch("genre_helpers.get_musicbrainz_album_info", side_effect=mark("MusicBrainz")), \
             patch("genre_helpers.get_lastfm_track_info", side_effect=mark("LastFM Track")), \
             patch("genre_helpers.get_spotify_artist_genres", side_effect=mark("Spotify Artist")), \
             patch("genre_helpers.get_wikipedia_album_info", side_effect=mark("Wikipedia")), \
             patch("genre_helpers.get_itunes_album_info", side_effect=mark("iTunes")):
            results = lookup_genres(
                "artist", "album", "song", album_id, self.cfg, track_id=track_id
            )

        return list(results.keys()), calls

    def test_order_with_album_and_track_id(self):
        keys, calls = self._run(album_id="alb", track_id="trk")
        expected = [
            "Discogs",
            "Spotify Album",
            "Spotify Track Artist",
            "LastFM Album",
            "MusicBrainz",
            "LastFM Track",
            "Spotify Artist",
            "Wikipedia",
            "iTunes",
        ]
        self.assertEqual(keys, expected)
        self.assertEqual(calls, expected)

    def test_order_without_optional_ids(self):
        keys, calls = self._run(album_id=None, track_id=None)
        expected = [
            "Discogs",
            "LastFM Album",
            "MusicBrainz",
            "LastFM Track",
            "Spotify Artist",
            "Wikipedia",
            "iTunes",
        ]
        self.assertEqual(keys, expected)
        self.assertEqual(calls, expected)


if __name__ == "__main__":
    unittest.main()
