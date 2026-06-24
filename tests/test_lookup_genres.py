import unittest
from unittest.mock import patch

from genre_helpers import lookup_genres


class LookupGenresOrderTest(unittest.TestCase):
    def setUp(self):
        self.cfg = {
            "DISCOGS": {"API_KEY": "d"},
            "LASTFM": {"API_KEY": "l"},
        }

    def _run(self):
        calls = []

        def mark(name):
            def _fn(*_args, **_kwargs):
                calls.append(name)
                return [name]
            return _fn

        with patch("genre_helpers.get_discogs_album_info", side_effect=mark("Discogs")), \
             patch("genre_helpers.get_lastfm_album_info", side_effect=mark("LastFM Album")), \
             patch("genre_helpers.get_musicbrainz_album_info", side_effect=mark("MusicBrainz")), \
             patch("genre_helpers.get_lastfm_track_info", side_effect=mark("LastFM Track")), \
             patch("genre_helpers.get_wikipedia_album_info", side_effect=mark("Wikipedia")), \
             patch("genre_helpers.get_itunes_album_info", side_effect=mark("iTunes")):
            results = lookup_genres("artist", "album", "song", self.cfg)

        return list(results.keys()), calls

    def test_provider_order(self):
        keys, calls = self._run()
        expected = [
            "Discogs",
            "LastFM Album",
            "MusicBrainz",
            "LastFM Track",
            "Wikipedia",
            "iTunes",
        ]
        self.assertEqual(keys, expected)
        self.assertEqual(calls, expected)


if __name__ == "__main__":
    unittest.main()
