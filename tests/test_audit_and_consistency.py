import unittest

import pandas as pd

import sorter_core
from genre_normalization import load_genre_roots
from audit_genres import find_suspects


class ArtistConsistencyTest(unittest.TestCase):
    def setUp(self):
        self.rules = load_genre_roots()

    def _roots(self, df, consistency):
        ordering, _ = sorter_core._order_albums(
            df, 0.6, 10, 2.0, self.rules, overrides=None,
            ordering_mode="two_level", artist_consistency=consistency,
        )
        return dict(zip(ordering["Unique Album"], ordering["Root Genre"]))

    def test_majority_root_snaps_outlier_album(self):
        # Same artist: 2 rock albums + 1 album that resolves to pop -> snapped to rock.
        df = pd.DataFrame([
            {"Unique Album": "a", "Album": "a", "Artist": "Band", "Album Genre": ["Indie Rock"], "Album ID": "a"},
            {"Unique Album": "b", "Album": "b", "Artist": "Band", "Album Genre": ["Alternative Rock"], "Album ID": "b"},
            {"Unique Album": "c", "Album": "c", "Artist": "Band", "Album Genre": ["Dance Pop"], "Album ID": "c"},
        ])
        without = self._roots(df, consistency=False)
        with_ = self._roots(df, consistency=True)
        self.assertEqual(without["c"], "Pop")     # left alone by default
        self.assertEqual(with_["c"], "Rock")      # snapped to artist majority

    def test_override_pinned_root_not_snapped(self):
        overrides = {"band|c": {"tags": [], "root": "jazz"}}
        df = pd.DataFrame([
            {"Unique Album": "a", "Album": "a", "Artist": "Band", "Album Genre": ["Indie Rock"], "Album ID": "a"},
            {"Unique Album": "b", "Album": "b", "Artist": "Band", "Album Genre": ["Alternative Rock"], "Album ID": "b"},
            {"Unique Album": "c", "Album": "c", "Artist": "Band", "Album Genre": ["Dance Pop"], "Album ID": "c"},
        ])
        ordering, _ = sorter_core._order_albums(
            df, 0.6, 10, 2.0, self.rules, overrides=overrides,
            ordering_mode="two_level", artist_consistency=True,
        )
        roots = dict(zip(ordering["Unique Album"], ordering["Root Genre"]))
        self.assertEqual(roots["c"], "Jazz")  # override wins over consistency snap


class AuditTest(unittest.TestCase):
    def test_flags_unknown_low_source_and_outlier(self):
        albums = [
            {"artist": "Band", "album": "1", "root": "Rock", "source": "Discogs", "genres": "Indie Rock"},
            {"artist": "Band", "album": "2", "root": "Rock", "source": "Discogs", "genres": "Rock"},
            {"artist": "Band", "album": "3", "root": "Pop", "source": "Discogs", "genres": "Pop"},   # outlier
            {"artist": "Solo", "album": "x", "root": "Unknown", "source": "Discogs", "genres": ""},   # unknown
            {"artist": "Solo", "album": "y", "root": "Pop", "source": "iTunes", "genres": "Pop"},     # low source
        ]
        suspects = find_suspects(albums)
        flagged = {(s["artist"], s["album"]) for s in suspects}
        self.assertIn(("Band", "3"), flagged)   # differs from artist majority (Rock)
        self.assertIn(("Solo", "x"), flagged)   # unknown root
        self.assertIn(("Solo", "y"), flagged)   # low-reliability source
        self.assertNotIn(("Band", "1"), flagged)
        self.assertNotIn(("Band", "2"), flagged)


if __name__ == "__main__":
    unittest.main()
