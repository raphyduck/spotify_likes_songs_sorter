import unittest

import pandas as pd

import sorter_core
from genre_normalization import load_genre_roots, infer_root, count_fragmented_roots


def _df(albums):
    """albums: list of (unique_album, artist, genres-list)."""
    rows = []
    for uid, artist, genres in albums:
        rows.append({
            "Unique Album": uid, "Album": uid, "Artist": artist,
            "Album Genre": genres, "Album ID": uid,
        })
    return pd.DataFrame(rows)


class TwoLevelOrderingTest(unittest.TestCase):
    def setUp(self):
        self.rules = load_genre_roots()
        # Interleave families so a naive order would ping-pong between them.
        self.albums = [
            ("a_pop_punk", "X", ["Pop Punk", "Emo"]),
            ("b_bebop", "Y", ["Bebop", "Jazz"]),
            ("c_skate", "Z", ["Skate Punk"]),
            ("d_house", "W", ["Deep House", "Electronic"]),
            ("e_hardcore", "V", ["Hardcore Punk"]),
            ("f_swing", "U", ["Swing", "Jazz"]),
            ("g_techno", "T", ["Techno"]),
            ("h_melodic", "S", ["Melodic Hardcore"]),
        ]

    def _order(self, mode):
        df = _df(self.albums)
        ordering, metrics = sorter_core._order_albums(
            df, 0.6, 10, 2.0, self.rules, overrides=None, ordering_mode=mode
        )
        names = list(ordering.sort_values("Sort Order")["Unique Album"])
        return names, metrics

    def test_each_root_family_is_contiguous(self):
        names, metrics = self._order("two_level")
        roots_in_order = []
        by_uid = {uid: infer_root(g, self.rules) for uid, _, g in self.albums}
        for name in names:
            roots_in_order.append(by_uid[name])
        # By construction every family forms a single contiguous block.
        self.assertEqual(count_fragmented_roots(roots_in_order), 0)
        self.assertEqual(metrics["two_level"]["fragmented"], 0)

    def test_two_level_minimizes_fragmentation(self):
        # The design guarantee: two-level never fragments more than the other
        # modes (and is 0 by construction). It trades a little raw adjacent
        # overlap for this contiguity, so we assert the fragmentation win, not
        # an overlap win (strict contiguity is a constraint, not a free lunch).
        _, metrics = self._order("two_level")
        self.assertEqual(metrics["two_level"]["fragmented"], 0)
        self.assertLessEqual(
            metrics["two_level"]["fragmented"], metrics["legacy"]["fragmented"]
        )
        self.assertLessEqual(
            metrics["two_level"]["fragmented"], metrics["roots"]["fragmented"]
        )

    def test_metrics_report_all_three_modes(self):
        _, metrics = self._order("two_level")
        self.assertEqual(set(metrics), {"legacy", "roots", "two_level"})

    def test_mode_selection_changes_order_deterministically(self):
        a1, _ = self._order("two_level")
        a2, _ = self._order("two_level")
        self.assertEqual(a1, a2)  # deterministic
        legacy, _ = self._order("legacy")
        # All modes are permutations of the same album set.
        self.assertEqual(set(a1), set(legacy))

    def test_tracks_of_album_stay_grouped_after_merge(self):
        # Two tracks of the same album must remain adjacent and in track order.
        rows = [
            {"Unique Album": "alb", "Album": "alb", "Artist": "X", "Album Genre": ["Pop Punk"],
             "Album ID": "alb", "Disc Number": 1, "Track Number": 2, "Tidal Track ID": "t2"},
            {"Unique Album": "alb", "Album": "alb", "Artist": "X", "Album Genre": ["Pop Punk"],
             "Album ID": "alb", "Disc Number": 1, "Track Number": 1, "Tidal Track ID": "t1"},
            {"Unique Album": "jz", "Album": "jz", "Artist": "Y", "Album Genre": ["Bebop"],
             "Album ID": "jz", "Disc Number": 1, "Track Number": 1, "Tidal Track ID": "t3"},
        ]
        df = pd.DataFrame(rows)
        ordering, _ = sorter_core._order_albums(df, 0.6, 10, 2.0, self.rules,
                                                ordering_mode="two_level")
        merged = (
            pd.merge(df, ordering, on="Unique Album", how="left")
            .sort_values(["Sort Order", "Disc Number", "Track Number"])
        )
        ids = list(merged["Tidal Track ID"])
        self.assertEqual(ids.index("t1") + 1, ids.index("t2"))  # t1 immediately before t2


if __name__ == "__main__":
    unittest.main()
