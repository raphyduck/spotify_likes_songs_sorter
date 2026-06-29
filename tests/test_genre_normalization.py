import unittest

import numpy as np

from genre_normalization import (
    load_genre_roots,
    root_of,
    primary_root,
    display_root,
    genre_similarity_matrix,
    avg_adjacent_overlap,
    count_fragmented_roots,
)


class RootMappingTest(unittest.TestCase):
    def setUp(self):
        self.rules = load_genre_roots()

    def test_default_rules_load(self):
        self.assertTrue(self.rules)  # genre_roots.json present and parseable

    def test_punk_family_collapses(self):
        for tag in ["Punk", "Pop Punk", "Skate Punk", "Emo"]:
            self.assertEqual(root_of(tag, self.rules), "punk")

    def test_french_family(self):
        for tag in ["Chanson", "Variété Française", "French Pop"]:
            self.assertEqual(root_of(tag, self.rules), "chanson_fr")

    def test_fallback_to_first_word(self):
        # A tag matching no rule falls back to its first word.
        self.assertEqual(root_of("Zeuhl Prog", []), "zeuhl")

    def test_primary_root_uses_first_tag(self):
        self.assertEqual(primary_root(["Pop Punk", "Indie"], self.rules), "punk")
        self.assertEqual(primary_root([], self.rules), "unknown")

    def test_display_root(self):
        self.assertEqual(display_root("chanson_fr"), "Chanson Fr")


class SimilarityTest(unittest.TestCase):
    def setUp(self):
        self.rules = load_genre_roots()

    def test_root_weight_pulls_same_family_closer(self):
        # Two punk albums with NO shared sub-tag, plus an unrelated jazz album.
        genre_lists = [["Pop Punk"], ["Skate Punk"], ["Bebop"]]
        plain = genre_similarity_matrix(genre_lists, self.rules, root_weight=0.0)
        weighted = genre_similarity_matrix(genre_lists, self.rules, root_weight=3.0)
        # With no root weight the two punk albums share nothing -> 0 similarity.
        self.assertAlmostEqual(plain[0, 1], 0.0, places=6)
        # Root weighting makes them similar, and still distinct from jazz.
        self.assertGreater(weighted[0, 1], 0.5)
        self.assertGreater(weighted[0, 1], weighted[0, 2])

    def test_empty_genres_safe(self):
        sim = genre_similarity_matrix([[], []], self.rules, root_weight=2.0)
        self.assertEqual(sim.shape, (2, 2))


class MetricTest(unittest.TestCase):
    def test_avg_adjacent_overlap(self):
        sets = [{"a", "b"}, {"a", "b"}, {"c"}]
        # overlap(1,1)=1.0 ; overlap({a,b},{c})=0 -> mean 0.5
        self.assertAlmostEqual(avg_adjacent_overlap(sets), 0.5)

    def test_count_fragmented_roots(self):
        self.assertEqual(count_fragmented_roots(["punk", "punk", "jazz"]), 0)
        # punk appears in two separate runs -> 1 fragmented family.
        self.assertEqual(count_fragmented_roots(["punk", "jazz", "punk"]), 1)


if __name__ == "__main__":
    unittest.main()
