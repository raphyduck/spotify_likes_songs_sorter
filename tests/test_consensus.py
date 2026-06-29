import unittest
from unittest.mock import MagicMock

from genre_normalization import merge_consensus


class MergeConsensusTest(unittest.TestCase):
    def test_multi_source_agreement_outranks_single_source_outlier(self):
        collected = [
            ("Discogs", ["Indie Rock", "Post-Punk"]),
            ("LastFM Album", ["Indie Rock"]),
            ("Spotify Artist", ["Pop"]),  # lone, low-weight outlier
        ]
        merged = merge_consensus(collected)
        self.assertEqual(merged[0], "Indie Rock")  # confirmed by two sources, highest weight
        # The low-weight single-source "Pop" is dropped by the keep_ratio cut.
        self.assertNotIn("Pop", merged)

    def test_preserves_display_casing_and_dedupes(self):
        merged = merge_consensus([("Discogs", ["Jazz", "jazz", "JAZZ"])])
        self.assertEqual(merged, ["Jazz"])

    def test_empty(self):
        self.assertEqual(merge_consensus([]), [])

    def test_deterministic_tie_break(self):
        collected = [("LastFM Album", ["Beta", "Alpha"])]  # equal weight
        self.assertEqual(merge_consensus(collected), ["Alpha", "Beta"])


class ResolverConsensusTest(unittest.TestCase):
    def test_resolver_consensus_calls_all_and_merges(self):
        import sorter_core
        from genre_cache import GenreCache

        p1 = MagicMock(return_value=["Indie Rock"])
        p2 = MagicMock(return_value=["Indie Rock", "Shoegaze"])
        backend = MagicMock()
        backend.get_genre_providers.return_value = [("Discogs", p1), ("LastFM Album", p2)]

        cache = GenreCache(backend="none")
        resolve = sorter_core._make_genre_resolver(
            backend, {}, cache, overrides=None, resolution="consensus"
        )
        genres, source = resolve("s", "Artist", "Album", "alb", None)

        self.assertEqual(source, "Consensus")
        self.assertIn("Indie Rock", genres)
        p1.assert_called_once()
        p2.assert_called_once()  # consensus consults every provider

    def test_first_match_short_circuits(self):
        import sorter_core
        from genre_cache import GenreCache

        p1 = MagicMock(return_value=["Indie Rock"])
        p2 = MagicMock(return_value=["Shoegaze"])
        backend = MagicMock()
        backend.get_genre_providers.return_value = [("Discogs", p1), ("LastFM Album", p2)]

        cache = GenreCache(backend="none")
        resolve = sorter_core._make_genre_resolver(
            backend, {}, cache, overrides=None, resolution="first_match"
        )
        genres, source = resolve("s", "Artist", "Album", "alb", None)

        self.assertEqual((genres, source), (["Indie Rock"], "Discogs"))
        p2.assert_not_called()  # second provider never reached


if __name__ == "__main__":
    unittest.main()
