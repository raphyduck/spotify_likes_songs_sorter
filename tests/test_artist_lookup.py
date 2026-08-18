"""Verification du nom sur la recherche d'artiste Spotify.

Prendre ``items[0]`` en aveugle attribuait le reggae de Bob Marley a t.A.T.u.,
le country de Merle Haggard au metal symphonique de Haggard et le manele de
"Bogdan Artistu" a U2.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genre_helpers import (  # noqa: E402
    artist_hit_matches,
    artist_query_variants,
    get_spotify_artist_genres,
)


class FakeSpotify:
    """Renvoie une liste d'artistes fixe, quelle que soit la requete."""

    def __init__(self, items):
        self.items = items
        self.queries = []

    def search(self, q, type=None, limit=None):
        self.queries.append(q)
        return {"artists": {"items": self.items}}


class ArtistMatchTest(unittest.TestCase):
    def test_accepts_accent_punctuation_and_word_order(self):
        self.assertTrue(artist_hit_matches("Celine Dion", "Céline Dion"))
        self.assertTrue(artist_hit_matches("Cesaria Evora", "Cesária Évora"))
        self.assertTrue(artist_hit_matches("Neg'Marrons", "Nèg' Marrons"))
        self.assertTrue(artist_hit_matches("Leonard Cohen", "Cohen, Leonard"))
        self.assertTrue(artist_hit_matches("A-Ha", "a-ha"))

    def test_rejects_homonyms_and_artist_word_noise(self):
        self.assertFalse(artist_hit_matches("Trust", "Men I Trust"))
        self.assertFalse(artist_hit_matches("Europe", "D-Block Europe"))
        self.assertFalse(artist_hit_matches("Haggard", "Merle Haggard"))
        self.assertFalse(artist_hit_matches("Talk Talk", "Talking Heads"))
        self.assertFalse(artist_hit_matches("Dr. Dre", "Drake"))
        self.assertFalse(artist_hit_matches("Placebo", "Placebo Effect"))
        self.assertFalse(artist_hit_matches("U2", "Bogdan Artistu"))
        self.assertFalse(artist_hit_matches("P!nk", "ARTIST: UNKNOWN"))
        self.assertFalse(artist_hit_matches("Muse", "Artist Music Video"))


class ArtistVariantTest(unittest.TestCase):
    def test_primary_artist_is_tried_after_the_full_string(self):
        self.assertEqual(
            artist_query_variants("Magic System, Ahmed Chawki")[:2],
            ["Magic System, Ahmed Chawki", "Magic System"],
        )

    def test_two_part_name_is_also_tried_reversed(self):
        self.assertIn("Leonard Cohen", artist_query_variants("Cohen, Leonard"))

    def test_feat_and_bullet_are_separators(self):
        self.assertEqual(
            artist_query_variants("Ofenbach feat. Norma Jean Martine")[1], "Ofenbach"
        )
        self.assertEqual(
            artist_query_variants("Harris & Ford \u2022 2 Engel")[1], "Harris & Ford"
        )

    def test_blank_artist_yields_nothing(self):
        self.assertEqual(artist_query_variants(""), [])
        self.assertEqual(artist_query_variants(None), [])


class GetSpotifyArtistGenresTest(unittest.TestCase):
    def test_wrong_artist_is_discarded_rather_than_used(self):
        sp = FakeSpotify([{"name": "Merle Haggard", "genres": ["country"]}])
        self.assertEqual(get_spotify_artist_genres(sp, "Haggard"), [])

    def test_matching_artist_further_down_the_list_is_found(self):
        sp = FakeSpotify([
            {"name": "Kool & The Gang", "genres": ["disco"]},
            {"name": "t.A.T.u.", "genres": ["russian pop"]},
        ])
        self.assertEqual(get_spotify_artist_genres(sp, "t.A.T.u."), ["russian pop"])

    def test_multi_artist_string_falls_back_to_the_primary_artist(self):
        sp = FakeSpotify([{"name": "Magic System", "genres": ["coupe decale"]}])
        self.assertEqual(
            get_spotify_artist_genres(sp, "Magic System, Ahmed Chawki"),
            ["coupe decale"],
        )

    def test_query_is_quoted_first(self):
        sp = FakeSpotify([])
        get_spotify_artist_genres(sp, "Placebo")
        self.assertEqual(sp.queries[0], 'artist:"Placebo"')

    def test_empty_genres_do_not_count_as_a_match(self):
        sp = FakeSpotify([{"name": "Cher", "genres": []}])
        self.assertEqual(get_spotify_artist_genres(sp, "Cher"), [])


if __name__ == "__main__":
    unittest.main()
