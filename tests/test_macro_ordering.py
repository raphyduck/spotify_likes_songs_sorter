"""Ordre macro des familles : 2-opt/Or-opt et insertion des singletons.

Avant ces changements, l'ordre des familles etait un glouton du plus proche
voisin (29 % de la proximite atteignable exploitee) et les familles a un seul
album finissaient dans une queue ALPHABETIQUE en fin de playlist.
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sorter_core
from genre_normalization import load_genre_roots


def path_score(order, sim):
    return sum(sim[a, b] for a, b in zip(order, order[1:]))


class ImprovePathTest(unittest.TestCase):
    def test_2opt_fixes_a_bad_greedy_chain(self):
        # 6 points sur une ligne : le chemin optimal est 0-1-2-3-4-5.
        # On part d'un ordre volontairement croise.
        pos = np.array([0, 1, 2, 3, 4, 5], dtype=float)
        sim = 1.0 / (1.0 + np.abs(pos[:, None] - pos[None, :]))
        np.fill_diagonal(sim, 0.0)
        bad = [0, 3, 1, 4, 2, 5]
        got = sorter_core._improve_path_order(bad, sim)
        self.assertGreaterEqual(path_score(got, sim), path_score(bad, sim))
        self.assertEqual(path_score(got, sim), path_score([0, 1, 2, 3, 4, 5], sim))

    def test_never_degrades(self):
        rng = np.random.RandomState(7)
        for _ in range(5):
            sim = rng.rand(9, 9)
            sim = (sim + sim.T) / 2
            np.fill_diagonal(sim, 0.0)
            start = list(range(9))
            got = sorter_core._improve_path_order(start, sim)
            self.assertGreaterEqual(path_score(got, sim) + 1e-9, path_score(start, sim))
            self.assertEqual(sorted(got), start)  # permutation, rien perdu

    def test_deterministic(self):
        rng = np.random.RandomState(3)
        sim = rng.rand(8, 8)
        sim = (sim + sim.T) / 2
        np.fill_diagonal(sim, 0.0)
        a = sorter_core._improve_path_order(list(range(8)), sim)
        b = sorter_core._improve_path_order(list(range(8)), sim)
        self.assertEqual(a, b)


class InsertSingletonsTest(unittest.TestCase):
    def test_singleton_lands_next_to_its_closest_family(self):
        # familles 0-1-2 en chaine ; le singleton 3 est tres proche de 1.
        sim = np.array([
            [0.0, 0.8, 0.1, 0.1],
            [0.8, 0.0, 0.8, 0.9],
            [0.1, 0.8, 0.0, 0.1],
            [0.1, 0.9, 0.1, 0.0],
        ])
        got = sorter_core._insert_singletons([0, 1, 2], [3], sim)
        i = got.index(3)
        self.assertIn(1, [got[j] for j in (i - 1, i + 1) if 0 <= j < len(got)])

    def test_related_singletons_chain_together(self):
        # 0 est la seule grande famille ; 1 et 2 sont des singletons tres
        # proches l'un de l'autre (Kizomba / Kompa) et un peu proches de 0.
        sim = np.array([
            [0.0, 0.3, 0.25],
            [0.3, 0.0, 0.9],
            [0.25, 0.9, 0.0],
        ])
        got = sorter_core._insert_singletons([0], [1, 2], sim)
        self.assertEqual(abs(got.index(1) - got.index(2)), 1)

    def test_all_indices_kept(self):
        rng = np.random.RandomState(1)
        sim = rng.rand(10, 10)
        sim = (sim + sim.T) / 2
        got = sorter_core._insert_singletons([0, 1, 2], [3, 4, 5, 6, 7, 8, 9], sim)
        self.assertEqual(sorted(got), list(range(10)))


class NoAlphabeticalTailTest(unittest.TestCase):
    def test_singleton_families_are_not_dumped_at_the_end(self):
        rules = load_genre_roots()
        rows = []
        # Deux grandes familles (2 albums chacune) + un singleton proche de
        # la premiere. L'ancien code l'aurait mis en dernier (queue).
        albums = [
            ("p1", "A", ["Pop Punk", "Punk"]),
            ("p2", "B", ["Skate Punk", "Punk"]),
            ("j1", "C", ["Bebop", "Jazz"]),
            ("j2", "D", ["Swing", "Jazz"]),
            ("solo", "E", ["Punk Blues"]),  # famille de repli a 1 album
        ]
        for uid, artist, genres in albums:
            rows.append({"Unique Album": uid, "Album": uid, "Artist": artist,
                         "Album Genre": genres, "Album ID": uid})
        df = pd.DataFrame(rows)
        ordering, _ = sorter_core._order_albums(df, 0.6, 10, 2.0, rules,
                                                ordering_mode="two_level")
        names = list(ordering.sort_values("Sort Order")["Unique Album"])
        i = names.index("solo")
        neighbours = {names[j] for j in (i - 1, i + 1) if 0 <= j < len(names)}
        self.assertTrue(neighbours & {"p1", "p2"},
                        f"solo devrait voisiner le bloc punk, ordre: {names}")


if __name__ == "__main__":
    unittest.main()
