import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock

from genre_overrides import load_overrides, lookup_override


def _write(tmp, payload):
    path = os.path.join(tmp, "overrides.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    return path


class OverrideLookupTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_accent_and_case_insensitive_artist_match(self):
        path = _write(self.tmp.name, {"overrides": [
            {"match": "Sigur Rós", "tags": ["Post-Rock"], "root": "rock"},
        ]})
        ov = load_overrides(path)
        # Queried without accent / different case still matches.
        entry = lookup_override(ov, "sigur ros", "Takk...")
        self.assertIsNotNone(entry)
        self.assertEqual(entry["root"], "rock")
        self.assertEqual(entry["tags"], ["Post-Rock"])

    def test_artist_album_key_precedence(self):
        path = _write(self.tmp.name, {"overrides": [
            {"match": "Various Artists", "root": "pop"},
            {"match": "Various Artists|Jazz Anthology", "root": "jazz"},
        ]})
        ov = load_overrides(path)
        self.assertEqual(lookup_override(ov, "Various Artists", "Jazz Anthology")["root"], "jazz")
        self.assertEqual(lookup_override(ov, "Various Artists", "Other")["root"], "pop")

    def test_missing_file_is_inactive(self):
        self.assertEqual(load_overrides(os.path.join(self.tmp.name, "nope.json")), {})
        self.assertIsNone(lookup_override({}, "X", "Y"))


class OverrideIntegrationTest(unittest.TestCase):
    def test_override_wins_over_providers_and_cache(self):
        import sorter_core
        from genre_cache import GenreCache

        provider = MagicMock(return_value=["Wrong Genre"])
        backend = MagicMock()
        backend.get_genre_providers.return_value = [("Discogs", provider)]
        overrides = {"sigur ros": {"tags": ["Post-Rock", "Ambient"], "root": "rock"}}

        cache = GenreCache(backend="none")
        resolve = sorter_core._make_genre_resolver(backend, {}, cache, overrides)
        genres, source = resolve("s", "Sigur Rós", "Takk...", "alb", None)

        self.assertEqual(source, "Override")
        self.assertEqual(genres, ["Post-Rock", "Ambient"])
        provider.assert_not_called()  # providers never consulted

    def test_album_root_uses_override(self):
        import sorter_core
        overrides = {"juanes": {"tags": [], "root": "latin"}}
        # Tags would infer "rock", but the override forces "latin".
        root = sorter_core._album_root(["Rock", "Pop Rock"], "Juanes", "Mi Sangre",
                                       rules=[], overrides=overrides)
        self.assertEqual(root, "latin")


if __name__ == "__main__":
    unittest.main()
