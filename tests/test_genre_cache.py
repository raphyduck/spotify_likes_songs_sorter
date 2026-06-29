import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from genre_cache import GenreCache, make_key


class FakeRedis:
    """Minimal in-memory stand-in for a redis client."""

    def __init__(self):
        self.store = {}

    def ping(self):
        return True

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value, ex=None):
        self.store[key] = value


class MakeKeyTest(unittest.TestCase):
    def test_album_id_preferred_and_secret_free(self):
        self.assertEqual(make_key("abc123", "Album", "Artist"), "genre:album:abc123")

    def test_name_fallback_is_normalized(self):
        self.assertEqual(
            make_key(None, "  In   Rainbows ", "Radiohead"),
            "genre:name:in rainbows|radiohead",
        )


class FileCacheTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "genre_cache.json")
        self.addCleanup(self.tmp.cleanup)

    def test_set_get_roundtrip_and_persistence(self):
        cache = GenreCache(backend="file", file_path=self.path)
        key = make_key("alb1", "A", "Artist")
        cache.set(key, ["Rock", "Indie Rock"], "Discogs")
        cache.close()

        # Brand new instance (cold L1) must read the value back from disk.
        cache2 = GenreCache(backend="file", file_path=self.path)
        self.assertEqual(cache2.get(key), (["Rock", "Indie Rock"], "Discogs"))

    def test_negative_result_roundtrip_and_short_ttl(self):
        clock = [1000.0]
        cache = GenreCache(
            backend="file", file_path=self.path,
            negative_ttl_hours=1, time_fn=lambda: clock[0],
        )
        key = make_key("alb2", "A", "Artist")
        cache.set(key, [], "None")
        cache.close()

        cache2 = GenreCache(backend="file", file_path=self.path, time_fn=lambda: clock[0])
        self.assertEqual(cache2.get(key), ([], "None"))

        # After the 1h negative TTL elapses, the entry expires and is retried.
        clock[0] += 3601
        cache3 = GenreCache(backend="file", file_path=self.path, time_fn=lambda: clock[0])
        self.assertIsNone(cache3.get(key))

    def test_refresh_ignores_stored_value_but_still_writes(self):
        cache = GenreCache(backend="file", file_path=self.path)
        key = make_key("alb3", "A", "Artist")
        cache.set(key, ["Jazz"], "LastFM Album")
        cache.close()

        refresh = GenreCache(backend="file", file_path=self.path, refresh=True)
        self.assertIsNone(refresh.get(key))  # ignores L2 on read
        refresh.set(key, ["Bebop"], "MusicBrainz")
        refresh.close()

        verify = GenreCache(backend="file", file_path=self.path)
        self.assertEqual(verify.get(key), (["Bebop"], "MusicBrainz"))


class RedisCacheTest(unittest.TestCase):
    def test_uses_redis_when_reachable(self):
        fake = FakeRedis()
        with patch("redis.from_url", return_value=fake):
            cache = GenreCache(backend="redis")
        self.assertEqual(cache.backend, "redis")
        key = make_key("alb", "A", "Artist")
        cache.set(key, ["Rock"], "Discogs")
        # Stored in the fake redis, retrievable from a fresh-L1 instance.
        with patch("redis.from_url", return_value=fake):
            cache2 = GenreCache(backend="redis")
        self.assertEqual(cache2.get(key), (["Rock"], "Discogs"))

    def test_falls_back_to_file_when_redis_down(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = os.path.join(tmp.name, "c.json")
        with patch("redis.from_url", side_effect=ConnectionError("nope")):
            cache = GenreCache(backend="redis", file_path=path)
        self.assertEqual(cache.backend, "file")
        key = make_key("alb", "A", "Artist")
        cache.set(key, ["Rock"], "Discogs")
        cache.close()
        self.assertTrue(os.path.exists(path))


class NoCacheTest(unittest.TestCase):
    def test_disabled_backend_never_persists(self):
        cache = GenreCache(backend="none")
        self.assertFalse(cache.enabled)
        key = make_key("alb", "A", "Artist")
        cache.set(key, ["Rock"], "Discogs")
        # L1 still serves within a run, but a fresh instance has nothing.
        self.assertEqual(cache.get(key), (["Rock"], "Discogs"))
        self.assertIsNone(GenreCache(backend="none").get(key))


class ResolverIntegrationTest(unittest.TestCase):
    def test_providers_called_once_across_runs(self):
        import sorter_core

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = os.path.join(tmp.name, "c.json")

        calls = {"n": 0}

        def provider():
            calls["n"] += 1
            return ["Rock"]

        backend = MagicMock()
        backend.get_genre_providers.return_value = [("Discogs", provider)]

        cache1 = GenreCache(backend="file", file_path=path)
        resolve1 = sorter_core._make_genre_resolver(backend, {}, cache1)
        self.assertEqual(resolve1("s", "Artist", "Album", "alb", None), (["Rock"], "Discogs"))
        cache1.close()

        cache2 = GenreCache(backend="file", file_path=path)
        resolve2 = sorter_core._make_genre_resolver(backend, {}, cache2)
        self.assertEqual(resolve2("s", "Artist", "Album", "alb", None), (["Rock"], "Discogs"))
        cache2.close()

        self.assertEqual(calls["n"], 1)  # second run served from the file cache


if __name__ == "__main__":
    unittest.main()
