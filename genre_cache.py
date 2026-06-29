"""Persistent, two-level genre cache.

Genre lookups are slow (many third-party HTTP calls) but almost perfectly
deterministic per album/artist, so caching them across runs turns a 15-20 min
enrichment pass into a few seconds on subsequent runs.

Two levels:
  * L1 — an in-process dict (fast, lives for the run).
  * L2 — a persistent store: Redis by default, automatically falling back to a
    JSON file when Redis is unreachable (a warning is printed, the run goes on).

Values keep the resolved genres *and* their ``source`` so the CSV's ``source``
column is preserved across cached runs. Negative results (no genre found) are
also cached, but with a short TTL so they get retried before long.
"""

import json
import os
import sys
import tempfile
import time

DEFAULT_REDIS_URL = "redis://localhost:6379/0"
DEFAULT_TTL_DAYS = 90
DEFAULT_NEGATIVE_TTL_HOURS = 6


def _default_file_path():
    return os.path.join(
        os.path.expanduser("~"), ".cache", "likes_songs_sorter", "genre_cache.json"
    )


def _slug(value):
    return " ".join(str(value or "").strip().lower().split())


def make_key(album_id, album_name, artist_name):
    """Build a stable, secret-free cache key for an album/artist lookup."""
    if album_id:
        return f"genre:album:{album_id}"
    return f"genre:name:{_slug(album_name)}|{_slug(artist_name)}"


class GenreCache:
    """Two-level (memory + Redis/file) cache for album/artist genres."""

    def __init__(self, backend="redis", redis_url=DEFAULT_REDIS_URL, file_path=None,
                 ttl_days=DEFAULT_TTL_DAYS, negative_ttl_hours=DEFAULT_NEGATIVE_TTL_HOURS,
                 refresh=False, time_fn=time.time):
        self.refresh = refresh
        self._time = time_fn
        self._mem = {}
        self._redis = None
        self._file_path = None
        self._file_data = None
        self.ttl = int(ttl_days * 86400)
        self.negative_ttl = int(negative_ttl_hours * 3600)

        requested = (backend or "none").strip().lower()
        if requested == "none":
            self.backend = "none"
            return

        if requested == "redis":
            if self._try_init_redis(redis_url):
                self.backend = "redis"
                return
            print("⚠️ Redis cache unavailable; falling back to file cache.", file=sys.stderr)
            requested = "file"

        if requested == "file":
            self._init_file(file_path)
            self.backend = "file"
            return

        print(f"⚠️ Unknown cache backend '{backend}'; caching disabled.", file=sys.stderr)
        self.backend = "none"

    # --- backend setup --------------------------------------------------------
    def _try_init_redis(self, redis_url):
        try:
            import redis
            client = redis.from_url(
                redis_url, socket_connect_timeout=1, socket_timeout=1
            )
            client.ping()
            self._redis = client
            return True
        except Exception as exc:
            print(f"⚠️ Could not connect to Redis ({exc}).", file=sys.stderr)
            return False

    def _init_file(self, file_path):
        self._file_path = file_path or _default_file_path()
        try:
            os.makedirs(os.path.dirname(self._file_path), exist_ok=True)
        except OSError:
            pass
        self._file_data = {}
        if os.path.exists(self._file_path):
            try:
                with open(self._file_path, "r", encoding="utf-8") as fh:
                    loaded = json.load(fh)
                if isinstance(loaded, dict):
                    self._file_data = loaded
            except (OSError, ValueError):
                self._file_data = {}

    # --- public API -----------------------------------------------------------
    @property
    def enabled(self):
        return self.backend != "none"

    def get(self, key):
        """Return ``(genres, source)`` if cached and unexpired, else ``None``."""
        if key in self._mem:
            return self._mem[key]
        if not self.enabled or self.refresh:
            return None
        record = self._l2_get(key)
        if record is None:
            return None
        value = (list(record.get("genre") or []), record.get("source") or "None")
        self._mem[key] = value
        return value

    def set(self, key, genres, source):
        """Store a lookup result in L1 and L2 (negatives get a short TTL)."""
        genres = list(genres or [])
        value = (genres, source)
        self._mem[key] = value
        if not self.enabled:
            return
        ttl = self.ttl if genres else self.negative_ttl
        record = {"genre": genres, "source": source, "ts": int(self._time())}
        self._l2_set(key, record, ttl)

    def close(self):
        """Flush the file backend (Redis persists on every write)."""
        if self.backend == "file":
            self._flush_file()

    # --- L2: redis ------------------------------------------------------------
    def _l2_get(self, key):
        if self._redis is not None:
            try:
                raw = self._redis.get(key)
            except Exception:
                return None
            if not raw:
                return None
            try:
                return json.loads(raw)
            except ValueError:
                return None
        if self._file_data is not None:
            return self._file_get(key)
        return None

    def _l2_set(self, key, record, ttl):
        if self._redis is not None:
            try:
                self._redis.set(key, json.dumps(record), ex=ttl if ttl > 0 else None)
            except Exception:
                pass
            return
        if self._file_data is not None:
            record = dict(record)
            record["ttl"] = ttl
            self._file_data[key] = record
            self._flush_file()

    # --- L2: file -------------------------------------------------------------
    def _file_get(self, key):
        record = self._file_data.get(key)
        if not isinstance(record, dict):
            return None
        ttl = record.get("ttl")
        ts = record.get("ts")
        if ttl and ts is not None and self._time() - ts > ttl:
            # Expired: drop it so it gets retried.
            self._file_data.pop(key, None)
            return None
        return record

    def _flush_file(self):
        if self._file_path is None or self._file_data is None:
            return
        try:
            directory = os.path.dirname(self._file_path) or "."
            os.makedirs(directory, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(self._file_data, fh)
            os.replace(tmp, self._file_path)
        except OSError as exc:
            print(f"⚠️ Could not write genre cache file ({exc}).", file=sys.stderr)


def build_cache_from_config(config, refresh=False, disabled=False):
    """Construct a :class:`GenreCache` from a ``[CACHE]`` settings section."""
    if disabled:
        return GenreCache(backend="none")
    backend = config.get("CACHE", "backend", fallback="redis")
    redis_url = config.get("CACHE", "redis_url", fallback=DEFAULT_REDIS_URL)
    file_path = config.get("CACHE", "file_path", fallback=None) or None
    ttl_days = float(config.get("CACHE", "ttl_days", fallback=DEFAULT_TTL_DAYS))
    negative_ttl_hours = float(
        config.get("CACHE", "negative_ttl_hours", fallback=DEFAULT_NEGATIVE_TTL_HOURS)
    )
    return GenreCache(
        backend=backend,
        redis_url=redis_url,
        file_path=file_path,
        ttl_days=ttl_days,
        negative_ttl_hours=negative_ttl_hours,
        refresh=refresh,
    )
