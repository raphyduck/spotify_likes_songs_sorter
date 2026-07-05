import hashlib
import unittest
from unittest.mock import MagicMock, patch

from backends import SubsonicBackend


def _make_backend():
    backend = SubsonicBackend()
    backend.base_url = "https://navidrome.example.com"
    backend.username = "user"
    backend.password = "secret"
    backend._discogs_key = "d"
    backend._lastfm_key = "l"
    return backend


def _mock_response(payload):
    resp = MagicMock()
    resp.json.return_value = {"subsonic-response": payload}
    return resp


class AuthParamsTest(unittest.TestCase):
    def test_salt_differs_between_calls(self):
        backend = _make_backend()
        first = backend._auth_params()
        second = backend._auth_params()
        self.assertNotEqual(first["s"], second["s"])
        self.assertNotEqual(first["t"], second["t"])

    def test_token_is_salted_password_hash(self):
        backend = _make_backend()
        params = backend._auth_params()
        expected = hashlib.md5(
            ("secret" + params["s"]).encode("utf-8")
        ).hexdigest()
        self.assertEqual(params["t"], expected)
        self.assertEqual(params["u"], "user")
        self.assertEqual(params["f"], "json")

    def test_error_payload_raises(self):
        backend = _make_backend()
        error = {"status": "failed",
                 "error": {"code": 40, "message": "Wrong username or password"}}
        with patch("requests.get", return_value=_mock_response(error)):
            with self.assertRaises(RuntimeError):
                backend._request("ping")


class GetLikedSongsTest(unittest.TestCase):
    def test_maps_getstarred2_response(self):
        backend = _make_backend()
        payload = {
            "status": "ok",
            "starred2": {
                "song": [
                    {
                        "id": "tr-1",
                        "title": "Paranoid Android",
                        "artist": "Radiohead",
                        "album": "OK Computer",
                        "albumId": "al-1",
                        "track": 2,
                        "discNumber": 1,
                        "genre": "Alternative Rock",
                    },
                    {"id": "tr-2"},  # everything missing -> fallback labels
                ]
            },
        }
        with patch("requests.get", return_value=_mock_response(payload)):
            rows = backend.get_liked_songs()

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0], {
            "Song": "Paranoid Android",
            "Artist": "Radiohead",
            "Album": "OK Computer",
            "Album ID": "al-1",
            "Track Number": 2,
            "Disc Number": 1,
            "Subsonic Track ID": "tr-1",
        })
        self.assertEqual(rows[1]["Song"], "Unknown Song")
        self.assertEqual(rows[1]["Artist"], "Unknown Artist")
        self.assertEqual(rows[1]["Album"], "Unknown Album")
        # The file's genre tag is stashed for the Local Tag provider.
        self.assertEqual(backend._genre_by_track, {"tr-1": ["Alternative Rock"]})

    def test_multi_genre_tags_are_split(self):
        song = {"id": "tr-3", "genre": "Rock; Post-Punk / New Wave"}
        self.assertEqual(
            SubsonicBackend._song_genres(song),
            ["Rock", "Post-Punk", "New Wave"],
        )

    def test_opensubsonic_genres_list_wins(self):
        song = {"id": "tr-4", "genre": "Rock",
                "genres": [{"name": "Shoegaze"}, {"name": "Dream Pop"}]}
        self.assertEqual(
            SubsonicBackend._song_genres(song), ["Shoegaze", "Dream Pop"]
        )


class ProviderOrderTest(unittest.TestCase):
    def test_local_tag_first_then_name_based_chain(self):
        backend = _make_backend()
        backend._genre_by_track["tr-1"] = ["Alternative Rock"]
        providers = backend.get_genre_providers(
            "song", "artist", "album", "album", "al-1", "tr-1", None
        )
        self.assertEqual(
            [name for name, _ in providers],
            ["Local Tag", "Discogs", "LastFM Album", "MusicBrainz",
             "LastFM Track", "Wikipedia", "iTunes"],
        )
        # The first provider serves the local tag with zero network calls.
        self.assertEqual(providers[0][1](), ["Alternative Rock"])

    def test_local_tag_empty_falls_through(self):
        backend = _make_backend()
        providers = backend.get_genre_providers(
            "song", "artist", "album", "album", "al-1", "tr-unknown", None
        )
        self.assertEqual(providers[0][0], "Local Tag")
        self.assertIsNone(providers[0][1]())


class AddTracksTest(unittest.TestCase):
    def test_chunks_and_repeats_song_id_param(self):
        backend = _make_backend()
        rows = [{"Subsonic Track ID": f"tr-{i}"} for i in range(250)]
        rows.append({"Subsonic Track ID": None})  # dropped
        ok = _mock_response({"status": "ok"})
        with patch("requests.get", return_value=ok) as mock_get:
            added, local = backend.add_tracks({"id": "pl-1", "name": "x"}, rows)

        self.assertEqual((added, local), (250, 0))
        self.assertEqual(mock_get.call_count, 2)  # 200 + 50
        params = mock_get.call_args_list[0].kwargs["params"]
        song_ids = [v for k, v in params if k == "songIdToAdd"]
        self.assertEqual(len(song_ids), 200)
        self.assertEqual(dict(params)["playlistId"], "pl-1")


if __name__ == "__main__":
    unittest.main()
