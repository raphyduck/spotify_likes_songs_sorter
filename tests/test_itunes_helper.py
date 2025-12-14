import unittest
from unittest.mock import Mock, patch

from genre_helpers import get_itunes_album_info


class ItunesHelperTest(unittest.TestCase):
    def test_parses_recorded_response(self):
        sample = {
            "resultCount": 1,
            "results": [
                {
                    "collectionType": "Album",
                    "collectionName": "In Between Dreams (Bonus Track Version)",
                    "artistName": "Jack Johnson",
                    "primaryGenreName": "Rock",
                    "genres": ["Rock", "Music"],
                }
            ],
        }
        dummy_response = Mock()
        dummy_response.json.return_value = sample
        with patch("genre_helpers.requests.get", return_value=dummy_response) as mock_get:
            genres = get_itunes_album_info("In Between Dreams", "Jack Johnson")
        self.assertEqual(genres, ["Rock", "Music"])
        mock_get.assert_called_once()


if __name__ == "__main__":
    unittest.main()
