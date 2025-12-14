import requests
from difflib import SequenceMatcher
import re
from collections import OrderedDict, Counter
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from bs4 import BeautifulSoup
from urllib.parse import quote_plus

BLACKLIST = {"wrong tag", "incorrect tag"}

def clean_tags(tags):
    return [t for t in tags if t.lower() not in BLACKLIST]

def clean_album_name(name):
    """
    Remove common parenthetical qualifiers from album names
    (e.g., Deluxe, Remaster, Edition, Anniversary, Reissue).
    """
    return re.sub(
        r"\s*\([^)]*(deluxe|remaster(ed)?|edition|anniversary|reissue)[^)]*\)",
        "",
        name,
        flags=re.IGNORECASE
    ).strip()

def get_discogs_album_info(album_name, artist_name, api_key, max_results=5):
    """
    Fetch genres/styles from Discogs via master or release record.
    """
    try:
        search_url = "https://api.discogs.com/database/search"
        params = {
            "release_title": album_name,
            "artist": artist_name,
            "type": "release",
            "token": api_key,
            "per_page": max_results
        }
        r = requests.get(search_url, params=params, timeout=5)
        results = r.json().get("results", [])
        for result in results:
            # try master record first
            master_id = result.get("master_id")
            if master_id:
                murl = f"https://api.discogs.com/masters/{master_id}"
                mr = requests.get(murl, params={"token": api_key}, timeout=5)
                mdata = mr.json()
                genres = mdata.get("genres", []) + mdata.get("styles", [])
                if genres:
                    return clean_tags(genres)
            # fallback to release record
            title = result.get("title", "").lower()
            score = SequenceMatcher(
                None, title, f"{artist_name} - {album_name}".lower()
            ).ratio()
            if score > 0.5:
                genres = result.get("genre", []) + result.get("style", [])
                if genres:
                    return clean_tags(genres)
    except Exception:
        pass
    return []

def get_lastfm_album_info(album_name, artist_name, api_key):
    """
    Retrieve album tags from Last.fm; return empty if unavailable.
    """
    try:
        url = "http://ws.audioscrobbler.com/2.0/"
        params = {
            "method": "album.getInfo",
            "api_key": api_key,
            "artist": artist_name,
            "album": album_name,
            "format": "json"
        }
        data = requests.get(url, params=params, timeout=5).json()
        if data.get("error"):
            return []
        album = data.get("album")
        if not isinstance(album, dict):
            return []
        tags = album.get("tags", {}).get("tag", [])
        return clean_tags([t.get("name", "") for t in tags if isinstance(t, dict)])
    except Exception:
        pass
    return []

def get_musicbrainz_album_info(album_name, artist_name, max_results=5):
    """
    Query MusicBrainz release-groups for genre tags.
    """
    try:
        url = "https://musicbrainz.org/ws/2/release-group/"
        params = {
            "query": f'release:"{album_name}" AND artist:"{artist_name}"',
            "fmt": "json",
            "limit": max_results
        }
        r = requests.get(url, params=params, timeout=5,
                         headers={"User-Agent": "SpotifySorter/1.0"})
        groups = r.json().get("release-groups", [])
        for grp in groups:
            title = grp.get("title", "").lower()
            score = SequenceMatcher(None, title, album_name.lower()).ratio()
            if score > 0.5:
                tags = [t.get("name", "") for t in grp.get("tags", [])]
                return clean_tags(tags)
    except Exception:
        pass
    return []

def get_lastfm_track_info(song_name, artist_name, api_key):
    """
    Retrieve top tags for a track from Last.fm.
    """
    try:
        url = "http://ws.audioscrobbler.com/2.0/"
        params = {
            "method": "track.getInfo",
            "api_key": api_key,
            "artist": artist_name,
            "track": song_name,
            "format": "json"
        }
        data = requests.get(url, params=params, timeout=5).json()
        if data.get("error"):
            return []
        tags = data.get("track", {}).get("toptags", {}).get("tag", [])
        return clean_tags([t.get("name", "") for t in tags if isinstance(t, dict)])
    except Exception:
        pass
    return []

def get_spotify_album_info(sp, album_id):
    """
    Aggregate genres from the album’s artists on Spotify.
    """
    try:
        alb = sp.album(album_id)
        genres = []
        for art in alb.get("artists", []):
            a = sp.artist(art.get("id"))
            genres.extend(a.get("genres", []))
        return clean_tags(genres)
    except Exception:
        pass
    return []

def get_wikipedia_album_info(album_name, artist_name):
    """
    Scrape album infobox on Wikipedia for the Genre field.
    """
    try:
        slug = quote_plus(f"{album_name} {artist_name}")
        url = f"https://en.wikipedia.org/wiki/{slug}"
        resp = requests.get(
            url,
            timeout=4,
            headers={"User-Agent": "SpotifySorter/1.0 (+github.com/spotify-likes-songs-sorter)"}
        )
        if resp.status_code >= 400:
            return []
        soup = BeautifulSoup(resp.text, "html.parser")
        info = soup.find("table", class_="infobox")
        if info:
            th = info.find("th", string="Genre")
            if th:
                td = th.find_next_sibling("td")
                tags = [a.get_text(strip=True) for a in td.find_all("a")]
                return clean_tags(tags)
    except Exception:
        pass
    return []

def get_itunes_album_info(album_name, artist_name):
    """
    Query Apple iTunes Search for lightweight genre hints.
    """
    try:
        params = {
            "term": f"{album_name} {artist_name}",
            "entity": "album",
            "media": "music",
            "limit": 3,
        }
        data = requests.get("https://itunes.apple.com/search", params=params, timeout=4).json()
        for item in data.get("results", []):
            if item.get("collectionType") != "Album":
                continue
            genres = []
            if item.get("primaryGenreName"):
                genres.append(item["primaryGenreName"])
            genres.extend(item.get("genres", []))
            if genres:
                unique = []
                for g in genres:
                    if g and g not in unique:
                        unique.append(g)
                return clean_tags(unique)
    except Exception:
        pass
    return []

def get_spotify_artist_genres(sp, artist_name):
    """
    Fetch genres directly from the artist record on Spotify.
    """
    try:
        res = sp.search(q=f"artist:{artist_name}", type="artist", limit=1)
        items = res.get("artists", {}).get("items", [])
        if items:
            return clean_tags(items[0].get("genres", []))
    except Exception:
        pass
    return []


def get_spotify_track_artist_genres(sp, track_id):
    """
    Fetch genres from the artists attached to a specific track.
    """
    try:
        track = sp.track(track_id)
        genres = []
        for artist in track.get("artists", []):
            art = sp.artist(artist.get("id"))
            genres.extend(art.get("genres", []))
        return clean_tags(genres)
    except Exception:
        pass
    return []

def lookup_genres(artist, album, song, album_id, cfg):
    """
    Run each service in turn and return an OrderedDict of their results.
    """
    sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(
        client_id=cfg["SPOTIFY"]["CLIENT_ID"],
        client_secret=cfg["SPOTIFY"]["CLIENT_SECRET"]
    ))
    return OrderedDict([
        ("Discogs",      get_discogs_album_info(album, artist, cfg["DISCOGS"]["API_KEY"])),
        ("LastFM Album", get_lastfm_album_info(album, artist, cfg["LASTFM"]["API_KEY"])),
        ("MusicBrainz",  get_musicbrainz_album_info(album, artist)),
        ("LastFM Track", get_lastfm_track_info(song, artist, cfg["LASTFM"]["API_KEY"])),
        ("Spotify Album",get_spotify_album_info(sp, album_id)),
        ("Wikipedia",    get_wikipedia_album_info(album, artist)),
        ("Spotify Artist", get_spotify_artist_genres(sp, artist)),
        ("iTunes",       get_itunes_album_info(album, artist)),
    ])

def normalize_and_sort_genres(genre_lists):
    """
    Title-case genre tags and sort each album’s tags by descending
    global frequency (broad → niche).
    """
    cleaned = [[g.strip().lower().title() for g in sub] for sub in genre_lists]
    counts = Counter(tag for sub in cleaned for tag in sub)
    return [sorted(sub, key=lambda t: counts[t], reverse=True) for sub in cleaned]

