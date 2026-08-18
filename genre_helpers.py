import requests
from difflib import SequenceMatcher
import re
import unicodedata
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
                         headers={"User-Agent": "MusicSorter/1.0"})
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

def get_musicbrainz_artist_genres(artist_name, max_results=5):
    """
    Query MusicBrainz for an artist's genre tags (community-voted).

    Dernier recours de la voie "par artiste" : Spotify a vide le champ
    genres de beaucoup de gros artistes (U2, P!nk, Cher...) et les toptags
    LastFM de leurs pistes reviennent souvent vides. MusicBrainz, lui, garde
    des tags votes. Le nom du resultat est verifie (artist_hit_matches) pour
    ne pas retomber dans le piege des homonymes.
    """
    try:
        url = "https://musicbrainz.org/ws/2/artist/"
        params = {
            "query": f'artist:"{artist_name}"',
            "fmt": "json",
            "limit": max_results,
        }
        r = requests.get(url, params=params, timeout=6,
                         headers={"User-Agent": "MusicSorter/1.0"})
        for item in r.json().get("artists", []):
            if not artist_hit_matches(artist_name, item.get("name", "")):
                continue
            tags = sorted(item.get("tags", []),
                          key=lambda t: -int(t.get("count", 0) or 0))
            names = [t.get("name", "") for t in tags
                     if int(t.get("count", 0) or 0) > 0]
            if names:
                return clean_tags(names[:5])
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
    Return an album's genres on Spotify.

    Prefers the album's own ``genres`` field when present, otherwise falls
    back to aggregating the genres of the album's artists.
    """
    try:
        alb = sp.album(album_id)
        album_genres = clean_tags(alb.get("genres", []) or [])
        if album_genres:
            return album_genres
        genres = []
        for art in alb.get("artists", []):
            a = sp.artist(art.get("id"))
            genres.extend(a.get("genres", []))
        return clean_tags(genres)
    except Exception:
        pass
    return []

def get_spotify_album_search_info(sp, album_name, artist_name):
    """
    Resolve a Spotify album by name + artist, then return its genres.

    Useful for cross-service enrichment (e.g. Tidal) where no Spotify album
    id is available: we search Spotify by name to obtain one, then reuse
    :func:`get_spotify_album_info`.
    """
    try:
        query = f'album:"{album_name}" artist:"{artist_name}"'
        res = sp.search(q=query, type="album", limit=1)
        items = res.get("albums", {}).get("items", [])
        if not items:
            return []
        album_id = items[0].get("id")
        if album_id:
            return get_spotify_album_info(sp, album_id)
    except Exception:
        pass
    return []

# Separateurs de chaines multi-artistes ("Magic System, Ahmed Chawki",
# "Harris & Ford . 2 Engel & Charlie", "Ofenbach feat. Norma Jean Martine"...).
_ARTIST_SPLIT_RE = re.compile(
    r"\s*(?:[;,\u2022/\u00b7|]|\bfeat\.?\b|\bft\.?\b|\bwith\b|\bvs\.?\b)\s*",
    re.IGNORECASE,
)


def _strip_accents(value):
    return "".join(
        c for c in unicodedata.normalize("NFKD", str(value or ""))
        if not unicodedata.combining(c)
    )


def _norm_artist(value):
    """Cle de comparaison insensible aux accents, a la casse et a la ponctuation."""
    return re.sub(r"[^a-z0-9]+", "", _strip_accents(value).lower())


def _artist_words(value):
    return set(re.findall(r"[a-z0-9]+", _strip_accents(value).lower()))


def artist_query_variants(artist_name):
    """Candidats de recherche pour une chaine d'artiste, le principal d'abord.

    Spotify ne connait pas "Magic System, Ahmed Chawki" (zero resultat) mais
    connait "Magic System". On essaie donc la chaine complete, puis le premier
    artiste seul, puis l'inversion "Nom, Prenom" -> "Prenom Nom".
    """
    raw = str(artist_name or "").strip()
    if not raw:
        return []
    variants = [raw]
    parts = [p.strip() for p in _ARTIST_SPLIT_RE.split(raw) if p.strip()]
    if len(parts) > 1:
        variants.append(parts[0])
        if len(parts) == 2:
            variants.append(f"{parts[1]} {parts[0]}")
    seen, out = set(), []
    for v in variants:
        if v.lower() not in seen:
            seen.add(v.lower())
            out.append(v)
    return out


def artist_hit_matches(requested, found):
    """Le resultat Spotify porte-t-il vraiment le nom demande ?

    Tolere les accents, la ponctuation et l'inversion des mots
    ("Celine Dion"/"Celine Dion", "Neg\'Marrons"/"Neg\' Marrons",
    "Cohen, Leonard"/"Leonard Cohen"), mais refuse les homonymes
    ("Trust"/"Men I Trust", "Europe"/"D-Block Europe", "Haggard"/"Merle
    Haggard") et les artistes dont le nom contient simplement le mot
    "artist" ("ARTIST: UNKNOWN", "Bogdan Artistu", "Artist Music Video"),
    qui remontent des que le filtre "artist:" se degrade en recherche floue.
    """
    a, b = _norm_artist(requested), _norm_artist(found)
    if not a or not b:
        return False
    if a == b:
        return True
    wa, wb = _artist_words(requested), _artist_words(found)
    return bool(wa) and wa == wb


def get_spotify_artist_genres(sp, artist_name):
    """
    Fetch genres directly from the artist record on Spotify.

    La valeur est mise entre guillemets (sans quoi le filtre ``artist:`` se
    degrade en recherche floue) et le nom du resultat est verifie : prendre
    ``items[0]`` en aveugle attribuait par exemple le reggae de Bob Marley a
    t.A.T.u. et le country de Merle Haggard au metal symphonique de Haggard.
    Quand aucun resultat ne correspond, on renvoie ``[]`` pour que la chaine
    de fournisseurs retombe sur Discogs / LastFM / MusicBrainz.
    """
    for variant in artist_query_variants(artist_name):
        # Forme entre guillemets d'abord ; la forme nue ensuite, car sur les
        # noms ponctues ("A-Ha", "U2") les guillemets font eux aussi deraper la
        # recherche. La verification du nom rend la seconde tentative sure.
        for query in (f'artist:"{variant}"', f"artist:{variant}"):
            try:
                res = sp.search(q=query, type="artist", limit=5)
            except Exception:
                continue
            for item in res.get("artists", {}).get("items", []) or []:
                if not artist_hit_matches(variant, item.get("name")):
                    continue
                genres = clean_tags(item.get("genres", []) or [])
                if genres:
                    return genres
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
            headers={"User-Agent": "MusicSorter/1.0 (+github.com/likes-songs-sorter)"}
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

def lookup_genres(artist, album, song, cfg):
    """
    Run each service in turn and return an OrderedDict of their results.

    Genres come exclusively from name-based providers (Discogs, Last.fm,
    MusicBrainz, Wikipedia, iTunes) so the lookup stays independent of any
    particular streaming service. Tidal does not expose album/artist genre
    metadata through its API, so the enrichment relies on these sources.
    """
    providers = [
        ("Discogs", lambda: get_discogs_album_info(album, artist, cfg["DISCOGS"]["API_KEY"])),
        ("LastFM Album", lambda: get_lastfm_album_info(album, artist, cfg["LASTFM"]["API_KEY"])),
        ("MusicBrainz", lambda: get_musicbrainz_album_info(album, artist)),
        ("LastFM Track", lambda: get_lastfm_track_info(song, artist, cfg["LASTFM"]["API_KEY"])),
        ("Wikipedia", lambda: get_wikipedia_album_info(album, artist)),
        ("iTunes", lambda: get_itunes_album_info(album, artist)),
    ]
    return OrderedDict((name, lookup()) for name, lookup in providers)

def normalize_and_sort_genres(genre_lists):
    """
    Title-case genre tags and sort each album’s tags by descending
    global frequency (broad → niche).
    """
    cleaned = [[g.strip().lower().title() for g in sub] for sub in genre_lists]
    counts = Counter(tag for sub in cleaned for tag in sub)
    return [sorted(sub, key=lambda t: counts[t], reverse=True) for sub in cleaned]
