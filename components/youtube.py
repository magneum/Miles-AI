import requests
import spacy
import json
import re

nlp = spacy.load("en_core_web_sm")
single_track_endpoint = "http://localhost:3000/singleTrack?songName={}"
random_track_endpoint = "http://localhost:3000/randomTrack"
custom_track_endpoint = "http://localhost:3000/customTrack?query={}&totalTracks={}"
random_genre_track_endpoint = "http://localhost:3000/randomGenreTrack?genre={}"
music_genres = {
    "pop": ["pop", "popular", "mainstream"],
    "rock": ["rock", "rock n roll"],
    "hip hop": ["hip hop", "rap", "urban"],
    "jazz": ["jazz", "smooth"],
    "blues": ["blues"],
    "classical": ["classical"],
    "country": ["country", "western"],
    "reggae": ["reggae"],
    "electronic": ["electronic", "edm"],
    "metal": ["metal"],
    "punk": ["punk"],
    "folk": ["folk"],
    "r&b": ["r&b", "rhythm and blues"],
    "latin": ["latin"],
    "indie": ["indie", "independent"],
}


def handle_single_track_request(song_name):
    endpoint = single_track_endpoint.format(song_name)
    response = requests.get(endpoint)
    if response.status_code == 200:
        video = json.loads(response.text)
        return video["url"]
    else:
        return None


def handle_random_track_request():
    response = requests.get(random_track_endpoint)
    if response.status_code == 200:
        video = json.loads(response.text)
        return video["url"]
    else:
        return None


def handle_random_genre_track_request(genre):
    endpoint = random_genre_track_endpoint.format(genre)
    response = requests.get(endpoint)
    if response.status_code == 200:
        video = json.loads(response.text)
        return video["url"]
    else:
        return None


def handle_custom_track_request(query, total_tracks):
    endpoint = custom_track_endpoint.format(query, total_tracks)
    response = requests.get(endpoint)
    if response.status_code == 200:
        tracks = json.loads(response.text)
        return [track["url"] for track in tracks]
    else:
        return None


def handle_user_input(user_input):
    doc = nlp(user_input)
    for token in doc:
        if token.ent_type_ == "MONEY" or token.like_num:
            numbers = re.findall("\d+", user_input)
            if numbers:
                query = re.sub("\d+", "", user_input).strip()
                total_tracks = int(numbers[0])
                video_urls = []
                for i in range(total_tracks):
                    video_url = handle_custom_track_request(query)
                    if video_url:
                        video_urls.append(video_url)
                return video_urls
        elif token.lemma_ in ["play", "suggest", "listen"]:
            genre = None
            for key, values in music_genres.items():
                for value in values:
                    if value in user_input:
                        genre = key
                        break
                if genre:
                    break
            if genre:
                return handle_random_genre_track_request(genre)
            else:
                return handle_random_track_request()
    return handle_single_track_request(user_input)


while True:
    user_input = input("What would you like to listen to? ")
    if user_input.lower() in ["quit", "exit"]:
        print("Goodbye!")
        break
    video_url = handle_user_input(user_input)
    if video_url:
        print(f"Playing {video_url}")
    else:
        print("Sorry, I could not find")
