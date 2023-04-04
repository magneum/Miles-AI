import spacy
import requests

BASE_URL = "http://localhost:3000"

ENDPOINTS = {
    "searchVideo": "/searchVideo",
    "getVideoDetails": "/getVideoDetails",
    "getPlaylistDetails": "/getPlaylistDetails",
    "searchVideosByGenre": "/searchVideosByGenre",
    "getVideosFromChannel": "/getVideosFromChannel",
    "getVideoDetailsByUrl": "/getVideoDetailsByUrl",
    "getRelatedVideos": "/getRelatedVideos",
}

nlp = spacy.load("en_core_web_sm")


def extract_entities(query):
    doc = nlp(query)
    entities = {}
    for ent in doc.ents:
        entities[ent.label_] = ent.text
    return entities


def handle_query(query):
    entities = extract_entities(query)
    if "TRACK_GENRE" in entities and "TRACK_COUNT" in entities:
        url = BASE_URL + ENDPOINTS["searchVideosByGenre"]
        params = {
            "genre": entities["TRACK_GENRE"],
            "numVideos": entities["TRACK_COUNT"],
        }
        response = requests.get(url, params=params)
    elif "TRACK_GENRE" in entities:
        url = BASE_URL + ENDPOINTS["searchVideosByGenre"]
        params = {"genre": entities["TRACK_GENRE"], "numVideos": 1}
        response = requests.get(url, params=params)
    elif "TRACK_ID" in entities:
        url = BASE_URL + ENDPOINTS["getVideoDetails"]
        params = {"videoId": entities["TRACK_ID"]}
        response = requests.get(url, params=params)
    else:
        response = {"message": "I'm sorry, I didn't understand your request."}
    return response


query = input("What would you like me to do? ")
response = handle_query(query)
if response.status_code == 200:
    data = response.json()
    print(data)
else:
    print(f"Error: {response.status_code}")
