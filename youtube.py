import spacy
import requests

nlp = spacy.load("en_core_web_sm")  # python -m spacy download en_core_web_sm


base_url = "http://localhost:3000"
endpoints = {
    "searchVideo": "/search?keyword={}",
    "getVideoDetails": "/video?videoId={}",
    "searchVideosByGenre": "/search?genre={}",
    "getVideosFromChannel": "/search?channelId={}",
}
