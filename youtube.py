import requests

base_url = "http://localhost:3000/"


def search_videos(keyword, num_videos):
    url = base_url + "searchVideo"
    payload = {"keyword": keyword, "numVideos": num_videos}
    response = requests.post(url, json=payload)
    return response.json()


# def get_video_details(video_id):
#     url = base_url + "getVideoDetails"
#     payload = {"videoId": video_id}
#     response = requests.post(url, json=payload)
#     return response.json()


# def search_videos_by_genre(genre, num_videos):
#     url = base_url + "searchVideosByGenre"
#     payload = {"genre": genre, "numVideos": num_videos}
#     response = requests.post(url, json=payload)
#     return response.json()


# def search_topic_videos(topic, num_videos):
#     keyword = "How to " + topic
#     return search_videos(keyword, num_videos)


# def get_playlist_details(playlist_id, num_videos):
#     url = base_url + "getPlaylistDetails"
#     payload = {"playlistId": playlist_id, "numVideos": num_videos}
#     response = requests.post(url, json=payload)
#     return response.json()


# def get_related_videos(video_id, num_videos):
#     url = base_url + "getRelatedVideos"
#     payload = {"relatedToVideoId": video_id, "numVideos": num_videos}
#     response = requests.post(url, json=payload)
#     return response.json()


# def get_videos_from_channel(channel_id, num_videos):
#     url = base_url + "getVideosFromChannel"
#     payload = {"channelId": channel_id, "numVideos": num_videos}
#     response = requests.post(url, json=payload)
#     return response.json()


# def get_video_details_by_url(video_url):
#     url = base_url + "getVideoDetailsByUrl"
#     payload = {"url": video_url}
#     response = requests.post(url, json=payload)
#     return response.json()


print(search_videos("Taylor Swift", 10))
# print(get_video_details("dQw4w9WgXcQ"))
# print(search_videos_by_genre("Cooking", 10))
# print(search_topic_videos("meditate", 5))
# print(get_playlist_details("PLMC9KNkIncKtPzgY-5rmhvj7fax8fdxoj", 10))
# print(get_related_videos("dQw4w9WgXcQ", 10))
# print(get_videos_from_channel("UC-lHJZR3Gqxm24_Vd_AJ5Yw", 10))
# print(get_video_details_by_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ"))
