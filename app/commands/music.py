import re
import requests
import pygame
from io import BytesIO
from urllib.parse import urlparse
import validators
import html
import logging

# regex patterns for various commands
regex_play = r"(play|start)( the)?\s?(song|music)?\s?(.*)"
regex_stop = r"(stop|end|finish|terminate)( the)?\s?(song|music)?"
regex_pause = r"pause( the)?\s?(song|music)?"
regex_resume = r"resume( the)?\s?(song|music)?"
regex_volume_up = r"volume (up|increase)"
regex_volume_down = r"volume (down|decrease)"
regex_specific_song = r"(play|start)( the)?\s?(song|music)?\s?(.*)\s(by|of)\s?(.*)"
regex_specific_artist = r"(play|start)( the)?\s?(song|music)?\s?(by|of)\s?(.*)"
regex_specific_album = r"(play|start)( the)?\s?(album)\s?(.*)"


# initialize logger
logger = logging.getLogger(__name__)


def handle_music_player(usersaid, audiourl):
    try:
        # initialize pygame mixer
        pygame.mixer.init()

        # sanitize user input
        usersaid = html.escape(usersaid).strip()

        # parse audio url
        parsed_audio_url = urlparse(audiourl)

        # check if user wants to play a specific song
        match = re.match(regex_specific_song, usersaid)
        if match:
            songname = match.group(1)
            artist = match.group(2)
            parsed_audio_url = urlparse(audiourl)
            url = f"{parsed_audio_url.scheme}://{parsed_audio_url.netloc}/{artist}/{songname}.mp3"
            if validators.url(url):
                response = requests.get(url)
                if response.status_code == 200:
                    pygame.mixer.music.load(BytesIO(response.content))
                    pygame.mixer.music.play()
                    print(f"Now playing: {songname} by {artist}")
                else:
                    raise ValueError("Unable to fetch audio file")
            else:
                raise ValueError("Invalid audio file URL")

        # check if user wants to play a song
        match = re.match(regex_play, usersaid)
        if match:
            songname = match.group(1)
            parsed_audio_url = urlparse(audiourl)
            url = f"{parsed_audio_url.scheme}://{parsed_audio_url.netloc}/{songname}.mp3"
            if validators.url(url):
                response = requests.get(url)
                if response.status_code == 200:
                    pygame.mixer.music.load(BytesIO(response.content))
                    pygame.mixer.music.play()
                    print(f"Now playing: {songname}")
                else:
                    raise ValueError("Unable to fetch audio file")
            else:
                raise ValueError("Invalid audio file URL")

        # check if user wants to resume the music
        elif re.match(regex_resume, usersaid):
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.unpause()
                print("Resuming the music")

        # check if user wants to pause the music
        elif re.match(regex_pause, usersaid):
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.pause()
                print("Pausing the music")

        # check if user wants to stop the music
        elif re.match(regex_stop, usersaid):
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
                print("Stopping the music")

        # check if user wants to increase the volume
        elif re.match(regex_volume_up, usersaid):
            pygame.mixer.music.set_volume(
                pygame.mixer.music.get_volume() + 0.1)
            print("Increasing the volume")

        # check if user wants to decrease the volume
        elif re.match(regex_volume_down, usersaid):
            pygame.mixer.music.set_volume(
                pygame.mixer.music.get_volume() - 0.1)
            print("Decreasing the volume")

    except requests.exceptions.RequestException as e:
        logger.exception("Error during network operation: %s", e)
    except pygame.error as e:
        logger.exception("Error during Pygame operation: %s", e)
    except Exception as e:
        logger.exception("Error: %s", e)


# # check if user wants to play a specific artist
# elif re.match(regex_specific_artist, usersaid):
#     artistname = re.match(regex_specific_artist, usersaid).group(5)
#     response = requests.get(audiourl + artistname)
#     pygame.mixer.music.load(BytesIO(response.content))
#     pygame.mixer.music.play()

# # check if user wants to play a specific album
# elif re.match(regex_specific_album, usersaid):
#     albumname = re.match(regex_specific_album, usersaid).group(4)
#     response = requests.get(audiourl + albumname)
#     pygame.mixer.music.load(BytesIO(response.content))
#     pygame.mixer.music.play()
