import html
import re
import os
import random
import logging
import string
import subprocess
import pygame
import requests
from app import *


# def handle_music_command(usersaid):
#     try:
#         logger = logging.getLogger(__name__)
#         pygame.mixer.init()
#         usersaid = html.escape(usersaid).strip()

#         match = re.match(
#             r"play(ing)?\s+(?P<query>.+)|(?P<stop>stop|pause|resume)|(volume)?(?P<direction>up|down)",
#             usersaid,
#         )

#         if match:
#             if match.group("query"):
#                 filen = "".join(
#                     random.choice(string.ascii_lowercase) for _ in range(10)
#                 )
#                 query = match.group("query")
#                 miles_speaker(f"Wait for download to finish for: {query}")
#                 response = requests.get(
#                     "http://localhost:3000/youtube", params={"q": query}
#                 )
#                 data = response.json()
#                 name = data["name"]
#                 filename = f"{filen}.mp3"
#                 subprocess.run(["bin/ffmpeg.exe", "-i", data["url"], filename])

#                 if os.path.exists(filename):
#                     pygame.mixer.music.load(filename)
#                     pygame.mixer.music.play()
#                     print(f"Now playing: {name}")
#                     miles_speaker(f"Now playing: {name}")
#                 else:
#                     raise ValueError("Unable to fetch audio file")

#             elif match.group("stop"):
#                 if pygame.mixer.music.get_busy():
#                     if match.group("stop") == "pause":
#                         pygame.mixer.music.pause()
#                         print("Pausing the music")
#                         miles_speaker("Pausing the music")
#                     elif match.group("stop") == "resume":
#                         pygame.mixer.music.unpause()
#                         print("Resuming the music")
#                         miles_speaker("Resuming the music")
#                     else:
#                         pygame.mixer.music.stop()
#                         print("Stopping the music")
#                         miles_speaker("Stopping the music")
#                         if os.path.exists(filename):
#                             os.remove(filename)

#             elif match.group("volume"):
#                 rounded_volume = pygame.mixer.music.get_volume()
#                 if match.group("direction") == "up":
#                     new_volume = rounded_volume + 0.1
#                     if new_volume <= 1:
#                         pygame.mixer.music.set_volume(new_volume)
#                         print("Increasing the volume")
#                         miles_speaker("Increasing the volume")
#                     else:
#                         print("Volume is already at maximum")
#                         miles_speaker("Volume is already at maximum")
#                 elif match.group("direction") == "down":
#                     new_volume = rounded_volume - 0.1
#                     if new_volume >= 0:
#                         pygame.mixer.music.set_volume(new_volume)
#                         print("Decreasing the volume")
#                         miles_speaker("Decreasing the volume")
#                     else:
#                         print("Volume is already at minimum")
#                         miles_speaker("Volume is already at minimum")

#     except subprocess.CalledProcessError as e:
#         logger.exception("Error during FFmpeg operation: %s", e)
#     except pygame.error as e:
#         logger.exception("Error during Pygame operation: %s", e)
#     except Exception as e:
#         logger.exception("Error: %s", e)


def handle_music_command(usersaid, volume=None):
    pygame.mixer.init()
    usersaid = html.escape(usersaid).strip()

    match = re.match(
        r"play(ing)?\s+(?P<query>.+)|(?P<stop>stop|pause|resume)|(?P<volume>volume)\s+(?P<direction>up|down)",
        usersaid,
    )

    if match:
        if match.group("query"):
            filen = "".join(random.choice(string.ascii_lowercase) for _ in range(10))
            query = match.group("query")
            miles_speaker(f"Wait for download to finish for: {query}")
            response = requests.get(
                "http://localhost:3000/youtube", params={"q": query}
            )
            data = response.json()
            name = data["name"]
            filename = f"{filen}.mp3"
            subprocess.run(["bin/ffmpeg.exe", "-i", data["url"], filename])

            if os.path.exists(filename):
                pygame.mixer.music.load(filename)
                pygame.mixer.music.play()
                print(f"Now playing: {name}")
                miles_speaker(f"Now playing: {name}")
            else:
                raise ValueError("Unable to fetch audio file")

        elif match.group("stop"):
            if pygame.mixer.music.get_busy():
                if match.group("stop") == "pause":
                    pygame.mixer.music.pause()
                    print("Pausing the music")
                    miles_speaker("Pausing the music")
                elif match.group("stop") == "resume":
                    pygame.mixer.music.unpause()
                    print("Resuming the music")
                    miles_speaker("Resuming the music")
                else:
                    pygame.mixer.music.stop()
                    print("Stopping the music")
                    miles_speaker("Stopping the music")
                    # if os.path.exists(filename):
                    #     os.remove(filename)

        elif match.group("volume"):
            if volume is not None:
                new_volume = round(volume, 2)
            else:
                current_volume = pygame.mixer.music.get_volume()
                new_volume = round(current_volume, 2)

            if match.group("direction") == "up":
                if new_volume + 0.1 <= 1:
                    new_volume += 0.1
                    pygame.mixer.music.set_volume(new_volume)
                    print(f"Increasing the volume. Current volume: {new_volume:.2f}")
                    miles_speaker(
                        f"Increasing the volume. Current volume: {new_volume:.2f}"
                    )
                else:
                    print("Volume is already at maximum.")
                    miles_speaker("Volume is already at maximum.")
            elif match.group("direction") == "down":
                if new_volume - 0.1 >= 0:
                    new_volume -= 0.1
                    pygame.mixer.music.set_volume(new_volume)
                    print(f"Decreasing the volume. Current volume: {new_volume:.2f}")
                    miles_speaker(
                        f"Decreasing the volume. Current volume: {new_volume:.2f}"
                    )
                else:
                    print("Volume is already at minimum.")
                    miles_speaker("Volume is already at minimum.")
