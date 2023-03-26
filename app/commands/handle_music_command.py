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


def handle_music_command(usersaid, volume=None):
    pygame.mixer.init()
    usersaid = html.escape(usersaid).strip()

    match = re.match(
        r"play(ing)?\s+(?P<query>.+)|(?P<stop>stop|pause|resume)|(?P<volume>volume)\s+(?P<direction>up|down)",
        usersaid,
    )

    if match:
        if match.group("query"):
            query = match.group("query")
            miles_speaker(f"Wait for download to finish for: {query}")
            response = requests.get(
                "http://localhost:3000/youtube", params={"q": query}
            )
            data = response.json()
            title = data["name"]
            filename = f"{title}.mp3"
            if os.path.exists(filename):
                pass
            else:
                subprocess.run(["bin/ffmpeg.exe", "-i", data["url"], filename])

            if os.path.exists(filename):
                pygame.mixer.music.load(filename)
                pygame.mixer.music.play()
                print(f"Now playing: {title}")
                miles_speaker(f"Now playing: {title}")
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
                    if os.path.exists(filename):
                        try:
                            os.remove(filename)
                        except:
                            pass

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
