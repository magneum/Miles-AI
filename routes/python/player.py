import re
import os
import html
import random
import string
import pygame
import requests
import subprocess
from main import milesVoice


def player(usersaid, volume=None):
    try:
        pygame.mixer.init()
        usersaid = html.escape(usersaid).strip()
        match = re.match(
            r"play(ing)?\s+(?P<query>.+)|(?P<stop>stop|pause|resume)|(?P<volume>volume)\s+(?P<direction>up|down)",
            usersaid,
        )

        if match:
            if match.group("query"):
                filen = "".join(
                    random.choice(string.ascii_lowercase) for _ in range(10)
                )
                query = match.group("query")
                milesVoice(f"Wait for download to finish for: {query}")
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
                    milesVoice(f"Now playing: {name}")
                else:
                    raise ValueError("Unable to fetch audio file")

            elif match.group("stop"):
                if pygame.mixer.music.get_busy():
                    if match.group("stop") == "pause":
                        pygame.mixer.music.pause()
                        print("Pausing the music")
                        milesVoice("Pausing the music")
                    elif match.group("stop") == "resume":
                        pygame.mixer.music.unpause()
                        print("Resuming the music")
                        milesVoice("Resuming the music")
                    else:
                        pygame.mixer.music.stop()
                        print("Stopping the music")
                        milesVoice("Stopping the music")
                        if os.path.exists(filename):
                            os.remove(filename)

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
                        print(
                            f"Increasing the volume. Current volume: {new_volume:.2f}"
                        )
                        milesVoice(
                            f"Increasing the volume. Current volume: {new_volume:.2f}"
                        )
                    else:
                        print("Volume is already at maximum.")
                        milesVoice("Volume is already at maximum.")
                elif match.group("direction") == "down":
                    if new_volume - 0.1 >= 0:
                        new_volume -= 0.1
                        pygame.mixer.music.set_volume(new_volume)
                        print(
                            f"Decreasing the volume. Current volume: {new_volume:.2f}"
                        )
                        milesVoice(
                            f"Decreasing the volume. Current volume: {new_volume:.2f}"
                        )
                    else:
                        print("Volume is already at minimum.")
                        milesVoice("Volume is already at minimum.")
    except Exception as e:
        print(e)
        milesVoice("Unable to play music")
