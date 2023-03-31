import html
import re
import os
import random
import logging
import string
import subprocess
import pygame
import requests
from app.miles_speaker import *


# Define the function to handle music commands
def handle_music_command(usersaid, volume=None):
    try:
        # Initialize Pygame mixer
        pygame.mixer.init()

        # Escape special characters in user input
        usersaid = html.escape(usersaid).strip()

        # Use regular expressions to match user input to a music command
        match = re.match(
            r"play(ing)?\s+(?P<query>.+)|(?P<stop>stop|pause|resume)|(?P<volume>volume)\s+(?P<direction>up|down)",
            usersaid,
        )

        # If a matching command is found:
        if match:
            # If the user wants to play music:
            if match.group("query"):
                # Generate a random filename for the audio file
                filen = "".join(
                    random.choice(string.ascii_lowercase) for _ in range(10)
                )
                # Get the query from the user input
                query = match.group("query")
                # Speak to user to wait for the audio file to download
                miles_speaker(f"Wait for download to finish for: {query}")
                # Query the server for the audio file
                response = requests.get(
                    "http://localhost:3000/youtube", params={"q": query}
                )
                data = response.json()
                # Get the name and filename of the audio file
                name = data["name"]
                filename = f"{filen}.mp3"
                # Use FFmpeg to download and convert the audio file
                subprocess.run(["bin/ffmpeg.exe", "-i", data["url"], filename])
                # If the file exists, load and play it with Pygame
                if os.path.exists(filename):
                    pygame.mixer.music.load(filename)
                    pygame.mixer.music.play()
                    print(f"Now playing: {name}")
                    miles_speaker(f"Now playing: {name}")
                else:
                    # If the file cannot be found, raise an error
                    raise ValueError("Unable to fetch audio file")

            # If the user wants to stop, pause, or resume music:
            elif match.group("stop"):
                if pygame.mixer.music.get_busy():
                    if match.group("stop") == "pause":
                        # Pause the music and speak to the user
                        pygame.mixer.music.pause()
                        print("Pausing the music")
                        miles_speaker("Pausing the music")
                    elif match.group("stop") == "resume":
                        # Resume the music and speak to the user
                        pygame.mixer.music.unpause()
                        print("Resuming the music")
                        miles_speaker("Resuming the music")
                    else:
                        # Stop the music and speak to the user
                        pygame.mixer.music.stop()
                        print("Stopping the music")
                        miles_speaker("Stopping the music")
                        # if os.path.exists(filename):
                        #     os.remove(filename)

            elif match.group("volume"):
                # If the function was passed a volume parameter
                if volume is not None:
                    new_volume = round(volume, 2)
                # If the function was not passed a volume parameter
                else:
                    # Get the current volume and round it to two decimal places
                    current_volume = pygame.mixer.music.get_volume()
                    new_volume = round(current_volume, 2)

                # Check if the direction is up
                if match.group("direction") == "up":
                    # Check if increasing the volume by 0.1 will not exceed the maximum volume of 1
                    if new_volume + 0.1 <= 1:
                        # Increase the volume by 0.1 and set the new volume
                        new_volume += 0.1
                        pygame.mixer.music.set_volume(new_volume)
                        print(
                            f"Increasing the volume. Current volume: {new_volume:.2f}"
                        )
                        miles_speaker(
                            f"Increasing the volume. Current volume: {new_volume:.2f}"
                        )
                    else:
                        # If the maximum volume is already reached, inform the user
                        print("Volume is already at maximum.")
                        miles_speaker("Volume is already at maximum.")
                # If the direction is down
                elif match.group("direction") == "down":
                    # Check if decreasing the volume by 0.1 will not go below the minimum volume of 0
                    if new_volume - 0.1 >= 0:
                        # Decrease the volume by 0.1 and set the new volume
                        new_volume -= 0.1
                        pygame.mixer.music.set_volume(new_volume)
                        print(
                            f"Decreasing the volume. Current volume: {new_volume:.2f}"
                        )
                        miles_speaker(
                            f"Decreasing the volume. Current volume: {new_volume:.2f}"
                        )
                    else:
                        # If the minimum volume is already reached, inform the user
                        print("Volume is already at minimum.")
                        miles_speaker("Volume is already at minimum.")
    # If there is an exception, print the error message and inform the user that music cannot be played
    except Exception as e:
        print(e)
        miles_speaker("Unable to play music")
