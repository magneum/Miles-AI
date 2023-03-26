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
                filename = f"{name}.mp3"
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

    # Regardless of the outcome, quit the mixer and reinitialize it
    finally:
        pygame.mixer.quit()
        pygame.mixer.init()


"""
import html
import os
import random
import re
import string
import subprocess

import pygame
import requests
from app.miles_speaker import *


def handle_music_command(usersaid, volume=None):
    try:
        # Initialize the pygame mixer module
        pygame.mixer.init()

        # Escape any special HTML characters in the user's input and strip whitespace
        usersaid = html.escape(usersaid).strip()

        # Use regular expressions to match the user's input to a music command
        match = re.match(
            r"play(ing)?\s+(?P<query>.+)|(?P<stop>stop|pause|resume)|(?P<volume>volume)\s+(?P<direction>up|down)",
            usersaid,
        )

        # If there's no match, return immediately
        if not match:
            return

        # If the user's input matches a music query
        if match.group("query"):
            query = match.group("query")

            miles_speaker(
                f"Wait for download to finish for: {query}"
            )  # Use the MilesAI text-to-speech engine to say that the file is being downloaded
            # Make a request to the local YouTube downloader API to get the audio data for the query
            response = requests.get(
                "http://localhost:3000/youtube", params={"q": query}
            )

            data = response.json()  # Parse the JSON response
            name = data["name"]  # Get the name of the track from the JSON data
            filename = f"{name}.mp3"  # Create a filename for the downloaded file
            # Use subprocess to call ffmpeg to download the audio file
            try:
                subprocess.run(
                    ["ffmpeg", "-i", data["url"], "-f", "mp3", "audio.mp3"], check=True
                )
            except subprocess.CalledProcessError as e:
                print(f"Error downloading audio file: {e}")
                return
            # If the file doesn't exist, raise an error
            if not os.path.exists(filename):
                raise ValueError("Unable to fetch audio file")

            # Load the audio file into the pygame mixer
            pygame.mixer.music.load(filename)
            # Start playing the audio file
            pygame.mixer.music.play()
            # Print and say the name of the track that is now playing
            print(f"Now playing: {name}")
            miles_speaker(f"Now playing: {name}")

        # If the user's input matches a music stop command
        elif match.group("stop"):
            # If there's no music playing, return immediately
            if not pygame.mixer.music.get_busy():
                return

            # If the user wants to pause the music
            if match.group("stop") == "pause":
                # Pause the music playback and print/say that it's been paused
                pygame.mixer.music.pause()
                print("Pausing the music")
                miles_speaker("Pausing the music")
            # If the user wants to resume the music
            elif match.group("stop") == "resume":
                # Unpause the music playback and print/say that it's been resumed
                pygame.mixer.music.unpause()
                print("Resuming the music")
                miles_speaker("Resuming the music")
            # If the user wants to stop the music
            else:
                # Stop the music playback and print/say that it's been stopped
                pygame.mixer.music.stop()
                print("Stopping the music")
                miles_speaker("Stopping the music")
                # If the downloaded file still exists, delete it
                # if os.path.exists(filename):
                #     os.remove(filename)

        # If the user's input matches a volume command
        elif match.group("volume"):
            # Get the current volume of the pygame mixer
            current_volume = pygame.mixer.music.get_volume()
            new_volume = round(current_volume, 2)

            # If a new volume is specified as a parameter, use that instead
            if volume is not None:
                new_volume = round(volume, 2)

            # If the direction is "up", increase the volume by 0.1
            if match.group("direction") == "up":
                if (
                    new_volume + 0.1 <= 1
                ):  # check if the new volume doesn't exceed the maximum
                    new_volume += 0.1
                    pygame.mixer.music.set_volume(new_volume)
                    print(f"Increasing the volume. Current volume: {new_volume:.2f}")
                    miles_speaker(
                        f"Increasing the volume. Current volume: {new_volume:.2f}"
                    )
                else:
                    print("Volume is already at maximum.")
                    miles_speaker("Volume is already at maximum.")

            # If the direction is "down", decrease the volume by 0.1
            elif match.group("direction") == "down":
                if (
                    new_volume - 0.1 >= 0
                ):  # check if the new volume doesn't go below the minimum
                    new_volume -= 0.1
                    pygame.mixer.music.set_volume(new_volume)
                    print(f"Decreasing the volume. Current volume: {new_volume:.2f}")
                    miles_speaker(
                        f"Decreasing the volume. Current volume: {new_volume:.2f}"
                    )
                else:
                    print("Volume is already at minimum.")
                    miles_speaker("Volume is already at minimum.")

    # If there is an exception, print the error message and inform the user that music cannot be played
    except Exception as e:
        print(e)
        miles_speaker("Unable to play music")

    # Regardless of the outcome, quit the mixer and reinitialize it
    finally:
        pygame.mixer.quit()
        pygame.mixer.init()

"""
