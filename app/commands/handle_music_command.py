import subprocess
import os
import re
import pygame
import html
import logging
import random
import string
from app import *
# =============================================================================================================
# regex patterns for various commands
regex_play = r"(play|start)( the)?\s?(song|music)?\s?(.*)"
regex_stop = r"(stop|end|finish|terminate)( the)?\s?(song|music)?"
regex_pause = r"pause( the)?\s?(song|music)?"
regex_resume = r"resume( the)?\s?(song|music)?"
regex_volume_up = r"volume (up|increase)"
regex_volume_down = r"volume (down|decrease)"
# initialize logger
logger = logging.getLogger(__name__)
# =============================================================================================================


def handle_music_command(usersaid, audiourl):
    try:
        pygame.mixer.init()
        usersaid = html.escape(usersaid).strip()

        match = re.match(regex_play, usersaid)
        if match:
            # Generate a random string of 10 lowercase letters
            songname = "".join(random.choice(string.ascii_lowercase)
                               for _ in range(10))
            filename = f"{songname}.mp3"
            subprocess.run(["bin/ffmpeg.exe", "-i", audiourl, filename])

            if os.path.exists(filename):
                pygame.mixer.music.load(filename)
                pygame.mixer.music.play()
                print(f"Now playing: {songname}")
                miles_speaker(f"Now playing: {songname}")
            else:
                raise ValueError("Unable to fetch audio file")

        # check if user wants to resume the music
        elif re.match(regex_resume, usersaid):
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.unpause()
                print("Resuming the music")
                miles_speaker("Resuming the music")

        # check if user wants to pause the music
        elif re.match(regex_pause, usersaid):
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.pause()
                print("Pausing the music")
                miles_speaker("Pausing the music")

        # check if user wants to stop the music
        elif re.match(regex_stop, usersaid):
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
                print("Stopping the music")
                miles_speaker("Stopping the music")

        # check if user wants to increase the volume
        elif re.match(regex_volume_up, usersaid):
            pygame.mixer.music.set_volume(
                pygame.mixer.music.get_volume() + 0.1)
            print("Increasing the volume")
            miles_speaker("Increasing the volume")

        # check if user wants to decrease the volume
        elif re.match(regex_volume_down, usersaid):
            pygame.mixer.music.set_volume(
                pygame.mixer.music.get_volume() - 0.1)
            print("Decreasing the volume")
            miles_speaker("Decreasing the volume")

    except subprocess.CalledProcessError as e:
        logger.exception("Error during FFmpeg operation: %s", e)
    except pygame.error as e:
        logger.exception("Error during Pygame operation: %s", e)
    except Exception as e:
        logger.exception("Error: %s", e)
