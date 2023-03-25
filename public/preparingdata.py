"""
This code defines two functions record_user_audio and record_background_noise that use the sounddevice library to record audio input and save it as WAV files in specific directories.

The record_user_audio function prompts the user to press ENTER to start recording voice and press ENTER again to stop recording. It then records audio input from the microphone for 0.5 seconds at a sampling rate of 44100 Hz and saves the recording as a WAV file in the public/audio/user_audio/ directory. The function also prints a message to indicate the current recording number.

The record_background_noise function is similar to record_user_audio, except that it records audio input for background noise and saves the recording as a WAV file in the public/audio/bg_noises/ directory.

The code then prompts the user to input a value (either "1" or "2") to determine which function to call. If the input is "1", it calls record_user_audio function. If the input is "2", it calls record_background_noise function. Otherwise, it prints an error message indicating invalid input.
"""


# Import required modules
import os
import sounddevice as sd
from scipy.io.wavfile import write

# Define a function to record user audio


def record_user_audio(n_times=100):
    # Set the filepath where the recorded audio will be saved
    filepath = "public/audio/user_audio/"
    # Prompt the user to start recording
    input("Press Enter to record voice.\nPress ENTER to record next.\nTo stop press CTRL+C.")
    # Loop n_times to record user audio
    for i in range(n_times):
        fs = 44100  # Set the sampling frequency
        duration = 1  # Set the duration of the recording
        # Record audio using the sounddevice module
        myrecording = sd.rec(int(fs * duration), samplerate=fs, channels=2)
        sd.wait()  # Wait until the recording is finished
        # Save the recorded audio to a WAV file
        write(filepath + str(i) + ".wav", fs, myrecording)
        # Print progress
        input(f"Currently at: {i + 1}/{n_times}")

# Define a function to record background noise


def record_background_noise(n_times=100):
    # Set the filepath where the recorded audio will be saved
    filepath = "public/audio/bg_noises/"
    # Loop n_times to record background noise
    for i in range(n_times):
        fs = 44100  # Set the sampling frequency
        duration = 1  # Set the duration of the recording
        # Record audio using the sounddevice module
        myrecording = sd.rec(int(fs * duration), samplerate=fs, channels=2)
        sd.wait()  # Wait until the recording is finished
        # Save the recorded audio to a WAV file
        write(filepath + str(i) + ".wav", fs, myrecording)
        # Print progress
        print(f"Currently at: {i + 1/n_times}")


# Prompt the user to select a recording option
value = input(
    "Type 1 for record_user_audio()\nType 2 for record_background_noise()\n:")
if value == "1":
    record_user_audio()  # Record user audio
elif value == "2":
    record_background_noise()  # Record background noise
else:
    print("Invalid input")  # Display error message for invalid input
