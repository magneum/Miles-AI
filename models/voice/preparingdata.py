import sounddevice as sd
from scipy.io.wavfile import write

# Define a function to record user audio
def record_user_audio(n_times=100):
    # Set the filepath where the recorded audio will be saved
    filepath = "public/audio/user_audio/"
    # Prompt the user to start recording
    input(
        "Press Enter to record voice.\nPress ENTER to record next.\nTo stop press CTRL+C."
    )
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
value = input("Type 1 for record_user_audio()\nType 2 for record_background_noise()\n:")
if value == "1":
    record_user_audio()  # Record user audio
elif value == "2":
    record_background_noise()  # Record background noise
else:
    print("Invalid input")  # Display error message for invalid input
