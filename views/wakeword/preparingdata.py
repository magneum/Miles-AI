import sounddevice as sd
from scipy.io.wavfile import write


def record_user_audio(n_times=100):
    filepath = "public/audio/user_audio/"
    input("Press Enter to record voice.")
    for i in range(n_times):
        fs = 44100
        duration = 0.5
        myrecording = sd.rec(int(fs * duration), samplerate=fs, channels=2)
        sd.wait()
        write(filepath + str(i) + ".wav", fs, myrecording)
        input(f"Currently at: {i + 1}/{n_times}")


def record_background_noise(n_times=100):
    filepath = "public/audio/bg_noises/"
    for i in range(n_times):
        fs = 44100
        duration = 0.5
        myrecording = sd.rec(int(fs * duration), samplerate=fs, channels=2)
        sd.wait()
        write(filepath + str(i) + ".wav", fs, myrecording)
        print(f"Currently at: {i + 1/n_times}")


value = input("> 1 for record_user_audio()\n> 2 for record_background_noise()\n\n:")
if value == "1":
    record_user_audio()
elif value == "2":
    record_background_noise()
else:
    print("Invalid input")
