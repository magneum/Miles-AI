import librosa
import numpy as np
import sounddevice as sd
from colorama import Fore, Style
from scipy.io.wavfile import write
from tensorflow.python.keras.models import load_model


fs = 44100
seconds = 2
filename = "prediction.wav"
model = load_model("models/wakeword/best_hyper_model.h5")
class_names = ["Wake Word NOT Detected", "Wake Word Detected"]

i = 0
print(f"{Fore.YELLOW}Prediction Started!")
while True:
    print("Say Now: ")
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()
    write(filename, fs, myrecording)
    audio, sample_rate = librosa.load(filename)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfcc_processed = np.mean(mfcc.T, axis=0)
    prediction = model.predict(np.expand_dims(mfcc_processed, axis=0))
    if prediction[:, 1] > 0.99:
        print(f"{Fore.GREEN}Wake Word Detected for ({i}){Style.RESET_ALL}")
        print("Confidence:", prediction[:, 1])
        i += 1
    else:
        print(f"{Fore.RED}Wake Word NOT Detected{Style.RESET_ALL}")
        print("Confidence:", prediction[:, 0])
