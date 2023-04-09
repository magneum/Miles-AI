import os
import librosa
import numpy as np
import pandas as pd
from colorama import Fore, Style

Style.RESET_ALL = ""

sample = "public/audio/bg_noises/0.wav"
try:
    data, sample_rate = librosa.load(sample)
    mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40)
except Exception as e:
    print(Fore.RED + f"Error: Failed to load sample audio file: {e}")
    exit(1)

all_data = []
data_path_dict = {
    0: [
        "public/audio/bg_noises/" + file_path
        for file_path in os.listdir("public/audio/bg_noises/")
    ],
    1: [
        "public/audio/user_audio/" + file_path
        for file_path in os.listdir("public/audio/user_audio/")
    ],
}

for class_label, list_of_files in data_path_dict.items():
    for single_file in list_of_files:
        try:
            audio, sample_rate = librosa.load(single_file)
            mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
            mfcc_processed = np.mean(mfcc.T, axis=0)
            all_data.append([mfcc_processed, class_label])
        except Exception as e:
            print(Fore.YELLOW + f"Warning: Failed to process file {single_file}: {e}")

    print(Fore.GREEN + f"Info: Successfully preprocessed class label {class_label}")

if all_data:
    df = pd.DataFrame(all_data, columns=["feature", "class_label"])
    df.to_pickle("models/wakeword/audio_data.csv")
    print(
        Fore.GREEN
        + "Info: Successfully saved preprocessed audio data to models/wakeword/audio_data.csv"
    )
else:
    print(Fore.RED + "Error: No audio data processed. Failed to save audio_data.csv")
