import os
import librosa
import numpy as np
import pandas as pd
import librosa.display
import matplotlib.pyplot as plt

print(librosa.__version__)  # type: ignore

sample = "public/audio/bg_noises/0.wav"
data, sample_rate = librosa.load(sample)
# plt.title("wave form")
# librosa.display.waveplot(data, sr=sample_rate)
# plt.show()

mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40)
# print(f"Shape of mfcc: {mfccs.shape}")

# plt.title("MFCC")
# librosa.display.specshow(mfccs, sr=sample_rate, x_axis="time")
# plt.show()


all_data = []
data_path_dict = {
    0: ["public/audio/bg_noises/" + file_path for file_path in os.listdir("public/audio/bg_noises/")],
    1: ["public/audio/user_audio/" + file_path for file_path in os.listdir("public/audio/user_audio/")],
}

for class_label, list_of_files in data_path_dict.items():
    for single_file in list_of_files:
        audio, sample_rate = librosa.load(single_file)  # Loading file
        mfcc = librosa.feature.mfcc(
            y=audio, sr=sample_rate, n_mfcc=40)  # Apllying mfcc
        mfcc_processed = np.mean(mfcc.T, axis=0)  # some pre-processing
        all_data.append([mfcc_processed, class_label])
    print(f"Info: Succesfully Preprocessed Class Label {class_label}")

df = pd.DataFrame(all_data, columns=["feature", "class_label"])

###### SAVING FOR FUTURE USE ###
df.to_pickle("public/audio/audio_data.csv")

# pip install numba==0.48
# pip uninstall --yes librosa
# pip install librosa --force-reinstall
