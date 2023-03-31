import os
import librosa
import numpy as np
import pandas as pd
import librosa.display
import matplotlib.pyplot as plt

print(librosa.__version__)

# Loading a sample audio file
sample = "public/audio/bg_noises/0.wav"
data, sample_rate = librosa.load(sample)

# Visualizing the waveform
plt.title("Waveform")
librosa.display.waveplot(data, sr=sample_rate)
plt.show()

# Extracting MFCC features
mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40)

# Visualizing MFCC features
plt.title("MFCC")
librosa.display.specshow(mfccs, sr=sample_rate, x_axis="time")
plt.show()

# Preprocessing and extracting features from all audio files
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
        # Loading the audio file
        audio, sample_rate = librosa.load(single_file)

        # Applying MFCC and preprocessing
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfcc_processed = np.mean(mfcc.T, axis=0)

        # Appending the processed features and class label to a list
        all_data.append([mfcc_processed, class_label])

    print(f"Info: Successfully preprocessed class label {class_label}")

# Creating a pandas dataframe from the processed data and saving it for future use
df = pd.DataFrame(all_data, columns=["feature", "class_label"])
df.to_pickle("models/voice/audio_data.csv")
