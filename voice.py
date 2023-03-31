import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Set up data paths
data_path = "public/audio/"

# Define function to extract audio features
def extract_features(file_path):
    with open(file_path, "rb") as f:
        audio, sr = librosa.load(f, sr=None)
    mfcc = librosa.feature.mfcc(audio, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc, axis=1)
    return mfcc_mean


# Extract audio features and labels for each file
features = []
labels = []
for label, sub_dir in enumerate(os.listdir(data_path)):
    sub_dir_path = os.path.join(data_path, sub_dir)
    if not os.path.isdir(sub_dir_path):
        continue
    for file in os.listdir(sub_dir_path):
        file_path = os.path.join(sub_dir_path, file)
        feature = extract_features(file_path)
        features.append(feature)
        labels.append(label)
features = np.array(features)
labels = np.array(labels)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

# Train MLP classifier
clf = MLPClassifier(
    hidden_layer_sizes=(100,),
    max_iter=500,
    alpha=0.0001,
    solver="adam",
    verbose=10,
    random_state=42,
    tol=0.000000001,
)
clf.fit(X_train, y_train)

# Test accuracy
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {acc}")

# +==============================================+
import sounddevice as sd
import wave


# Start recording
print("Recording started")
recording = sd.rec(int(5 * 44100), samplerate=44100, channels=1)
sd.wait()  # Wait for recording to complete
print("Recording finished")

# Save recording to file
with wave.open("recording.wav", "wb") as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(44100)
    f.writeframes(b"".join([x.tobytes() for x in recording]))

print(f"Recording saved to recording.wav")
