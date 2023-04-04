import librosa
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from tensorflow.python.keras.models import load_model

# Set variables
fs = 44100
seconds = 2
filename = "prediction.wav"
class_names = ["Wake Word NOT Detected", "Wake Word Detected"]

# Load trained model
model = load_model("public/audio/wake_word.h5")

# Start recording audio
print("Prediction Started: ")
i = 0
print("Say Now: ")
myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
sd.wait()

# Save recorded audio to file
write(filename, fs, myrecording)

# Load the saved audio file
audio, sample_rate = librosa.load(filename)

# Extract Mel-frequency cepstral coefficients (MFCC) features from audio
mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
mfcc_processed = np.mean(mfcc.T, axis=0)

# Use the model to predict if the wake word was detected
prediction = model.predict(np.expand_dims(mfcc_processed, axis=0))

# Print the prediction and confidence level
if prediction[:, 1] > 0.99:
    print(f"Wake Word Detected for ({i})")
    print("Confidence:", prediction[:, 1])
    i += 1
else:
    print(f"Wake Word NOT Detected")
    print("Confidence:", prediction[:, 0])
