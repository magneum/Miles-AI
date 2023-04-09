import speech_recognition as sr
from keras.models import load_model
import sounddevice as sd
import numpy as np

model = load_model("models/wakeword/best_hyper_model.h5")


def audio_callback(indata, frames, time, status):
    indata = np.expand_dims(indata, axis=0)
    indata = np.squeeze(indata, axis=-1)
    indata = indata[:, :40]
    prediction = model.predict(indata)
    if np.any(prediction[0] > 0.8):
        print("Wake Word Found")
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening for speech...")
            audio = recognizer.listen(source)
            try:
                text = recognizer.recognize_google(audio)
                print("Speech detected: ", text)
            except sr.UnknownValueError:
                print("Unable to recognize speech.")
            except sr.RequestError:
                print("Speech recognition service unavailable.")
    else:
        print("Wake Word not detected.")


fs = 16000
stream = sd.InputStream(samplerate=16000, channels=1, callback=audio_callback)
stream.start()

print("Recording audio...")

while True:
    if stream.active:
        continue
    else:
        break

stream.stop()
stream.close()

print("Recording stopped.")
