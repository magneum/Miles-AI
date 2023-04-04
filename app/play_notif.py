import numpy as np
import pyaudio


def play_notif(freq, duration):
    signal = np.sin(
        freq * 2 * np.pi * np.linspace(0, duration, int(duration * 44100), False)
    )
    stream = pyaudio.PyAudio().open(
        format=pyaudio.paFloat32, channels=1, rate=44100, output=True
    )
    stream.write(signal.astype(np.float32).tobytes())
    stream.close()
    pyaudio.PyAudio().terminate()
