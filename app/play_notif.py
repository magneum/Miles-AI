import numpy as np
import pyaudio
# =============================================================================================================


def play_notif(freq, duration):
    # Generate a sine wave audio signal with the given frequency and duration
    signal = np.sin(freq * 2 * np.pi * np.linspace(0,
                    duration, int(duration * 44100), False))

    # Open a PyAudio stream for audio output
    stream = pyaudio.PyAudio().open(format=pyaudio.paFloat32,
                                    channels=1,
                                    rate=44100,
                                    output=True)

    # Write the audio signal to the stream
    stream.write(signal.astype(np.float32).tobytes())

    # Close the audio stream
    stream.close()

    # Terminate the PyAudio interface
    pyaudio.PyAudio().terminate()
# =============================================================================================================
