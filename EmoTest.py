import os
import cv2
import numpy as np
import urllib.request
from keras.models import load_model
from colorama import Fore, Style

cascade_file_path = "corpdata/Fer2013-img/haarcascade_frontalface_default.xml"
if not os.path.exists(cascade_file_path):
    cv2_url = "https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml"
    print(
        Fore.YELLOW
        + "Haar cascade classifier file not found. Downloading..."
        + Style.RESET_ALL
    )
    urllib.request.urlretrieve(cv2_url, cascade_file_path)
    print(
        Fore.GREEN
        + "Haar cascade classifier file downloaded successfully!"
        + Style.RESET_ALL
    )

emotion_model = load_model("face_emotion_model.h5")
face_cascade = cv2.CascadeClassifier(cascade_file_path)


def main():
    Capture = cv2.VideoCapture(0)
    while True:
        ret, frame = Capture.read()
        if not ret:
            break
        Grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        Faces = face_cascade.detectMultiScale(
            Grayscale, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        for x, y, w, h in Faces:
            face_roi = Grayscale[y : y + h, x : x + w]
            face_roi = cv2.resize(face_roi, (48, 48))
            face_roi = face_roi.astype("float32") / 255.0
            face_roi = np.expand_dims(face_roi, axis=0)
            emotions = emotion_model.predict(face_roi)
            emotion_labels = [
                "Angry",
                "Disgust",
                "Fear",
                "Happy",
                "Sad",
                "Surprise",
                "Neutral",
            ]
        username = None
        predicted_emotion = emotion_labels[np.argmax(emotions)]
        for File in os.listdir("userNamedfaces"):
            img = cv2.imread(os.path.join("userNamedfaces", File), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (48, 48))
            img = img.astype("float32") / 255.0
            img = np.expand_dims(img, axis=0)
            img_emotions = emotion_model.predict(img)
            img_predicted_emotion = emotion_labels[np.argmax(img_emotions)]
            if img_predicted_emotion == predicted_emotion:
                username = File.split(".")[0]
                break
        if username:
            print(
                Fore.CYAN
                + "Detected username: "
                + Fore.YELLOW
                + f"{username}"
                + Style.RESET_ALL
            )
            print(
                Fore.CYAN
                + "Predicted emotion: "
                + Fore.YELLOW
                + f"{predicted_emotion}"
                + Style.RESET_ALL
            )
        cv2.imshow("Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    Capture.release()
    cv2.destroyAllWindows()


main()
