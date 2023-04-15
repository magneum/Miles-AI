import os
import cv2
import keras
import numpy as np
import urllib.request
from colorama import Fore, Style

cascade_file_path = "corpdata/haarcascade_frontalface_default.xml"
mesh_model_path = "corpdata/face_mesh_model/face_mesh_model.prototxt"
mesh_weights_path = "corpdata/face_mesh_model/face_mesh_model.caffemodel"

if not os.path.exists(cascade_file_path):
    cv2_url = "https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml"
    print(Fore.YELLOW + "Haar cascade classifier file not found. Downloading...")
    print(Style.RESET_ALL)
    urllib.request.urlretrieve(cv2_url, cascade_file_path)
    print(Fore.GREEN + "Haar cascade classifier file downloaded successfully!")
    print(Style.RESET_ALL)

if not os.path.exists(mesh_model_path) or not os.path.exists(mesh_weights_path):
    face_mesh_model_url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel"
    print(Fore.YELLOW + "Face mesh model files not found. Downloading...")
    print(Style.RESET_ALL)
    urllib.request.urlretrieve(face_mesh_model_url, mesh_weights_path)
    print(Fore.GREEN + "Face mesh model files downloaded successfully!")
    print(Style.RESET_ALL)

emotion_model = keras.models.load_model("face_emotion_model.h5")
face_cascade = cv2.CascadeClassifier(cascade_file_path)


def detect_faces(frame):
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        grayscale, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    face_rectangles = []
    for x, y, w, h in faces:
        face_rectangles.append((x, y, w, h))
    return face_rectangles


def detect_face_mesh(frame, net):
    frame_height, frame_width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()
    face_keypoints = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            face_keypoints.append(((x1, y1), (x2, y2)))
    return face_keypoints


def main():
    net = cv2.dnn.readNetFromCaffe(mesh_model_path, mesh_weights_path)
    capture = cv2.VideoCapture(0)
    while True:
        success, frame = capture.read()
        if not success:
            break
        print("Captured frame")
        face_rectangles = detect_faces(frame)
        face_keypoints = detect_face_mesh(frame, net)
        for x, y, w, h in face_rectangles:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        for (x1, y1), (x2, y2) in face_keypoints:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        for x, y, w, h in face_rectangles:
            face_roi = frame[y : y + h, x : x + w]
            face_roi_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            face_roi_gray = cv2.resize(face_roi_gray, (48, 48))
            face_roi_gray = face_roi_gray / 255.0
            face_roi_gray = np.expand_dims(face_roi_gray, axis=0)
            face_roi_gray = np.expand_dims(face_roi_gray, axis=-1)
            emotion_labels = [
                "Angry",
                "Disgust",
                "Fear",
                "Happy",
                "Sad",
                "Surprise",
                "Neutral",
            ]
            emotion_predictions = emotion_model.predict(face_roi_gray)
            emotion_index = np.argmax(emotion_predictions[0])
            emotion_label = emotion_labels[emotion_index]
            cv2.putText(
                frame,
                emotion_label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
        cv2.imshow("Webcam Face Mesh Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
