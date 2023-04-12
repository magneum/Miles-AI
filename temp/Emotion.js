import cv from "opencv4nodejs"; // yarn add opencv4nodejs
import tf from "@tensorflow/tfjs-node"; // yarn add @tensorflow/tfjs-node

const model = await tf.loadLayersModel("emotion_detection_model.h5");
const camera = new cv.VideoCapture(0);
camera.set(cv.CAP_PROP_FPS, 30);
cv.namedWindow("Emotion Detection", cv.WINDOW_NORMAL);

while (true) {
  const frame = camera.read();
  const grayFrame = frame.cvtColor(cv.COLOR_BGR2GRAY);
  const resizedFrame = grayFrame.resize(new cv.Size(28, 28));
  const normalizedFrame = resizedFrame.normalize(
    0,
    255,
    cv.NORM_MINMAX,
    cv.CV_32F
  );
  const tensorFrame = tf.tensor4d([normalizedFrame.getDataAsArray()]);
  const prediction = model.predict(tensorFrame);
  const predictionArray = prediction.arraySync()[0];
  const maxIndex = predictionArray.indexOf(Math.max(...predictionArray));
  const emotions = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
  ];
  const predictedEmotion = emotions[maxIndex];
  frame.putText(
    predictedEmotion,
    new cv.Point(10, 50),
    cv.FONT_HERSHEY_SIMPLEX,
    1,
    new cv.Vec(0, 0, 255),
    2
  );
  cv.imshow("Emotion Detection", frame);
  const key = cv.waitKey(100);
  if (key === 113) {
    break;
  }
}

camera.release();
cv.destroyAllWindows();
