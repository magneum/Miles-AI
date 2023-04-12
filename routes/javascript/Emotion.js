import Jimp from "jimp";
import { promisify } from "util";
import terminalImage from "terminal-image";
import tensorflow from "@tensorflow/tfjs-node";

const modelPath = "models/FaceEmo/Face_Emotion_Model.h5";
const emotions = [
  "Angry",
  "Disgust",
  "Fear",
  "Happy",
  "Sad",
  "Surprise",
  "Neutral",
];

const main = async () => {
  const model = await tensorflow.loadLayersModel(modelPath);
  if (!model) {
    console.error("Failed to load the emotion detection model.");
    return;
  }

  const camera = await Jimp.create(640, 480);
  const frame = await promisify(camera.getBuffer.bind(camera))(Jimp.MIME_JPEG);
  const image = await Jimp.read(frame);
  const window = Jimp.create(640, 480);

  while (true) {
    const grayFrame = await image.clone().grayscale();
    const resizedFrame = await grayFrame.clone().resize(28, 28);
    const normalizedFrame = await resizedFrame.clone().normalize();

    const tensorFrame = tensorflow.tensor4d(
      [normalizedFrame.bitmap.data],
      [
        1,
        normalizedFrame.bitmap.height,
        normalizedFrame.bitmap.width,
        normalizedFrame.bitmap.channels,
      ]
    );
    const prediction = model.predict(tensorFrame);
    const predictionArray = await prediction.array();
    const maxIndex = predictionArray[0].indexOf(
      Math.max(...predictionArray[0])
    );
    const predictedEmotion = emotions[maxIndex];
    window
      .clone()
      .print(Jimp.loadFont(Jimp.FONT_SANS_16_WHITE), 10, 50, predictedEmotion);

    const buffer = await promisify(window.getBuffer.bind(window))(
      Jimp.MIME_JPEG
    );
    frame.data = Buffer.from(buffer);
    console.log(await terminalImage.buffer(frame.bitmap.data));
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
};

main();
