import express from "express";
import {
  searchVideo,
  getVideoDetails,
  searchVideosByGenre,
  getVideosFromChannel,
} from "./searchYoutube.js";

const app = express();

app.get("/searchVideo", async (req, res) => {
  const { keyword, numVideos } = req.query;
  const videos = await searchVideo(keyword, numVideos);
  res.send(videos);
});

app.get("/getVideoDetails", async (req, res) => {
  const { videoId } = req.query;
  const video = await getVideoDetails(videoId);
  res.send(video);
});

app.get("/searchVideosByGenre", async (req, res) => {
  const { genre, numVideos } = req.query;
  const videos = await searchVideosByGenre(genre, numVideos);
  res.send(videos);
});

app.get("/getVideosFromChannel", async (req, res) => {
  const { channelId, numVideos } = req.query;
  const videos = await getVideosFromChannel(channelId, numVideos);
  res.send(videos);
});

app.listen(3000, () => {
  console.log("Server is running on port 3000");
});
