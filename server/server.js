// > SINGLETRACK() function takes a song name as input and returns the first video search result for that song.
// > RANDOMGENRETRACK() function takes a music genre as input and returns a random video search result for that genre.
// > RANDOMTRACK() function selects a random music genre from a predefined list and returns a random video search result for that genre.
// > CUSTOMTRACK() function takes a search query and a number of desired search results as inputs, and returns an array of videos that match the search query.

import chalk from "chalk";
import express from "express";
import ytSearch from "yt-search";

const app = express();
const PORT = 3000;

const musicGenres = [
  { genre: "pop", query: "pop music" },
  { genre: "rock", query: "rock music" },
  { genre: "hip hop", query: "hip hop music" },
  { genre: "jazz", query: "jazz music" },
  { genre: "blues", query: "blues music" },
  { genre: "classical", query: "classical music" },
  { genre: "country", query: "country music" },
  { genre: "reggae", query: "reggae music" },
  { genre: "electronic", query: "electronic music" },
  { genre: "metal", query: "metal music" },
  { genre: "punk", query: "punk music" },
  { genre: "folk", query: "folk music" },
  { genre: "r&b", query: "r&b music" },
  { genre: "latin", query: "latin music" },
  { genre: "indie", query: "indie music" },
];

app.get("/singleTrack", async (req, res) => {
  try {
    const songName = req.query.songName;
    const results = await ytSearch(songName);
    const video = results.videos[0];
    res.status(200).send(video);
  } catch (error) {
    console.error(chalk.red(error));
    res.status(500).send("Internal Server Error");
  }
});

app.get("/randomTrack", async (req, res) => {
  try {
    const genreIndex = Math.floor(Math.random() * musicGenres.length);
    const { genre, query } = musicGenres[genreIndex];
    const results = await ytSearch(query);
    const videoIndex = Math.floor(Math.random() * results.videos.length);
    const video = results.videos[videoIndex];
    res.status(200).send(video);
  } catch (error) {
    console.error(chalk.red(error));
    res.status(500).send("Internal Server Error");
  }
});

app.get("/customTrack", async (req, res) => {
  try {
    const query = req.query.query;
    const totalTracks = req.query.totalTracks;
    const results = await ytSearch(query);
    const videos = results.videos.slice(0, totalTracks);
    res.status(200).send(videos);
  } catch (error) {
    console.error(chalk.red(error));
    res.status(500).send("Internal Server Error");
  }
});

app.get("/randomGenreTrack", async (req, res) => {
  try {
    const genre = req.query.genre;
    const { query } = musicGenres.find(
      (musicGenre) => musicGenre.genre === genre.toLowerCase()
    );
    const results = await ytSearch(query);
    const videoIndex = Math.floor(Math.random() * results.videos.length);
    const video = results.videos[videoIndex];
    res.status(200).send(video);
  } catch (error) {
    console.error(chalk.red(error));
    res.status(500).send("Internal Server Error");
  }
});

app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});

// http://localhost:3000/randomTrack
// http://localhost:3000/randomGenreTrack?genre=hip hop
// http://localhost:3000/singleTrack?songName=Bohemian Rhapsody
// http://localhost:3000/customTrack?query=guitar solos&totalTracks=5
