import cors from "cors";
import axios from "axios";
import chalk from "chalk";
import express from "express";
import { ytdlp } from "yt-dlp";
import Fetch from "node-fetch";
import ytSearch from "yt-search";

const app = express();
const PORT = 3000;
app.use(express.json());
app.use(cors());

const musicGenres = [
  { genre: "r&b", query: "r&b music" },
  { genre: "pop", query: "pop music" },
  { genre: "rock", query: "rock music" },
  { genre: "jazz", query: "jazz music" },
  { genre: "punk", query: "punk music" },
  { genre: "folk", query: "folk music" },
  { genre: "blues", query: "blues music" },
  { genre: "metal", query: "metal music" },
  { genre: "latin", query: "latin music" },
  { genre: "indie", query: "indie music" },
  { genre: "reggae", query: "reggae music" },
  { genre: "hip hop", query: "hip hop music" },
  { genre: "country", query: "country music" },
  { genre: "classical", query: "classical music" },
  { genre: "electronic", query: "electronic music" },
];

app.get("/news", async (req, res) => {
  const response = await axios.get(
    "https://newsapi.org/v2/top-headlines?country=in&apiKey=" + apiKey
  );
  console.log(response.data.articles[0]);
  res.send(response.data.articles[0]);
  console.log("Completed...");
});

app.get("/youtube", async (req, res) => {
  Fetch("https://magneum.vercel.app/api/youtube_sr?q=" + req.query.q, {
    method: "get",
    headers: {
      accept: "*/*",
      "accept-language": "en-US,en;q=0.9",
      "content-type": "application/x-www-form-urlencoded; charset=UTF-8",
    },
  }).then(async function (response) {
    const api_data = await response.json();
    console.log(api_data);
    ytdlp.audio
      .Auto_Sorted_Data({
        yturl: api_data.youtube_search[0].LINK,
        quality: "highest-possible",
      })
      .then((r) => {
        console.log(
          chalk.bgGreen("[PROMISE]:"),
          chalk.bgGrey("Auto_Sorted_Data()")
        );
        console.log(chalk.blue("Quality:"), chalk.gray(r.quality));
        console.log(chalk.blue("Resolution:"), chalk.gray(r.resolution));
        console.log(chalk.blue("Filesize:"), chalk.gray(r.filesize));
        console.log(chalk.blue("Audiochannels:"), chalk.gray(r.audiochannels));
        console.log(chalk.blue("Extensions:"), chalk.gray(r.extensions));
        console.log(chalk.blue("Audiocodec:"), chalk.gray(r.acodec));
        console.log(chalk.blue("Url:"), chalk.gray(r.url));
        res.send({
          name: api_data.youtube_search[0].TITLE,
          url: r.url,
        });
      })
      .catch((error) => console.log(chalk.bgRed("ERROR: "), chalk.gray(error)));
  });
});

app.get("/weather", async (req, res) => {
  try {
    const response = await axios.get(
      "http://api.openweathermap.org/data/2.5/weather?q=siliguri&appid=" +
        oapi_key
    );
    const temperature = Math.round(response.data.main.temp - 273.15);
    const humidity = response.data.main.humidity;
    const wind_speed = response.data.wind.speed;
    const weather_description = response.data.weather[0].description;
    console.log(`Temperature: ${temperature}°C`);
    console.log(`Humidity: ${humidity}%`);
    console.log(`Wind Speed: ${wind_speed} m/s`);
    console.log(`Weather Description: ${weather_description}`);
    res.send([
      {
        temperature: temperature + "°C",
        humidity: humidity + "%",
        wind_speed: wind_speed + "m/s",
        weather_description: weather_description,
      },
    ]);
  } catch (error) {
    console.error(error);
  }
});

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

app.get("/google", async (req, res) => {
  try {
    const query = req.query.query;
    res.status(200).send(query);
  } catch (error) {
    console.error(chalk.red(error));
    res.status(500).send("Internal Server Error");
  }
});

app.get("/wikipedia", async (req, res) => {
  try {
    const query = req.query.query;
    res.status(200).send(query);
  } catch (error) {
    console.error(chalk.red(error));
    res.status(500).send("Internal Server Error");
  }
});

app.get("/jokes", async (req, res) => {
  try {
    const query = req.query.query;
    res.status(200).send(query);
  } catch (error) {
    console.error(chalk.red(error));
    res.status(500).send("Internal Server Error");
  }
});

app.get("/datetime", async (req, res) => {
  try {
    const query = req.query.query;
    res.status(200).send(query);
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
