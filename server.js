import Fetch from "node-fetch";
import { ytdlp } from "yt-dlp";
import express from "express";
import axios from "axios";
import cors from "cors";
import c from "chalk";
const app = express();
const port = 3000;

app.use(express.json());
app.use(cors()); // add CORS middleware

app.get("/news", async (req, res) => {
  // // handle GET request for fetching news
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
        yturl: api_data.youtube_search[0].LINK, // required
        quality: "lowest-possible", // required
      })
      .then((r) => {
        console.log(c.bgGreen("[PROMISE]:"), c.bgGrey("Auto_Sorted_Data()"));
        console.log(c.blue("Quality:"), c.gray(r.quality));
        console.log(c.blue("Resolution:"), c.gray(r.resolution));
        console.log(c.blue("Filesize:"), c.gray(r.filesize));
        console.log(c.blue("Audiochannels:"), c.gray(r.audiochannels));
        console.log(c.blue("Extensions:"), c.gray(r.extensions));
        console.log(c.blue("Audiocodec:"), c.gray(r.acodec));
        console.log(c.blue("Url:"), c.gray(r.url));
        res.send({
          name: api_data.youtube_search[0].TITLE,
          url: r.url,
        });
      })
      .catch((error) => console.log(c.bgRed("ERROR: "), c.gray(error.message)));
  });
});

app.get("/weather", async (req, res) => {
  // // handle GET request for fetching weather
  try {
    // Make a request to the OpenWeatherMap API
    // Create the API URL using the city name and API key
    const response = await axios.get(
      "http://api.openweathermap.org/data/2.5/weather?q=siliguri&appid=" +
        oapi_key
    );
    // Extract the relevant weather information
    const temperature = Math.round(response.data.main.temp - 273.15); // temperature in Celsius
    const humidity = response.data.main.humidity; // humidity percentage
    const wind_speed = response.data.wind.speed; // wind speed in meter/sec
    const weather_description = response.data.weather[0].description; // description of current weather
    // Print the weather forecast
    console.log(`Temperature: ${temperature}°C`);
    console.log(`Humidity: ${humidity}%`);
    console.log(`Wind Speed: ${wind_speed} m/s`);
    console.log(`Weather Description: ${weather_description}`);
    // send data in json
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

app.listen(port, () => {
  console.log(`Server listening at http://localhost:${port}`);
});
