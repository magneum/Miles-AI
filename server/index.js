// > SINGLETRACK() function takes a song name as input and returns the first video search result for that song.
// > RANDOMGENRETRACK() function takes a music genre as input and returns a random video search result for that genre.
// > RANDOMTRACK() function selects a random music genre from a predefined list and returns a random video search result for that genre.
// > CUSTOMTRACK() function takes a search query and a number of desired search results as inputs, and returns an array of videos that match the search query.

import chalk from "chalk";
import ytSearch from "yt-search";

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

export async function singleTrack(songName) {
  const results = await ytSearch(songName);
  return results.videos[0];
}

export async function randomTrack() {
  const genreIndex = Math.floor(Math.random() * musicGenres.length);
  const { genre, query } = musicGenres[genreIndex];
  const results = await ytSearch(query);
  const videoIndex = Math.floor(Math.random() * results.videos.length);
  return { genre: genre, video: results.videos[videoIndex] };
}

export async function randomGenreTrack(genre) {
  const { query } = musicGenres.find(
    (musicGenre) => musicGenre.genre === genre.toLowerCase()
  );
  const results = await ytSearch(query);
  const videoIndex = Math.floor(Math.random() * results.videos.length);
  return { genre, video: results.videos[videoIndex] };
}

export async function customTrack(query, totalTracks) {
  const results = await ytSearch(query);
  const videos = results.videos.slice(0, totalTracks);
  return videos;
}

singleTrack("Bohemian Rhapsody")
  .then((video) => {
    console.log(
      `Single track result for ${chalk.green(
        "Bohemian Rhapsody"
      )}: ${chalk.yellow(video.title)}`
    );
  })
  .catch((err) => {
    console.error(chalk.red(err));
  });

randomTrack()
  .then((result) => {
    console.log(
      `Random track result for ${chalk.green(result.genre)}: ${chalk.yellow(
        result.video.title
      )}`
    );
  })
  .catch((err) => {
    console.error(chalk.red(err));
  });

randomGenreTrack("hip hop")
  .then((result) => {
    console.log(
      `Random track result for ${chalk.green(result.genre)}: ${chalk.yellow(
        result.video.title
      )}`
    );
  })
  .catch((err) => {
    console.error(chalk.red(err));
  });

customTrack("guitar solos", 5)
  .then((tracks) => {
    console.log(chalk.green("Custom tracks found:"));
    tracks.forEach((track, index) => {
      console.log(chalk.yellow(`${index + 1}. ${track.title}`));
    });
  })
  .catch((error) => {
    console.error(chalk.red(`Error: ${error}`));
  });
