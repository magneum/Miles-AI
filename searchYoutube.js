import ytSearch from "yt-search";

export const searchVideo = async (keyword, numVideos = 1) => {
  const searchResults = await ytSearch({ query: keyword });
  const { videos } = searchResults;
  return videos.slice(0, numVideos);
};

export const getVideoDetails = async (videoId) => {
  const searchResults = await ytSearch({ videoId });
  const { videos } = searchResults;
  return videos[0];
};

export const getPlaylistDetails = async (playlistId, numVideos = 1) => {
  const searchResults = await ytSearch({ listId: playlistId });
  const { playlists } = searchResults;
  return playlists.slice(0, numVideos);
};

export const searchVideosByGenre = async (genre, numVideos = 1) => {
  const searchResults = await ytSearch({ genre });
  const { videos } = searchResults;
  return videos.slice(0, numVideos);
};

export const getVideosFromChannel = async (channelId, numVideos = 1) => {
  const searchResults = await ytSearch({ channelId });
  const { videos } = searchResults;
  return videos.slice(0, numVideos);
};

export const getVideoDetailsByUrl = async (url) => {
  const searchResults = await ytSearch({ videoUrl: url });
  const { videos } = searchResults;
  return videos[0];
};

export const getRelatedVideos = async (relatedToVideoId, numVideos = 1) => {
  const searchResults = await ytSearch({ related: relatedToVideoId });
  const { videos } = searchResults;
  return videos.slice(0, numVideos);
};
