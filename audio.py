import youtube_dl

video_url = 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'

ydl_opts = {
    'format': 'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '192',
    }]
}

with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    info_dict = ydl.extract_info(video_url, download=False)
    audio_url = info_dict['url']

print(audio_url)
