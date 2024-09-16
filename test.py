from yt_dlp import YoutubeDL
from yt_dlp.utils import download_range_func
import json

args = {
    "format" : "mp4/best",
    "download_ranges" : download_range_func(None, [(407.92, 409.126)]),
    "force_keyframes_at_cuts" : True
}

URLS = ['https://www.youtube.com/watch?v=jQb9NL9_S6U']
with YoutubeDL(args) as ydl:
    ydl.download(URLS)