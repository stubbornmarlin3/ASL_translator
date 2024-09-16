from yt_dlp import YoutubeDL
from yt_dlp.utils import download_range_func
import json
import cv2

filepath = "./MS-ASL/MSASL_train.json"

with open(filepath) as f:
    data = f.read()

data = json.loads(data)

for i in range(5):
    print(data[i]["start_time"])
    print(data[i]["end_time"])
    print(data[i]["height"])
    print(data[i]["width"])
    print(data[i]["box"])
    print(data[i]["clean_text"])
    print(data[i]["url"])

    args = {
    "format" : "mp4/best",
    "download_ranges" : download_range_func(None, [(data[i]["start_time"], data[i]["end_time"])]),
    "force_keyframes_at_cuts" : True,
    "outtmpl" : f"./Videos/{data[i]['clean_text']}.%(ext)s" 
    }

    try:
        with YoutubeDL(args) as ydl:
            ydl.download(data[i]["url"])
    except Exception as e:
        print(e)
        continue

    height = data[i]["height"]
    width = data[i]["width"]
    box = data[i]["box"]

    print(box[0]*width, box[1]*height, box[2]*width, box[3]*height)

    cap = cv2.VideoCapture(f"./Videos/{data[i]['clean_text']}.mp4")
    ret, frame = cap.read()

    frame = frame[int(box[1]*height):int(box[3]*height), int(box[0]*width):int(box[2]*width)]

    cv2.imshow("Image", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

