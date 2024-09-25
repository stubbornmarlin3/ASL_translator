from yt_dlp import YoutubeDL
from yt_dlp.utils import download_range_func
import json
import cv2
import numpy as np
import os

filepath = "./MS-ASL/MSASL_train.json"

with open(filepath) as f:
    data = f.read()

data = json.loads(data)

for i in range(12):
    print(data[i]["start_time"])
    print(data[i]["end_time"])
    print(data[i]["height"])
    print(data[i]["width"])
    print(data[i]["box"])
    print(data[i]["clean_text"])
    print(data[i]["url"])

    if os.path.isfile(f"./Videos/{data[i]['clean_text']}.mp4"):
        continue

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

    cap = cv2.VideoCapture(f"./Videos/{data[i]['clean_text']}.mp4")
    frames = []
    while True:
        ret, frame = cap.read()

        cv2.imshow("Frame", frame)
        cv2.waitKey(0)

        if not ret:
            break

        frame = np.array([cv2.cvtColor(cv2.resize(frame[int(box[0]*height):int(box[2]*height), int(box[1]*width):int(box[3]*width)], (256, 256)), cv2.COLOR_BGR2GRAY)])


        frames.append(frame.flatten())

    with open("./data.gz", "ab") as f:
        np.savetxt(f, np.array(frames), fmt="%3u")

    cap.release()


with open("./data.gz", "rb") as f:
    data = np.loadtxt(f)

print(data.shape)

image = np.array(data[0])
image = image.reshape((256,256))

cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()