# Dataset Class File
# Aidan Carter
# ASL Interpreter

from yt_dlp import YoutubeDL
from yt_dlp.utils import download_range_func, DownloadError
from json import loads
import cv2
import torch
import numpy as np
import os


class loggerOutputs:
    def error(msg):
        print(msg)
    def warning(msg):
        print(msg)
    def debug(msg):
        print(msg)

class Dataset:
    "Class to manipulate datasets"

    def __init__(self, filePath:str, labelPath:str, savePath:str, frameSize:tuple[int,int]=(224,224)) -> None:

        self.filePath = filePath
        self.labelPath = labelPath
        self.savePath = savePath
        self.frameSize = frameSize

        with open(self.filePath) as f:
            data = f.read()

        self.data:list[dict] = loads(data)

        with open(self.labelPath) as f:
            labels = f.read()
        
        self.labels:list = loads(labels)
        self.num_samples:int = len(self.data)

        self.skip:list = []
        
    def getUrl(self, index:int) -> str:
        "returns the url of the video for data[index]"

        return self.data[index]["url"]
    
    def getLabel(self, index:int) -> torch.Tensor:
        "returns the label of the video for data[index] as a one-hot vector"

        result = torch.zeros(len(self.labels))
        result[self.data[index]["label"]] = 1
        return result

    def getPixelCrop(self, index:int) -> tuple[slice, slice]:
        "returns slice object of pixel to crop video to. In format [heightStart:heightEnd, widthStart, widthEnd]"

        h = self.data[index]["height"]
        w = self.data[index]["width"]
        b = self.data[index]["box"]

        heightStart = int(b[0] * h)
        heightEnd = int(b[2] * h)

        widthStart = int(b[1] * w)
        widthEnd = int(b[3] * w)

        return slice(heightStart, heightEnd), slice(widthStart, widthEnd)
        
    def getClipTime(self, index:int) -> tuple[float, float]:
        "returns start and end time for clip"

        start = self.data[index]["start_time"]
        end = self.data[index]["end_time"]

        return start, end

    def isVideoDownloaded(self, index:int) -> bool:
        "returns whether video is downloaded or not"

        return os.path.exists(f"{self.savePath}/Videos/{index}.mp4")

    def downloadVideo(self, index:int) -> bool:
        "downloads clip from video for data[index]. returns True if download completes, False if video cannot be downloaded"

        if index in self.skip:
            return False

        if self.isVideoDownloaded(index):
            return True

        args = {
            "format" : "mp4/best",
            "download_ranges" : download_range_func(None, [self.getClipTime(index)]),
            "force_keyframes_at_cuts" : True,
            "outtmpl" : f"{self.savePath}/Videos/{index}.%(ext)s",
            "logger" : loggerOutputs,
        }

        try:
            with YoutubeDL(args) as ydl:
                ydl.download(self.getUrl(index))
        except DownloadError:
            self.skip.append(index)
            return False
        
        return True

    def deleteVideo(self, index:int) -> bool:
        "deletes clip from downloads (mainly to not fill up storage)"

        if self.isVideoDownloaded(index):
            os.remove(f"{self.savePath}/Videos/{index}.mp4")
        else:
            print("Could not delete video: File does not exist.")


    def extractFrames(self, index:int) -> torch.Tensor:
        "Each frame is cropped, converted to grayscale, and resized to _resizeTo. Pixel values are from 0-255 (not normalized to save disk space so remember to normalize for use in model). Frame is flattened and added to array of frames from video. Frames are saved to file by call to saveFrames(). Returns array of frames. Raises exception if video is not downloaded"

        if not self.isVideoDownloaded(index):
            raise Exception("Video Not Downloaded!")

        cap = cv2.VideoCapture(f"{self.savePath}/Videos/{index}.mp4")
        frames = []

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            croppedFrame = frame[self.getPixelCrop(index)]
            resizedFrame = cv2.resize(croppedFrame, self.frameSize)
            normFrame = resizedFrame / 255      # Normalize pixel values between 0 - 1

            frames.append(normFrame)

        cap.release()

        return torch.tensor(np.array(frames), dtype=torch.float32).permute(3,0,1,2)


if __name__ == "__main__":
    training = Dataset("./MS-ASL/MSASL_train.json", "./MS-ASL/MSASL_classes.json",  "./Training")

