# Dataset Class File
# Aidan Carter
# ASL Interpreter

from yt_dlp import YoutubeDL
from yt_dlp.utils import download_range_func, DownloadError
from json import loads
import cv2
import torch
import os


class Dataset:
    "Class to manipulate datasets"

    def __init__(self, filePath:str, savePath:str, frameSize:tuple[int,int]=(224,224)) -> None:

        self.filePath = filePath
        self.savePath = savePath
        self.frameSize = frameSize

        with open(self.filePath) as f:
            data = f.read()

        self.data:list[dict] = loads(data)
        
    def getUrl(self, index:int) -> str:
        "returns the url of the video for data[index]"

        return self.data[index]["url"]
    
    def getClass(self, index:int) -> str:
        "returns the class of the video for data[index]"

        return self.data[index]["clean_text"]

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

        if self.isVideoDownloaded(index):
            print("Video Already Downloaded!")
            return True

        args = {
            "format" : "mp4/best",
            "download_ranges" : download_range_func(None, [self.getClipTime(index)]),
            "force_keyframes_at_cuts" : True,
            "outtmpl" : f"{self.savePath}/Videos/{index}.%(ext)s",
            "quiet" : True,
            "no_warnings" : True,
        }

        try:
            with YoutubeDL(args) as ydl:
                ydl.download(self.getUrl(index))
        except DownloadError:
            print("Cannot Download Video!")
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
            grayscaleFrame = cv2.cvtColor(resizedFrame, cv2.COLOR_BGR2GRAY)
            normFrame = grayscaleFrame / 255      # Normalize pixel values between 0 - 1

            frames.append(normFrame)

        cap.release()

        return torch.tensor(frames, dtype=torch.float32)


if __name__ == "__main__":
    training = Dataset("./MS-ASL/MSASL_train.json", "./Training")

    print(training.downloadVideo(0))

    frame = training.extractFrames(0)

    print(frame.shape)
    
    model = torch.nn.Sequential(
        torch.nn.Conv2d(84, 84, 7, 2)
    )

    print(frame)

    model(frame)
