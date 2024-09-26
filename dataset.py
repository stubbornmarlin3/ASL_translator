# Dataset Class File
# Aidan Carter
# ASL Interpreter

from yt_dlp import YoutubeDL
from yt_dlp.utils import download_range_func, DownloadError
from json import loads
import cv2
import numpy as np
import os


class Dataset:
    "Class to manipulate datasets"

    def __init__(self, filePath:str, savePath:str, frameSize:tuple[int,int]=(256,256)) -> None:

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

    def saveFrames(self, index:int, frames:np.ndarray) -> None:
        "save array of frames to file (to be used as input for AI model)"

        if not os.path.exists(f"{self.savePath}/Frames"):
            os.mkdir(f"{self.savePath}/Frames")

        with open(f"{self.savePath}/Frames/{index}.exfr", "wb") as f:
            for pixelValue in frames.flatten():
                f.write(int(pixelValue).to_bytes())

    def loadFrames(self, index:int) -> np.ndarray:
        "load array of frames from file. Frames arrays are loaded from bytes. Each frame is (height * width) bytes"

        with open(f"{self.savePath}/Frames/{index}.exfr", "rb") as f:
            frames = []
            # Using walrus operator to assign while testing condition. Create frame using list comprehension by reading in x number of bytes (equal to height * width of frame) which converts each byte to an int, then assigns it to frame while testing if frame can be constructed (ie there is bytes to be read in the file). 
            while frame := [byte for byte in f.read(self.frameSize[0] * self.frameSize[1])]:
                frames.append(frame)

        return np.array(frames)

    def deleteFrames(self, index:int) -> None:
        "delete file of frames (usually not needed but to save storage I guess. These files shouldn't be too big though)"

        if os.path.exists(f"{self.savePath}/Frames/{index}.exfr"):
            os.remove(f"{self.savePath}/Frames/{index}.exfr")
        else:
            print("Could not delete frames: File does not exist.")

    def extractFrames(self, index:int) -> np.ndarray:
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

            flattenedFrame = np.array([grayscaleFrame]).flatten()
            frames.append(flattenedFrame)

        cap.release()

        return np.array(frames)


if __name__ == "__main__":
    training = Dataset("./MS-ASL/MSASL_train.json", "./Training")

    print(training.getUrl(0))
    print(training.getClass(0))

    test = np.ones((500,500))

    print(training.getPixelCrop(0))
    print(test[training.getPixelCrop(0)].shape)

    print(training.getClipTime(0))

    # print(training.isVideoDownloaded(0))
    # os.system("mkdir -p ./Training/Videos && touch ./Training/Videos/0.mp4")
    # print(training.isVideoDownloaded(0))
    # os.system("rm -rf ./Training")

    print(training.downloadVideo(0))

    # training.deleteVideo(4)

    training.deleteFrames(0)
    frame = training.extractFrames(0)
    training.saveFrames(0,frame)
    frame2 = training.loadFrames(0)

    print(frame == frame2)

    for i in range(10):
        training.downloadVideo(i)
        try:
            frame = training.extractFrames(i)
        except:
            continue
        training.saveFrames(i,frame)


    
