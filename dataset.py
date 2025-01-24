# Dataloader Class File
# Aidan Carter
# ASL Interpreter

from yt_dlp import YoutubeDL, DownloadError
from json import loads
import os
import ffmpeg
import cv2
import numpy as np

class loggerOutputs:
    def __init__(self, index:int, savePath:str):
        self.index = index
        self.savePath = savePath
    def error(self, msg):
        with open(f"{self.savePath}/download.log", "a") as f:
            f.write(f"\n|{self.index}|{msg}")
    def warning(self, msg):
        with open(f"{self.savePath}/download.log", "a") as f:
            f.write(f"\n|{self.index}|{msg}")
    def debug(self, msg):
        with open(f"{self.savePath}/download.log", "a") as f:
            f.write(f"\n|{self.index}|{msg}")

class Sample:
    def __init__(self, index:int, entry:dict, savePath:str="."):
        self.entry = entry
        self.index = index
        self.savePath = savePath

    def isVideoDownloaded(self) -> bool:
        return os.path.exists(f"{self.savePath}/Videos/{self.index}.mp4")
    
    def isVideoProcessed(self) -> bool:
        return (os.path.exists(f"{self.savePath}/Videos/{self.index}_rgb.mp4") and
                os.path.exists(f"{self.savePath}/Videos/{self.index}_flow.mp4"))

    def getUrl(self) -> str:
        return self.entry["url"]

    def getTrimTimes(self) -> tuple[str, str]:
        return (str(self.entry["start_time"]), str(self.entry["end_time"]))

    def getBoundingBox(self) -> tuple[float, float, float, float]:
        return tuple(self.entry["box"])
    
    def getResolution(self) -> tuple[int, int]:
        probe = ffmpeg.probe(f"{self.savePath}/Videos/{self.index}.mp4")

        # Get video stream by selecting the stream in probe where codec_type is video
        stream = [stream for stream in probe["streams"] if stream["codec_type"] == "video"][0]

        return (stream["width"], stream["height"])

    def downloadVideo(self, retryAttempts:int=3) -> None:
        # Check if video is downloaded or processed
        if self.isVideoDownloaded() or self.isVideoProcessed():
            return
        
        # Options for Youtube Downloader
        # Downloads best video with no audio
        # Saves file as "{index}.mp4"
        ydlArgs = {
            "format" : "bestvideo[ext=mp4]",
            "outtmpl" : f"{self.savePath}/Videos/{self.index}.mp4",
            "noplaylist" : True,
            "logger" : loggerOutputs(self.index, self.savePath),
            "cookiefile" : "cookies.txt",
            # "verbose" : True,
        }

        for i in range(retryAttempts):
            try:
                with YoutubeDL(ydlArgs) as ydl:
                    ydl.download(self.getUrl())
                    break
            except DownloadError as e:
                if "Private video" in str(e) or "Video unavailable" in str(e):
                    break
                

    def processVideo(self) -> None:
        # Check if video is already processed
        if self.isVideoProcessed():
            return
        # Check if video is not downloaded
        if not self.isVideoDownloaded():
            raise ValueError(f"{self.index}.mp4 does not exist.")

        # Use helper functions to get resolution of video, the trim times for the entry, and the bounding box
        width, height = self.getResolution()
        startTime, endTime = self.getTrimTimes()
        boxY0, boxX0, boxY1, boxX1 = self.getBoundingBox()

        # Get pixel values for top left corner from normalized values and resolution
        cropY = int(boxY0 * height)
        cropX = int(boxX0 * width)

        # Get crop width and height from difference from normalized values of bottom right corner
        cropHeight = int(boxY1 * height) - cropY
        cropWidth = int(boxX1 * width) - cropX

        # Get file, trim, crop, and output
        out, err = (
            ffmpeg.input(f"{self.savePath}/Videos/{self.index}.mp4", ss=startTime, to=endTime)
            .filter("crop", cropWidth, cropHeight, cropX, cropY)
            .output(f"{self.savePath}/Videos/{self.index}_tmp.mp4")
            .run(overwrite_output=True, quiet=True)
        )

        # Log stdout and stderr from ffmpeg into process.log
        with open(f"{self.savePath}/process.log", "a") as f:
            if out != b'':
                f.write(f"|{self.index}|{out.decode('utf-8')}")
            if err != b'':
                f.write(f"|{self.index}|{err.decode('utf-8')}")

        # Resize video to fit into input of model
        # While resizing RGB video, calculate optical flow frames
        # This will also be input into seperate model

        # Input video
        video = cv2.VideoCapture(f"{self.savePath}/Videos/{self.index}_tmp.mp4")
        # Output for RGB video
        outputRGB = cv2.VideoWriter(
            f"{self.savePath}/Videos/{self.index}_rgb.mp4",
            cv2.VideoWriter_fourcc(*"avc1"),
            video.get(cv2.CAP_PROP_FPS),
            (224, 224)
        )
        # Output for Optical Flow video
        outputFlow = cv2.VideoWriter(
            f"{self.savePath}/Videos/{self.index}_flow.mp4",
            cv2.VideoWriter_fourcc(*"avc1"),
            video.get(cv2.CAP_PROP_FPS),
            (224, 224)
        )

        # Get initial frame and convert to grayscale for optical flow calculation
        ret, frame = video.read()
        resizedFrame = cv2.resize(frame, (224, 224))
        prevGray = cv2.cvtColor(resizedFrame, cv2.COLOR_BGR2GRAY)

        while video.isOpened():
            # Get frame
            ret, frame = video.read()
            # If no more frames returned, break out of loop
            if not ret:
                break

            # Resize frame
            resizedFrame = cv2.resize(frame, (224, 224))
            # Write frame to output
            outputRGB.write(resizedFrame)

            # Convert current frame to grayscale
            gray = cv2.cvtColor(resizedFrame, cv2.COLOR_BGR2GRAY)
            # Calulate optical flow
            flow = cv2.calcOpticalFlowFarneback(
                prevGray,
                gray,
                flow=None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )
            # Make current gray frame into previous frame
            prevGray = gray
            # Normalize flow for visualization and convert to HSV
            magnitude, angle = cv2.cartToPolar(flow[...,0], flow[...,1])
            # Make empty frame
            hsv = np.zeros_like(resizedFrame)
            # Make H (Hue) be the direction of optical flow
            # Take angle in radians and convert to degrees 
            hsv[...,0] = angle * (180 / np.pi) / 2
            # Make S (Saturation) be max value 255
            hsv[...,1] = 255
            # Make V (Value) be normalized magnitude
            hsv[...,2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            # Convert the colors from HSV to RGB
            flowRGB = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            # Write frame to output
            outputFlow.write(flowRGB)

        # Release all of the cv2 objects
        video.release()
        outputRGB.release()
        outputFlow.release()

        # Delete unnecessary files
        os.remove(f"{self.savePath}/Videos/{self.index}_tmp.mp4")
        os.remove(f"{self.savePath}/Videos/{self.index}.mp4")

class Dataset:
    def __init__(self, datasetJsonFile:str, savePath:str="."):

        with open(datasetJsonFile) as f:
            entries = f.read()

        self.entries:list[dict] = loads(entries)
        self.savePath = savePath
        
        if not os.path.exists(f"{self.savePath}"):
            os.mkdir(f"{self.savePath}")
        if not os.path.exists(f"{self.savePath}/Videos"):
            os.mkdir(f"{self.savePath}/Videos")

    def download(self):
        # Go through list of entries
        for index, entry in enumerate(self.entries):
            if index == 10:
                break
            # Make sample object
            sample = Sample(index, entry, self.savePath)
            try:
                sample.downloadVideo()
                sample.processVideo()
            except Exception as e:
                # Write exceptions to log
                with open(f"{self.savePath}/download.log", "a") as f:
                    f.write(f"\n|{index}|{type(e)} {e}")
            finally:
                # Delete sample object just to make sure garbage collection gets it
                del sample

class Dataloader:
    pass


if __name__ == "__main__":
    dataset = Dataset("./MS-ASL/MSASL_train.json", "./Train")
    dataset.download()