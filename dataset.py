# Dataloader Class File
# Aidan Carter
# ASL Interpreter

from yt_dlp import YoutubeDL, DownloadError
from json import loads
import os
import ffmpeg
import cv2
import numpy as np
import torch
from torchcodec.decoders import VideoDecoder
import random

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
        "Returns True if video is in Videos folder"
        return os.path.exists(f"{self.savePath}/Videos/{self.index}.mp4")
    
    def isVideoProcessed(self) -> bool:
        "Returns True if video has been processed into rgb and flow frames. Videos are stored in Videos folder"
        return (os.path.exists(f"{self.savePath}/Videos/{self.index}_rgb.mp4") and
                os.path.exists(f"{self.savePath}/Videos/{self.index}_flow.mp4"))

    def getUrl(self) -> str:
        "Returns the url from the sample"
        return self.entry["url"]

    def getTrimTimes(self) -> tuple[str, str]:
        "Returns a tuple of the start and end time for the clip from the sample"
        return (str(self.entry["start_time"]), str(self.entry["end_time"]))

    def getBoundingBox(self) -> tuple[float, float, float, float]:
        "Returns a tuple of bounding box coords in format [Y0,X0,Y1,X1]"
        return tuple(self.entry["box"])
    
    def getResolution(self) -> tuple[int, int]:
        "Returns the resolution of the downloaded video as a tuple [width,height]"
        probe = ffmpeg.probe(f"{self.savePath}/Videos/{self.index}.mp4")

        # Get video stream by selecting the stream in probe where codec_type is video
        stream = [stream for stream in probe["streams"] if stream["codec_type"] == "video"][0]

        return (stream["width"], stream["height"])
    
    def getLabel(self) -> int:
        "Returns the label from the sample"
        return self.entry["label"]
    
    def load(self, flow:bool=False) -> torch.Tensor:
        """
        Returns a tensor of normalized pixel values from the downloaded and processed video.\n
        Tensor will be in shape [3, 64, 224, 224] with the 64 frames being contiguous but from random start point\n
        Will return values from the RGB processed video by default.\n
        If flow is True, will return optical flow frames instead.\n
        """

        # Make decoder for video
        decoder = VideoDecoder(f"{self.savePath}/Videos/{self.index}_{'flow' if flow else 'rgb'}.mp4", device=('cuda' if torch.cuda.is_available() else 'cpu'))

        # Make tensor of all frames decoded
        frames = decoder[:]
        # If less than 64 frames, extend last frame to get to 64 frames
        # Otherwise select a 64 frame clip from the video at random
        if frames.size(0) < 64:
            frames = torch.cat((frames, frames[-1].repeat(64-frames.size(0),1,1,1)))
        else:
            num = random.randint(0,max(0,frames.size(0)-64))    # To prevent index errors
            frames = frames[num:num+64]

        # Permute to get [channels, frames, width, height] and then normalize pixel values
        return frames.permute(1,0,2,3)

    def downloadVideo(self, retryAttempts:int=3) -> None:
        """
        Downloads video to {savePath}/Videos folder with name {index}.mp4\n
        Will exit program with exit code 1 if sign in is needed or content is blocked\n
        Can specify retryAttempts (default is 3)
        """
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
            "sleep_interval" : 2,
            "max_sleep_interval" : 5,
            "retries" : 2,
            "limit_rate" : "50K",
            "concurrent_fragments" : 1,
            # "verbose" : True,
        }

        for i in range(retryAttempts):
            try:
                with YoutubeDL(ydlArgs) as ydl:
                    ydl.download(self.getUrl())
                    break
            except DownloadError as e:
                if "content" in str(e):
                    # Print error index then exit
                    print(f"\n{self.index}", end="")
                    exit(1)
                if "Sign in to confirm" in str(e):
                    exit(1)
                if "unavailable" in str(e) or "private" in str(e):
                    break

    def processVideo(self) -> None:
        """
        Processes {index}.mp4 into two videos {index}_rgb.mp4 and {index}_flow.mp4\n
        Deletes {index.mp4} when complete, as it is not needed anymore
        """
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
            cv2.VideoWriter_fourcc(*"mp4v"),
            video.get(cv2.CAP_PROP_FPS),
            (224, 224)
        )
        # Output for Optical Flow video
        outputFlow = cv2.VideoWriter(
            f"{self.savePath}/Videos/{self.index}_flow.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
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

    def download(self, start:int=0):
        """
        Downloads all videos in dataset specified to save path specified.\n
        Can specify a starting index (defaults to 0)
        """
        # Go through list of entries
        numErr = 0
        for index in range(start, len(self.entries)):
            # Print status
            print(f"\rDownloading: {(index)/(len(self.entries)-1)*100:.3f}% | Completed: {index} / {len(self.entries)-1} | Errors: {numErr}", end="", flush=True)
            # Make sample object
            sample = Sample(index, self.entries[index], self.savePath)
            try:
                sample.downloadVideo()
                sample.processVideo()
            except Exception as e:
                # Write exceptions to log
                numErr += 1
                with open(f"{self.savePath}/download.log", "a") as f:
                    f.write(f"\n|{index}|{type(e)} {e}")
            finally:
                # Delete sample object just to make sure garbage collection gets it
                del sample
        print("\n0", end="") # Print 0 for bash exit condition

class Dataloader:
    def __init__(self, dataset:Dataset, subset:int, batchSize:int=1, flow:bool=False):
        self.dataset = dataset
        self.subset = subset
        self.batchSize = batchSize
        self.flow = flow
        self.currentIndex = 0

    def __len__(self):
        return len(self.dataset.entries)

    def __iter__(self):
        "Return iterable object"
        return self
    
    def __getitem__(self, index:int) -> tuple[torch.Tensor, int]:
        "Return tensor of loaded video at index and its label"
        if isinstance(index, slice):
            raise ValueError("Index cannot be slice")
        try:
            # Try to make and load sample
            sample = Sample(index, self.dataset.entries[index], self.dataset.savePath)
            if sample.getLabel() < self.subset:
                return (sample.load(self.flow), sample.getLabel())
            else:
                return None
        except IndexError:
            # If index out of range, entry does not exist to raise IndexError
            raise
        except ValueError:
            # If ValueError, video does not exist so just return None
            return None
        finally:
            # Make sure sample is deleted to free memory
            del sample
    
    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns next batch of tensors\n 
        Will attempt to return batchSize tensors (could be less if at end of samples)\n
        Raises StopIteration if there are no more tensors to return
        """
        # If current index is at end of entries, raise StopIteration
        if self.currentIndex == len(self.dataset.entries):
            raise StopIteration
        
        batchVideos = []
        batchLabels = []
        while len(batchVideos) < self.batchSize and self.currentIndex < len(self.dataset.entries):
            # Load video
            item = self.__getitem__(self.currentIndex)
            # Increment index
            self.currentIndex+=1
            # If no video to load, then just get next video
            if item == None:
                continue
            # Append loaded video to batch list
            batchVideos.append(item[0])
            batchLabels.append(item[1])

        # If batch is empty, raise StopIteration as there is no more videos
        if not batchVideos:
            raise StopIteration
        
        return (torch.stack(batchVideos), torch.tensor(batchLabels, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")))


if __name__ == "__main__":
    dataset = Dataset("./MS-ASL/MSASL_val.json", "./Valid")
    dataset.download()
    # e = dataset.entries
    # sample = Sample(17000, e[17000], "./Train")    
    # print(sample.load())
    # dataloader = Dataloader(dataset, 100, 3)
    # print(len(dataloader))
    # for batch, labels in dataloader:
    #     print(batch, labels)

    