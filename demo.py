import cv2
import numpy as np
import torch
from collections import deque
from asl import ASL
import time
import threading

classes = ["hello", "nice", "teacher", "eat", "no", "happy", "like", "orange", "want", "deaf", "school"]
currentPred = ""
threadLock = threading.Lock()
threadRunning = False

def modelEval(frameBuffer):

    global currentPred, threadRunning
    with threadLock:

        X = np.stack(frameBuffer, axis=0)
        X = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
        X = X.permute(0,4,1,2,3) / 255.0

        with torch.no_grad():
            pred = model(X)
            pred = torch.argmax(pred, dim=1).item()
            currentPred = classes[pred]
        threadRunning = False

def cameraPredict(model):
    global threadRunning, currentPred
    lastPredTime = time.time()
    camera = cv2.VideoCapture(0)
    frameBuffer = deque(maxlen=64)
    prevGray = None

    while True:
        ret, frame = camera.read()

        if not ret:
            break
        
        height, width, channels = frame.shape
        frame = frame[(height-1080)//2:(height+1080)//2, (width-1080)//2:(width+1080)//2]

        output = frame

        frame = cv2.resize(frame, (224,224))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prevGray is not None:
            
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

            # Normalize flow for visualization and convert to HSV
            magnitude, angle = cv2.cartToPolar(flow[...,0], flow[...,1])
            # Make empty frame
            hsv = np.zeros_like(frame)
            # Make H (Hue) be the direction of optical flow
            # Take angle in radians and convert to degrees 
            hsv[...,0] = angle * (180 / np.pi) / 2
            # Make S (Saturation) be max value 255
            hsv[...,1] = 255
            # Make V (Value) be normalized magnitude
            hsv[...,2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            # Convert the colors from HSV to RGB
            frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            frameBuffer.append(frame)

        prevGray = gray

        if not threadRunning and len(frameBuffer) == 64 and (time.time()-lastPredTime >= 0.5):

            threadRunning = True
            lastPredTime = time.time()

            thread = threading.Thread(
                target = modelEval,
                args=(list(frameBuffer),),
            )
            thread.start()

        cv2.putText(output, f"{currentPred}", (450, 1000), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 50, 0), 2)
        cv2.imshow("Camera", output)
        cv2.imshow("Optical Flow", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    model = ASL(10)
    model.load_state_dict(torch.load("./runs/Apr11_14-13-34_serverpc/2025-04-12_11-06-26_10_71-429_flow.ASL", map_location="cpu"))
    model.eval()
    cameraPredict(model)