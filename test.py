from dataset import Dataset
from i3d import I3D
import torch
import os
import random
from sys import argv

train = Dataset("./MS-ASL/MSASL_train.json", "./MS-ASL/MSASL_classes.json", "./Train")

i=3287
subset = 10

try:
    label = train.getLabel(i, subset)
except: # Not in subset
    exit(1)

if not train.downloadVideo(i):
    exit(1)

try:
    input = train.extractFrames(i)
    start_frame = random.randint(0,min(15, input.size(1)-1))   # Start at a random frame (towards the beginning of the video)
    input = input[:,start_frame:(start_frame+64),:,:]

    if random.randint(0,1):
        input = input.flip(dims=[3])    # Randomly flip video since ASL is the same left or right
except:    # For if frames cannot be extracted ie corrupt videos
    train.skip.append(i)
    exit(1)

# Extend video if less than 64 frames
if input.size(1) < 64:
    input = torch.cat((input, input[:,-1:,:,:].repeat(1,(64-input.size(1)),1,1)),1)

print(input)
print(label)