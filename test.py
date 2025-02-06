from collections import Counter
from dataset import Dataset, Dataloader
import torch

dataset = Dataloader(Dataset("./MS-ASL/MSASL_train.json", "./Train"))

labels = [label.item() for _, label in dataset]
classCounts = Counter(labels)

classWeights = torch.zeros(1000)

for label, count in classCounts.items():
    classWeights[label] = 1.0 / count

classWeights = classWeights / classWeights.sum()

with open("./Train/classWeights.pt", "wb") as f:
    torch.save(classWeights, f)