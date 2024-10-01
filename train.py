# Training
# Aidan Carter
# ASL Interpreter

from dataset import Dataset
from googLeNet import GoogLeNetCNN
import torch

def main():
    train = Dataset("./MS-ASL/MSASL_train.json", "./MS-ASL/MSASL_classes.json", "./Train")
    model = GoogLeNetCNN().to(torch.device("mps"))

    optim = torch.optim.SGD(model.parameters(), lr=0.4)

    num_epochs = 4

    losses = []
    validations = []

    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"=== === === Epoch {epoch} === === === ")

        avg_loss = 0.0

        # Train
        for i in range(40):

            # Get input and class

            if not train.downloadVideo(i):
                continue

            input = train.extractFrames(i)[:20].to(torch.device("mps"))
            label = train.getLabel(i).to(torch.device("mps"))

            # Empty gradients
            optim.zero_grad()

            # Forward pass
            predict = model(input)

            # Loss
            loss = torch.nn.functional.cross_entropy(predict, label, reduction="mean")

            # Backward pass
            loss.backward()

            # SGD update
            optim.step()

            # Summing losses
            avg_loss += loss.item()

        avg_loss /= 40
        print(f"Training average loss: {avg_loss}")




if __name__ == "__main__":
    main()