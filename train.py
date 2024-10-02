# Training
# Aidan Carter
# ASL Interpreter

from dataset import Dataset
from i3d import I3D
import torch

#Test

def main():
    train = Dataset("./MS-ASL/MSASL_train.json", "./MS-ASL/MSASL_classes.json", "./Train")
    model = I3D().to(torch.device("cuda"))

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
            print(f"{i}.", end="", flush=True)

            # Get input and class

            if not train.downloadVideo(i):
                continue

            input = train.extractFrames(i)[:,:64,:,:].to(torch.device("cuda"))
            label = train.getLabel(i).to(torch.device("cuda"))

            # Extend frames if less than 64
            if input.shape[1] < 64:
                input = torch.cat((input, input[:,-1:,:,:].repeat(1,(64 - input.size(1)),1,1)), 1)

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

            print(":", end="")

        avg_loss /= 40
        print(f"Training average loss: {avg_loss}")




if __name__ == "__main__":
    main()
    
