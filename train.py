# Training
# Aidan Carter
# ASL Interpreter

from dataset import Dataset
from i3d import I3D
import torch

dev = torch.device("cpu")

def main():
    train = Dataset("./MS-ASL/MSASL_train.json", "./MS-ASL/MSASL_classes.json", "./Train")
    model = I3D().to(dev)

    optim = torch.optim.SGD(model.parameters(), lr=0.4, momentum=0.9)

    num_epochs = 10

    losses = []
    validations = []

    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"=== === === Epoch {epoch} === === === ")

        avg_loss = 0.0

        # Train
        for i in range(train.num_samples):
            print(f"\rProgress: {(i/train.num_samples)*100:.3f}%  |  Index: {i}", end="", flush=True)

            # Get input and class

            if not train.downloadVideo(i):
                continue

            try:
                input = train.extractFrames(i)[:,:64,:,:].to(dev)
            except RuntimeError:    # For if frames cannot be extracted ie corrupt video
                continue
            label = train.getLabel(i).to(dev)

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

        avg_loss /= train.num_samples
        print(f"\nTraining average loss: {avg_loss}")




if __name__ == "__main__":
    main()
    
