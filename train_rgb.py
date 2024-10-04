# Training
# Aidan Carter
# ASL Interpreter

from dataset import Dataset
from i3d import I3D
import torch

dev = torch.device("cuda")

def main():
    train = Dataset("./MS-ASL/MSASL_train.json", "./MS-ASL/MSASL_classes.json", "./Train")
    valid = Dataset("./MS-ASL/MSASL_val.json", "./MS-ASL/MSASL_classes.json", "./Valid")

    model = I3D().to(dev)

    optim = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    num_epochs = 100

    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"=== === === === === Epoch {epoch} === === === === ===")

        avg_loss = 0.0

        # Train
        for i in range(train.num_samples):
            print(f"\rTraining: {((i+1)/train.num_samples)*100:.3f}%  |  Video Index: {i}", end="", flush=True)

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

        # Evaluations on validation set

        all_predict = []
        all_truth = []

        with torch.no_grad():
            for i in range(valid.num_samples):
                print(f"\rValidating: {((i+1)/valid.num_samples)*100:.3f}%  |  Video Index: {i}", end="", flush=True)

                if not valid.downloadVideo(i):
                    continue

                try:
                    input = valid.extractFrames(i)[:,:64,:,:].to(dev)
                except RuntimeError:    # For if frames cannot be extracted ie corrupt video
                    continue
                label = valid.getLabel(i).to(dev)

                # Extend frames if less than 64
                if input.shape[1] < 64:
                    input = torch.cat((input, input[:,-1:,:,:].repeat(1,(64 - input.size(1)),1,1)), 1)

                predict = model(input)
                all_predict.append(torch.argmax(predict))
                all_truth.append(torch.argmax(label))

        all_predict = torch.stack(all_predict)
        all_truth = torch.stack(all_truth)

        valid_acc = torch.sum(all_predict ==  all_truth) / valid.num_samples * 100
        print(f"\nValidation accuracy: {valid_acc:.3f}%")

        if valid_acc > best_acc or epoch == 0:
            best_acc = valid_acc
            torch.save(model.state_dict(), "./I3D_RGB_model.dat")


if __name__ == "__main__":
    main()
    
