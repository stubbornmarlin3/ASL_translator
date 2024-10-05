# Training
# Aidan Carter
# ASL Interpreter

from dataset import Dataset
from i3d import I3D
import torch

import numpy as np
import cv2

dev = torch.device("cpu")

def main():
    train = Dataset("./MS-ASL/MSASL_train.json", "./MS-ASL/MSASL_classes.json", "./Train")
    valid = Dataset("./MS-ASL/MSASL_val.json", "./MS-ASL/MSASL_classes.json", "./Valid")

    model = I3D().to(dev)

    optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

    num_epochs = 100
    batch_size = 20

    train_iter = train.num_samples // batch_size
    valid_iter = valid.num_samples // batch_size

    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"=== === === === === Epoch {epoch} === === === === ===")

        avg_loss = 0.0

        # Train
        for i in range(train_iter):
            print(f"\rTraining: {((i)/train_iter)*100:.3f}%  |  Batch: [{i*batch_size}:{((i+1)*batch_size)-1}] / {train.num_samples}", end="", flush=True)

            # Get input and class

            if not train.downloadVideo(i):
                continue

            j = i*batch_size
            while j < ((i+1)*batch_size):
                
                if not train.downloadVideo(j):
                    continue
                
                try:
                    input = train.extractFrames(j)[:,:64,:,:].to(dev)
                except RuntimeError:    # For if frames cannot be extracted ie corrupt videos
                    train.skip.append(j)
                    continue

                


            try:
                input = train.extractFrames(i)[:,:64,:,:].to(dev)
            except RuntimeError:    # For if frames cannot be extracted ie corrupt video
                train.skip.append(i)
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

        avg_loss /= train_iter
        print(f"\nTraining average loss: {avg_loss}")

        # Evaluations on validation set

        all_predict = []
        all_truth = []

        with torch.no_grad():
            for i in range(valid_iter):
                print(f"\rValidating: {((i+1)/train_iter)*100:.3f}%  |  Batch: [{i}:{i+batch_size-1}] / {train.num_samples}", end="", flush=True)

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

        assert len(all_predict) == len(all_truth)

        total_correct = torch.sum(all_predict ==  all_truth) 
        valid_acc = total_correct / len(all_predict) * 100
        print(f"\nValidation accuracy: {valid_acc:.3f}% ({total_correct}/{len(all_predict)})")

        if valid_acc > best_acc or epoch == 0:
            best_acc = valid_acc
            torch.save(model.state_dict(), "./I3D_RGB_model.dat")


if __name__ == "__main__":
    main()
    
