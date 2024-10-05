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
    batch_size = 5

    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"=== === === === === === === Epoch {epoch} === === === === === === ===")

        avg_loss = 0.0

        i = -1
        batch_count = 0
        # Train
        while i < (train.num_samples-1):
            batch_count+=1
            # Get input and class

            batch_input = []
            batch_labels = []

            j = batch_size
            while j > 0 and i < (train.num_samples-1):
                i+=1

                print(f"\rTraining: {((i+1)/train.num_samples)*100:.3f}%  |  Batch: {batch_count}  |  Sample: {i} / {train.num_samples}", end="", flush=True)

                if not train.downloadVideo(i):
                    continue
                
                try:
                    input = train.extractFrames(i)[:,:64,:,:].to(dev)
                except RuntimeError:    # For if frames cannot be extracted ie corrupt videos
                    train.skip.append(i)
                    continue

                # Extend video if less than 64 frames
                if input.size(1) < 64:
                    input = torch.cat((input, input[:,-1:,:,:].repeat(1,(64-input.size(1)),1,1)),1)
                
                batch_input.append(input)

                labels = train.getLabel(j).to(dev)
                batch_labels.append(labels)
                
                j-=1
            
            if len(batch_input) == 0:
                continue

            batch_input = torch.from_numpy(np.array(batch_input, np.float32))
            batch_labels = torch.from_numpy(np.array(batch_labels, np.float32))

            # Empty gradients
            optim.zero_grad()

            # Forward pass
            predict = model(batch_input)

            # Loss
            loss = torch.nn.functional.cross_entropy(predict, batch_labels, reduction="mean")

            # Backward pass
            loss.backward()

            # SGD update
            optim.step()

            # Summing losses
            avg_loss += loss.item()

        avg_loss /= batch_count
        print(f"\nTraining average loss: {avg_loss}")

        # Evaluations on validation set

        all_predict = []
        all_truth = []

        i = -1
        batch_count = 0

        with torch.no_grad():
            while i < (valid.num_samples-1):
                batch_count+=1
                # Get input and class

                batch_input = []
                batch_labels = []

                j = batch_size
                while j > 0 and i < (valid.num_samples-1):
                    i+=1

                    print(f"\rValidation: {((i+1)/valid.num_samples)*100:.3f}%  |  Batch: {batch_count}  |  Sample: {i} / {valid.num_samples}", end="", flush=True)

                    if not valid.downloadVideo(i):
                        continue
                    
                    try:
                        input = valid.extractFrames(i)[:,:64,:,:].to(dev)
                    except RuntimeError:    # For if frames cannot be extracted ie corrupt videos
                        valid.skip.append(i)
                        continue

                    # Extend video if less than 64 frames
                    if input.size(1) < 64:
                        input = torch.cat((input, input[:,-1:,:,:].repeat(1,(64-input.size(1)),1,1)),1)
                    
                    batch_input.append(input)

                    labels = valid.getLabel(j).to(dev)
                    batch_labels.append(labels)
                    
                    j-=1

                if len(batch_input) == 0:
                    continue
                
                batch_input = torch.from_numpy(np.array(batch_input, dtype=np.float32))
                batch_labels = torch.from_numpy(np.array(batch_labels, dtype=np.float32))

                predict = model(batch_input)
                all_predict.append(torch.argmax(predict, dim=1))
                all_truth.append(torch.argmax(batch_labels, dim=1))

        all_predict = torch.cat(all_predict)
        all_truth = torch.cat(all_truth)

        assert len(all_predict) == len(all_truth)

        total_correct = torch.sum(all_predict ==  all_truth) 
        valid_acc = total_correct / len(all_predict) * 100
        print(f"\nValidation accuracy: {valid_acc:.3f}% ({total_correct}/{len(all_predict)})")

        if valid_acc > best_acc or epoch == 0:
            best_acc = valid_acc
            torch.save(model.state_dict(), "./I3D_RGB_model.dat")


if __name__ == "__main__":
    main()
    
