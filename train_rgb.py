# Training
# Aidan Carter
# ASL Interpreter

from dataset import Dataset
from i3d import I3D
import torch
import os
import random
from sys import argv

if torch.cuda.is_available():
    dev = torch.device("cuda")
else:
    dev = torch.device("cpu")

def main():
    train = Dataset("./MS-ASL/MSASL_train.json", "./MS-ASL/MSASL_classes.json", "./Train")
    valid = Dataset("./MS-ASL/MSASL_val.json", "./MS-ASL/MSASL_classes.json", "./Valid")

    # Should be 100, 200, 500, or 1000
    subset = 10

    model = I3D(subset).to(dev)

    optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

    # For loading model from previous state after program is stopped (to not restart training unless need to)
    if "-l" in argv or "--load" in argv:
        try:
            model_savepath = argv[argv.index("-l")+1]
        except:
            model_savepath = argv[argv.index("--load")+1]
        if os.path.exists(model_savepath):
            params = torch.load(model_savepath, weights_only=True)

            model.load_state_dict(params["model_state"])
            optim.load_state_dict(params["optim_state"])
            last_loss = params["loss"]
            last_epoch = params["epoch"]
            last_valid_acc = params["valid_acc"]
            print(f"Loading from saved model: {model_savepath}\nLast epoch: {last_epoch}\nLast training loss: {last_loss}\nLast validation accuracy: {last_valid_acc:.3f}%")
        else:
            print(f"Model does not exist: {model_savepath}")
            exit(1)
    else:
        model_savepath = "./rgb_model.pt"
    
    num_epochs = 10
    batch_size = 1

    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"=== === === === === === === Epoch {epoch} === === === === === === ===")

        avg_loss = 0.0
        model.train()

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

                try:
                    label = train.getLabel(i, subset)
                except: # Not in subset
                    continue

                if not train.downloadVideo(i):
                    continue

                try:
                    input = train.extractFrames(i)
                    start_frame = random.randint(0,min(15, input.size(1)-1))   # Start at a random frame (towards the beginning of the video)
                    input = input[:,start_frame:(start_frame+64),:,:]

                    if random.randint(0,1):
                        input = input.flip(dims=[3])    # Randomly flip video since ASL is the same left or right
                except:    # For if frames cannot be extracted ie corrupt videos
                    train.skip.append(i)
                    continue

                # Extend video if less than 64 frames
                if input.size(1) < 64:
                    input = torch.cat((input, input[:,-1:,:,:].repeat(1,(64-input.size(1)),1,1)),1)
                
                batch_input.append(input)
                batch_labels.append(label)
            
                j-=1
            
            if len(batch_input) == 0:
                continue

            batch_input = torch.stack(batch_input).to(dev)
            batch_labels = torch.tensor(batch_labels, dtype=int).to(dev)

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
            model.eval()
            while i < (valid.num_samples-1):
                batch_count+=1
                # Get input and class

                batch_input = []
                batch_labels = []

                j = batch_size
                while j > 0 and i < (valid.num_samples-1):
                    i+=1

                    print(f"\rValidation: {((i+1)/valid.num_samples)*100:.3f}%  |  Batch: {batch_count}  |  Sample: {i} / {valid.num_samples}", end="", flush=True)

                    try:
                        label = valid.getLabel(i, subset)
                    except: # Not in subset
                        continue

                    if not valid.downloadVideo(i):
                        continue
                    
                    try:
                        input = valid.extractFrames(i)
                        start_frame = random.randint(0,min(15, input.size(1)-1))   # Start at a random frame (towards the beginning of the video)
                        input = input[:,start_frame:(start_frame+64),:,:]

                        if random.randint(0,1):
                            input = input.flip(dims=[3])    # Randomly flip video since ASL is the same left or right
                    except:    # For if frames cannot be extracted ie corrupt videos
                        valid.skip.append(i)
                        continue

                    # Extend video if less than 64 frames
                    if input.size(1) < 64:
                        input = torch.cat((input, input[:,-1:,:,:].repeat(1,(64-input.size(1)),1,1)),1)
                    
                    batch_input.append(input)
                    batch_labels.append(label)
                    
                    j-=1

                if len(batch_input) == 0:
                    continue
                
                batch_input = torch.stack(batch_input).to(dev)
                batch_labels = torch.tensor(batch_labels, dtype=int).to(dev)

                predict = model(batch_input)
                all_predict.append(torch.argmax(predict, dim=1))
                all_truth.append(batch_labels)

        all_predict = torch.cat(all_predict)
        all_truth = torch.cat(all_truth)

        assert len(all_predict) == len(all_truth)

        total_correct = torch.sum(all_predict ==  all_truth) 
        valid_acc = total_correct / len(all_predict) * 100
        print(f"\nValidation accuracy: {valid_acc:.3f}% ({total_correct}/{len(all_predict)})")

        if valid_acc > best_acc or epoch == 0:
            best_acc = valid_acc
            if os.path.exists(model_savepath):
                os.rename(model_savepath, f"{model_savepath}.old")
            torch.save({
                "epoch" : epoch,
                "model_state" : model.state_dict(),
                "optim_state" : optim.state_dict(),
                "loss" : loss,
                "valid_acc" : valid_acc
            }, model_savepath)



if __name__ == "__main__":
    main()
    
