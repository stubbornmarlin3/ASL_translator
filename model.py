from dataset import Dataloader, Dataset
from i3d import I3D
import torch
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

class ASLModel:
    def __init__(self, savePath:str, batchSize:int=10, subset:int=1000, flow:bool=False, loadModelName:str=None):
        self.subset = subset
        self.batchSize = batchSize
        self.model = I3D(subset).to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        if loadModelName != None:
            self.model.load_state_dict(torch.load(f"{savePath}/{loadModelName}", weights_only=True))
        self.lossFunc = torch.nn.CrossEntropyLoss()
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=3e-4, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=10)
        self.scaler = torch.amp.GradScaler()
        self.savePath = savePath
        self.flow = flow

        if not os.path.exists(self.savePath):
            os.mkdir(self.savePath)


    def test(self, validation:bool=False):
        if validation:
            dataloader = Dataloader(Dataset("./MS-ASL/MSASL_val.json", "./Valid"), self.subset, self.batchSize, self.flow)
        else:
            dataloader = Dataloader(Dataset("./MS-ASL/MSASL_test.json", "./Test"), self.subset, self.batchSize, self.flow)

        self.model.eval()
        avgLoss = []
        correct = []
        with torch.no_grad():
            for batch, (X, y) in enumerate(dataloader):
                # Make predictions
                pred = self.model(X)
                # Calculate loss
                loss = self.lossFunc(pred, y)

                # Save average loss and number of correct predictions
                avgLoss.append(loss)
                correct.extend((pred.argmax(1) == y).tolist())
                print(f"\r{'Validating' if validation else 'Testing'}: {dataloader.currentIndex / len(dataloader) * 100:.3f}% | Batch: {batch} | Loss: {loss.item():.6f}", end="", flush=True)

        # Calculate average loss and accuracy
        avgLoss = sum(avgLoss) / len(avgLoss)
        accuracy = correct.count(True) / len(correct) * 100       
        # Save model if validating    
        if validation:
            torch.save(self.model.state_dict(), f"{self.savePath}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{self.subset}_{f'{accuracy:.3f}'.replace('.','-')}_{'flow' if self.flow else 'rgb'}.i3d")
        print(f" | Average Loss: {avgLoss:.6f} | Accuracy: {accuracy:.3f}%")


    def train(self, numEpochs:int=1):
        writer = SummaryWriter()

        for epoch in range(numEpochs):
            print(f"--- Epoch {epoch+1} ---")
            self.model.train()

            dataloader = Dataloader(Dataset("./MS-ASL/MSASL_train.json", "./Train"), self.subset, self.batchSize, self.flow)
            for batch, (X, y) in enumerate(dataloader):
                
                with torch.amp.autocast(device_type="cuda"):
                    # Get predictions
                    assert torch.isfinite(X).all(), "Missing values"
                    pred = self.model(X)
                    # Calculate loss
                    loss = self.lossFunc(pred, y)
                    print(f"Predicted: {pred.argmax(1)} | Actual: {y}")
                    assert not torch.isnan(loss).any(), "NaN detected in loss"

                # Back prop and optimize
                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)


                for name, param in self.model.named_parameters():
                    writer.add_histogram(f'gradients/{name}', param.grad, epoch)

                self.scaler.step(self.optim)
                self.scaler.update()
                self.optim.zero_grad()

                print(f"\rTraining: {dataloader.currentIndex / len(dataloader) * 100:.3f}% | Batch: {batch} | Loss: {loss.item():.6f}", end="", flush=True)
            # Print newline
            print()
            # Run test on validation set
            self.test(validation=True)

        writer.close()

if __name__ == "__main__":
    model = ASLModel("./Models", batchSize=1, subset=1000)
    model.train(numEpochs=10)