from dataset import Dataloader, Dataset
from asl import ASL
import torch
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import sys

recordGrad = "--grad" in sys.argv
recordLoss = "--loss" in sys.argv
recordModel = "--model" in sys.argv


class ASLModel:
    def __init__(self, savePath:str, batchSize:int=10, subset:int=1000, flow:bool=False, loadModelName:str=None):
        self.max_acc = 0.0
        self.subset = subset
        self.batchSize = batchSize
        self.model = ASL(subset).to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        if loadModelName != None:
            self.model.load_state_dict(torch.load(f"{savePath}/{loadModelName}", weights_only=True))
        self.lossFunc = torch.nn.CrossEntropyLoss()
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, 50, 0.1)
        self.scaler = torch.amp.GradScaler()
        self.savePath = savePath
        self.flow = flow
        if recordGrad or recordLoss or recordModel:
            self.writer = SummaryWriter()
        
        if recordModel:
            self.writer.add_graph(self.model, torch.stack([Dataloader(Dataset("./MS-ASL/MSASL_train.json", "./Train"))[0][0]]))

        if not os.path.exists(self.savePath):
            os.mkdir(self.savePath)

    def test(self, validation:bool=False, epoch:int=None):
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

                if recordLoss:
                    self.writer.add_scalar(f"Epoch {epoch}/{'validLoss' if validation else 'testLoss'}", loss, batch)

                # Save average loss and number of correct predictions
                avgLoss.append(loss)
                correct.extend((pred.argmax(1) == y).tolist())
                print(f"\r{'Validating' if validation else 'Testing'}: {dataloader.currentIndex / len(dataloader) * 100:.3f}% | Batch: {batch} | Loss: {(sum(avgLoss)/len(avgLoss)):.6f} | Accuracy: {(correct.count(True)/len(correct)*100):.3f}%", end="", flush=True)
        # Print newline
        print()
        # Save model if validating    
        if validation:
            accuracy = correct.count(True) / len(correct) * 100   
            if accuracy > self.max_acc:   
                self.max_acc = accuracy 
                torch.save(self.model.state_dict(), f"{self.savePath}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{self.subset}_{f'{accuracy:.3f}'.replace('.','-')}_{'flow' if self.flow else 'rgb'}.ASL")
            if recordLoss:
                avgLoss = sum(avgLoss) / len(avgLoss)
                self.writer.add_scalar(f"{'Valid/accuracy' if validation else 'Test/accuracy'}", accuracy, epoch)
                self.writer.add_scalar(f"{'Valid/loss' if validation else 'Test/loss'}", avgLoss, epoch)

    def train(self, numEpochs:int=1):

        for epoch in range(numEpochs):
            print(f"--- Epoch {epoch+1} ---")
            self.model.train()

            avgLoss = []
            correct = []
            dataloader = Dataloader(Dataset("./MS-ASL/MSASL_train.json", "./Train"), self.subset, self.batchSize, self.flow)
            dataloader.train = True
            for batch, (X, y) in enumerate(dataloader):
                with torch.amp.autocast("cuda", enabled=True):
                    # Get predictions
                    pred = self.model(X)
                    # Calculate loss
                    loss = self.lossFunc(pred, y)

                if recordLoss:
                    self.writer.add_scalar(f"Epoch {epoch}/trainLoss", loss, batch)

                avgLoss.append(loss)
                correct.extend((pred.argmax(1) == y).tolist())

                # Back prop and optimize
                self.optim.zero_grad()
                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                if recordGrad:
                    try:
                        for name, param in self.model.named_parameters():
                            self.writer.add_histogram(f"Epoch {epoch}/grad_{name}", param.grad, batch)
                    except:
                        pass
                
                self.scaler.step(self.optim)
                self.scaler.update()

                print(f"\rTraining: {dataloader.currentIndex / len(dataloader) * 100:.3f}% | Batch: {batch} | Loss: {(sum(avgLoss)/len(avgLoss)):.6f} | Accuracy: {(correct.count(True)/len(correct)*100):.3f}%", end="", flush=True)

            if recordLoss:
                accuracy = correct.count(True) / len(correct) * 100      
                avgLoss = sum(avgLoss) / len(avgLoss)
                self.writer.add_scalar(f"Train/accuracy", accuracy, epoch)
                self.writer.add_scalar(f"Train/loss", avgLoss, epoch)
            # Print newline
            print()
            # Run test on validation set
            self.test(validation=True, epoch=epoch)
            self.scheduler.step()
            
        if recordGrad or recordLoss or recordModel:
            self.writer.close()

if __name__ == "__main__":
    model = ASLModel("./Models", batchSize=4, subset=10, flow=True)
    model.train(numEpochs=300)