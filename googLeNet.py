# GoogLeNet CNN Class File
# Aidan Carter
# ASL Interpreter

import torch

class GoogLeNetCNN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    def inceptionModule(self, input:torch.Tensor) -> torch.Tensor:
        pass

    def forward(self, input:torch.Tensor) -> torch.Tensor:
        pass


if __name__ == "__main__":
    model = GoogLeNetCNN()

    test = torch.rand((64,224,224), dtype=torch.float32)
    
    model(test)

    