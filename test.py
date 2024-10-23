import torch
# GoogLeNet CNN Class File
# Aidan Carter
# ASL Interpreter

import torch

class I3D(torch.nn.Module):
    def __init__(self, subset:int) -> None:
        super().__init__()

        self.activation = torch.nn.ReLU()
        
        self.conv0 = torch.nn.Conv3d(3, 64, kernel_size=7, stride=2, padding=3)

        self.maxpool0 = torch.nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))

        self.lnorm0 = torch.nn.LocalResponseNorm(64)

        self.conv1a = torch.nn.Conv3d(64, 64, kernel_size=1)
        self.conv1b = torch.nn.Conv3d(64, 192, kernel_size=3, padding=1)

        self.lnorm1 = torch.nn.LocalResponseNorm(192)

        self.maxpool1 = torch.nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.maxpool2 = torch.nn.MaxPool3d(kernel_size=2, stride=2)

        self.avgpool = torch.nn.AvgPool3d(kernel_size=7)

        self.dropout = torch.nn.Dropout3d(0.4)

        self.linear1 = torch.nn.Linear(16386, 1024)

        self.linear2 = torch.nn.Linear(1024, subset)

        self.softmax = torch.nn.Softmax(dim=1)

        self.IncA = torch.nn.ModuleList([
            torch.nn.Conv3d(192, 64, kernel_size=1),
            torch.nn.BatchNorm3d(64),

            torch.nn.Conv3d(192, 96, kernel_size=1),
            torch.nn.Conv3d(96, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(128),

            torch.nn.Conv3d(192, 16, kernel_size=1),
            torch.nn.Conv3d(16, 32, kernel_size=5, padding=2),
            torch.nn.BatchNorm3d(32),

            torch.nn.MaxPool3d(kernel_size=3, padding=(0,1,1)),
            torch.nn.Conv3d(192, 32, kernel_size=1, padding=(11,9,9)),
            torch.nn.BatchNorm3d(32),

        ])
        

    def Inception(self, input:torch.Tensor, func:torch.nn.ModuleList) -> torch.Tensor:
        convInc0 = self.activation(func[0](input))
        bnorm0 = func[1](convInc0)
        convInc1a = self.activation(func[2](input))
        convInc1b = self.activation(func[3](convInc1a))
        bnorm1 = func[4](convInc1b)
        convInc2a = self.activation(func[5](input))
        convInc2b = self.activation(func[6](convInc2a))
        bnorm2 = func[7](convInc2b)
        poolInc = func[8](input)
        convIncPool = self.activation(func[9](poolInc))
        bnormPool = func[10](convIncPool)
        return torch.cat((bnorm0, bnorm1, bnorm2, bnormPool), 1)

    def forward(self, input:torch.Tensor) -> torch.Tensor:
        conv0 = self.activation(self.conv0(input))
        pool0 = self.maxpool0(conv0)
        conv1a = self.conv1a(pool0)
        conv1b = self.activation(self.conv1b(conv1a))
        pool1 = self.maxpool0(conv1b)
        inc0 = self.Inception(pool1, self.IncA)
        pool4 = self.avgpool(inc0)
        drop = self.dropout(pool4)
        linear1 = self.activation(self.linear1(drop.flatten(1)))
        linear2 = self.activation(self.linear2(linear1))
        softmax = self.softmax(linear2)
        return softmax


if __name__ == "__main__":
    model = I3D()

    