# GoogLeNet CNN Class File
# Aidan Carter
# ASL Interpreter

import torch

class I3D(torch.nn.Module):
    def __init__(self) -> None:
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

        self.fc1 = torch.nn.Linear(1024, 1000)

        self.softmax = torch.nn.Softmax(dim=0)

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
        
        self.IncB = torch.nn.ModuleList([
            torch.nn.Conv3d(256, 128, kernel_size=1),
            torch.nn.BatchNorm3d(128),

            torch.nn.Conv3d(256, 128, kernel_size=1),
            torch.nn.Conv3d(128, 192, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(192),

            torch.nn.Conv3d(256, 32, kernel_size=1),
            torch.nn.Conv3d(32, 96, kernel_size=5, padding=2),
            torch.nn.BatchNorm3d(96),

            torch.nn.MaxPool3d(kernel_size=3, padding=(0,1,1)),
            torch.nn.Conv3d(256, 64, kernel_size=1, padding=(11,9,9)),
            torch.nn.BatchNorm3d(64),
        ])

        self.IncC = torch.nn.ModuleList([
            torch.nn.Conv3d(480, 192, kernel_size=1),
            torch.nn.BatchNorm3d(192),

            torch.nn.Conv3d(480, 96, kernel_size=1),
            torch.nn.Conv3d(96, 208, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(208),

            torch.nn.Conv3d(480, 16, kernel_size=1),
            torch.nn.Conv3d(16, 48, kernel_size=5, padding=2),
            torch.nn.BatchNorm3d(48),

            torch.nn.MaxPool3d(kernel_size=3, padding=(1,0,0)),
            torch.nn.Conv3d(480, 64, kernel_size=1, padding=5),
            torch.nn.BatchNorm3d(64),
        ])

        self.IncD = torch.nn.ModuleList([
            torch.nn.Conv3d(512, 160, kernel_size=1),
            torch.nn.BatchNorm3d(160),

            torch.nn.Conv3d(512, 112, kernel_size=1),
            torch.nn.Conv3d(112, 224, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(224),

            torch.nn.Conv3d(512, 24, kernel_size=1),
            torch.nn.Conv3d(24, 64, kernel_size=5, padding=2),
            torch.nn.BatchNorm3d(64),

            torch.nn.MaxPool3d(kernel_size=3, padding=(1,0,0)),
            torch.nn.Conv3d(512, 64, kernel_size=1, padding=5),
            torch.nn.BatchNorm3d(64),
        ])

        self.IncE = torch.nn.ModuleList([
            torch.nn.Conv3d(512, 128, kernel_size=1),
            torch.nn.BatchNorm3d(128),

            torch.nn.Conv3d(512, 128, kernel_size=1),
            torch.nn.Conv3d(128, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(256),

            torch.nn.Conv3d(512, 24, kernel_size=1),
            torch.nn.Conv3d(24, 64, kernel_size=5, padding=2),
            torch.nn.BatchNorm3d(64),

            torch.nn.MaxPool3d(kernel_size=3, padding=(1,0,0)),
            torch.nn.Conv3d(512, 64, kernel_size=1, padding=5),
            torch.nn.BatchNorm3d(64),
        ])

        self.IncF = torch.nn.ModuleList([
            torch.nn.Conv3d(512, 112, kernel_size=1),
            torch.nn.BatchNorm3d(112),

            torch.nn.Conv3d(512, 144, kernel_size=1),
            torch.nn.Conv3d(144, 288, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(288),

            torch.nn.Conv3d(512, 32, kernel_size=1),
            torch.nn.Conv3d(32, 64, kernel_size=5, padding=2),
            torch.nn.BatchNorm3d(64),

            torch.nn.MaxPool3d(kernel_size=3, padding=(1,0,0)),
            torch.nn.Conv3d(512, 64, kernel_size=1, padding=5),
            torch.nn.BatchNorm3d(64),
        ])

        self.IncG = torch.nn.ModuleList([
            torch.nn.Conv3d(528, 256, kernel_size=1),
            torch.nn.BatchNorm3d(256),

            torch.nn.Conv3d(528, 160, kernel_size=1),
            torch.nn.Conv3d(160, 320, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(320),

            torch.nn.Conv3d(528, 32, kernel_size=1),
            torch.nn.Conv3d(32, 128, kernel_size=5, padding=2),
            torch.nn.BatchNorm3d(128),

            torch.nn.MaxPool3d(kernel_size=3, padding=(1,0,0)),
            torch.nn.Conv3d(528, 128, kernel_size=1, padding=5),
            torch.nn.BatchNorm3d(128),
        ])

        self.IncH = torch.nn.ModuleList([
            torch.nn.Conv3d(832, 256, kernel_size=1),
            torch.nn.BatchNorm3d(256),

            torch.nn.Conv3d(832, 160, kernel_size=1),
            torch.nn.Conv3d(160, 320, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(320),

            torch.nn.Conv3d(832, 32, kernel_size=1),
            torch.nn.Conv3d(32, 128, kernel_size=5, padding=2),
            torch.nn.BatchNorm3d(128),

            torch.nn.MaxPool3d(kernel_size=3, padding=(0,1,1)),
            torch.nn.Conv3d(832, 128, kernel_size=1, padding=(3,2,2)),
            torch.nn.BatchNorm3d(128),
        ])

        self.IncI = torch.nn.ModuleList([
            torch.nn.Conv3d(832, 384, kernel_size=1),
            torch.nn.BatchNorm3d(384),

            torch.nn.Conv3d(832, 192, kernel_size=1),
            torch.nn.Conv3d(192, 384, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(384),

            torch.nn.Conv3d(832, 48, kernel_size=1),
            torch.nn.Conv3d(48, 128, kernel_size=5, padding=2),
            torch.nn.BatchNorm3d(128),

            torch.nn.MaxPool3d(kernel_size=3, padding=(0,1,1)),
            torch.nn.Conv3d(832, 128, kernel_size=1, padding=(3,2,2)),
            torch.nn.BatchNorm3d(128),
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
        inc1 = self.Inception(inc0, self.IncB)
        pool2 = self.maxpool1(inc1)
        inc2 = self.Inception(pool2, self.IncC)
        inc3 = self.Inception(inc2, self.IncD)
        inc4 = self.Inception(inc3, self.IncE)
        inc5 = self.Inception(inc4, self.IncF)
        inc6 = self.Inception(inc5, self.IncG)
        pool3 = self.maxpool2(inc6)
        inc7 = self.Inception(pool3, self.IncH)
        inc8 = self.Inception(inc7, self.IncI)
        pool4 = self.avgpool(inc8)
        linear = self.linear(pool4.flatten(1))
        softmax = self.softmax(linear)
        return softmax


if __name__ == "__main__":
    model = I3D()

    test = torch.rand((2,3,64,224,224), dtype=torch.float32)
    
    model(test)

    