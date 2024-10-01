# GoogLeNet CNN Class File
# Aidan Carter
# ASL Interpreter

import torch

class GoogLeNetCNN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.relu = torch.nn.ReLU()

        self.conv0 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool0 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv1a = torch.nn.Conv2d(64, 64, kernel_size=1)
        self.conv1b = torch.nn.Conv2d(64, 192, kernel_size=3, padding=1)

        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.IncA = torch.nn.ModuleList([
            torch.nn.Conv2d(192, 64, kernel_size=1),

            torch.nn.Conv2d(192, 96, kernel_size=1),
            torch.nn.Conv2d(96, 128, kernel_size=3, padding=1),

            torch.nn.Conv2d(192, 16, kernel_size=1),
            torch.nn.Conv2d(16, 32, kernel_size=5, padding=2),

            torch.nn.MaxPool2d(kernel_size=3, padding=1),
            torch.nn.Conv2d(192, 32, kernel_size=1, padding=9),
        ])
        
        self.IncB = torch.nn.ModuleList([
            torch.nn.Conv2d(256, 128, kernel_size=1),

            torch.nn.Conv2d(256, 128, kernel_size=1),
            torch.nn.Conv2d(128, 192, kernel_size=3, padding=1),

            torch.nn.Conv2d(256, 32, kernel_size=1),
            torch.nn.Conv2d(32, 96, kernel_size=5, padding=2),

            torch.nn.MaxPool2d(kernel_size=3, padding=1),
            torch.nn.Conv2d(256, 64, kernel_size=1, padding=9),
        ])

        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


    def Inception(self, input:torch.Tensor, func:list) -> torch.Tensor:

        convInc0 = self.relu(func[0](input))
        print(convInc0.shape)

        convInc1a = func[1](input)
        convInc1b = self.relu(func[2](convInc1a))
        print(convInc1b.shape)

        convInc2a = func[3](input)
        convInc2b = self.relu(func[4](convInc2a))
        print(convInc2b.shape)

        poolInc = func[5](input)
        convIncPool = self.relu(func[6](poolInc))
        print(convIncPool.shape)

        return torch.cat((convInc0, convInc1b, convInc2b, convIncPool), 1)


    def forward(self, input:torch.Tensor) -> torch.Tensor:
        print(input.shape)

        conv0 = self.relu(self.conv0(input))
        print(conv0.shape)

        pool0 = self.maxpool0(conv0)
        print(pool0.shape)

        conv1a = self.conv1a(pool0)
        conv1b = self.relu(self.conv1b(conv1a))
        print(conv1b.shape)

        pool1 = self.maxpool1(conv1b)
        print(pool1.shape)
    
        inc0 = self.Inception(pool1, self.IncA)
        print(inc0.shape)

        inc1 = self.Inception(inc0, self.IncB)
        print(inc1.shape)

        pool2 = self.maxpool2(inc1)
        print(pool2.shape)


if __name__ == "__main__":
    model = GoogLeNetCNN().to(torch.device("mps"))

    test = torch.rand((20,3,224,224), dtype=torch.float32).to(torch.device("mps"))
    
    model(test)

    