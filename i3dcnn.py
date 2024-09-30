# I3D CNN Class File
# Aidan Carter
# ASL Interpreter

import torch

class I3DCNN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv_0 = torch.nn.Conv3d(3, 3, 7, 2)
        self.max_pool_0 = torch.nn.MaxPool3d((1,3,3), (1,2,2))
        self.conv_1 = torch.nn.Conv3d(3, 3, 1)
        self.conv_2 = torch.nn.Conv3d(3, 3, 3)

        self.conv_3 = torch.nn.Conv3d(3, 3, 3, padding=1)
        
        self.max_pool_1 = torch.nn.MaxPool3d(3, 2)

        self.max_pool_2 = torch.nn.MaxPool3d(2, 2)

        self.avg_pool_0 = torch.nn.AvgPool3d((2,7,7))


    def inceptionModule(self, input:torch.Tensor) -> torch.Tensor:
        conv0 = self.conv_1(input)
        print(conv0.shape)
        
        conv1 = self.conv_3(conv0)
        print(conv1.shape)

        return torch.cat((conv0, conv1), dim=1)

    def forward(self, input:torch.Tensor) -> torch.Tensor:
        print(input.shape)
        conv0 = self.conv_0(input)
        print(conv0.shape)
        pool0 = self.max_pool_0(conv0)
        print(pool0.shape)
        conv1 = self.conv_1(pool0)
        print(conv1.shape)
        conv2 = self.conv_2(conv1)
        print(conv2.shape)
        pool1 = self.max_pool_0(conv2)
        print(pool1.shape)
        inc0 = self.inceptionModule(pool1)
        print(inc0.shape)
        inc1 = self.inceptionModule(inc0)
        print(inc1.shape)
        pool2 = self.max_pool_1(inc1)
        print(pool2.shape)
        inc2 = self.inceptionModule(pool2)
        print(inc2.shape)
        inc3 = self.inceptionModule(inc2)
        print(inc3.shape)
        inc4 = self.inceptionModule(inc3)
        print(inc4.shape)
        inc5 = self.inceptionModule(inc4)
        print(inc5.shape)
        inc6 = self.inceptionModule(inc5)
        print(inc6.shape)
        pool3 = self.max_pool_2(inc6)
        print(pool3.shape)
        inc7 = self.inceptionModule(pool3)
        print(inc7.shape)
        inc8 = self.inceptionModule(inc7)
        print(inc8.shape)
        pool4 = self.avg_pool_0(inc8)
        print(pool4.shape)
        conv3 = self.conv_1(pool4)
        print(conv3.shape)
        


if __name__ == "__main__":
    model = I3DCNN()

    test = torch.rand((3,64,224,224), dtype=torch.float32)
    
    model(test)

    