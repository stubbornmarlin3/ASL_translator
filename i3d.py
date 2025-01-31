# I3D CNN Class File
# Aidan Carter
# ASL Interpreter

import torch

class Inception(torch.nn.Module):
    def __init__(self, in_ch:int, out_ch_1x1:int, out_ch_3x3_reduce:int, out_ch_3x3:int, out_ch_5x5_reduce:int, out_ch_5x5:int):
        super().__init__()

        self.relu = torch.nn.LeakyReLU()

        self.conv_1x1 = torch.nn.Conv3d(in_ch, out_ch_1x1, kernel_size=1)
        
        self.conv_3x3_reduce = torch.nn.Conv3d(in_ch, out_ch_3x3_reduce, kernel_size=1)

        self.conv_3x3 = torch.nn.Conv3d(out_ch_3x3_reduce, out_ch_3x3, kernel_size=3, padding="same")

        self.conv_5x5_reduce = torch.nn.Conv3d(in_ch, out_ch_5x5_reduce, kernel_size=1)

        self.conv_5x5 = torch.nn.Conv3d(out_ch_5x5_reduce, out_ch_5x5, kernel_size=5, padding="same")


    def forward(self, input: torch.Tensor) -> torch.Tensor:

        conv_1x1 = self.relu(self.conv_1x1(input))

        conv_3x3_reduce = self.relu(self.conv_3x3_reduce(input))
        conv_3x3 = self.relu(self.conv_3x3(conv_3x3_reduce))

        conv_5x5_reduce = self.relu(self.conv_5x5_reduce(input))
        conv_5x5 = self.relu(self.conv_5x5(conv_5x5_reduce))

        return torch.cat((conv_1x1, conv_3x3, conv_5x5), dim=1)


class I3D(torch.nn.Module):

    def __init__(self, classes:int):
        super().__init__()

        self.relu = torch.nn.ReLU()

        self.conv0 = torch.nn.Conv3d(3, 64, kernel_size=7, stride=2)

        self.pool0 = torch.nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), ceil_mode=True)

        self.conv1 = torch.nn.Conv3d(64, 192, kernel_size=3, stride=1)

        self.pool1 = torch.nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), ceil_mode=True)

        self.inc0 = Inception(192, 64, 96, 128, 16, 32)

        self.inc1 = Inception(224, 128, 128, 192, 32, 96)

        self.pool2 = torch.nn.MaxPool3d(kernel_size=3, stride=2, ceil_mode=True)

        self.inc2 = Inception(416, 192, 96, 208, 16, 48)

        self.inc3 = Inception(448, 160, 112, 224, 24, 64)

        self.inc4 = Inception(448, 128, 128, 256, 24, 64)

        self.inc5 = Inception(448, 112, 144, 288, 32, 64)

        self.inc6 = Inception(464, 256, 160, 320, 32, 128)

        self.pool3 = torch.nn.MaxPool3d(kernel_size=3, stride=2, ceil_mode=True, padding=1)

        self.inc7 = Inception(704, 256, 160, 320, 32, 128)

        self.inc8 = Inception(704, 384, 192, 384, 48, 128)

        self.pool4 = torch.nn.AvgPool3d(kernel_size=7, stride=1)

        self.linear = torch.nn.Linear(896, classes)

    def forward(self, input:torch.Tensor) -> torch.Tensor:

        conv0 = self.relu(self.conv0(input))
        pool0 = self.pool0(conv0)
        conv1 = self.relu(self.conv1(pool0))
        pool1 = self.pool1(conv1)
        inc0 = self.inc0(pool1)
        inc1 = self.inc1(inc0)
        pool2 = self.pool2(inc1)
        inc2 = self.inc2(pool2)
        inc3 = self.inc3(inc2)
        inc4 = self.inc4(inc3)
        inc5 = self.inc5(inc4)
        inc6 = self.inc6(inc5)
        pool3 = self.pool3(inc6)
        inc7 = self.inc7(pool3)
        inc8 = self.inc8(inc7)
        pool4 = self.pool4(inc8)
        pool4_flattened = torch.flatten(pool4, start_dim=1)
        linear = self.linear(pool4_flattened)
        return(linear)

if __name__ == "__main__":
    model = I3D(1000).to(torch.device("mps"))
    test = torch.rand((2,3,64,224,224), dtype=torch.float32).to(torch.device("mps"))
    print(model(test))
