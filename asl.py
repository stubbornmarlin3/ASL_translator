import torch

class ASL(torch.nn.Module):
    def __init__(self, classes:int=10):
        super(ASL, self).__init__()
        self.conv1 = torch.nn.Conv3d(3, 32, kernel_size=(3,7,7), stride=(1,2,2), padding=(1,3,3))
        self.bn1 = torch.nn.BatchNorm3d(32)
        self.pool1 = torch.nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
        self.conv2 = torch.nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm3d(64)
        self.pool2 = torch.nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv3 = torch.nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = torch.nn.BatchNorm3d(128)
        self.conv4 = torch.nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = torch.nn.BatchNorm3d(256)
        self.pool3 = torch.nn.AdaptiveAvgPool3d(1)
        self.drop = torch.nn.Dropout(p=0.4, inplace=True)
        self.fc = torch.nn.Linear(256, classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.nn.functional.relu(x, inplace=True)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.nn.functional.relu(x, inplace=True)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.nn.functional.relu(x, inplace=True)
        x = self.conv4(x)
        x = self.bn4(x)
        x = torch.nn.functional.relu(x, inplace=True)
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = self.drop(x)
        x = self.fc(x)
        return x
    
if __name__ == "__main__":

    import time

    x = torch.randn((1,3,64,224,224))

    model = ASL()
    model.eval()
    
    start_time = time.time()
    with torch.no_grad():
        y = model(x)
    end_time = time.time()
    print("Seconds: ", end_time-start_time)