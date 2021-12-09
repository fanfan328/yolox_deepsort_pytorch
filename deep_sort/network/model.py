import torch
import torch.nn as nn
import torch.nn.functional as F

# 3x3 convolution
def conv(in_channels, out_channels, stride=1, kernel=3, padding=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel, 
                     stride=stride, padding=padding, bias=False)

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False, stride=1, ):
        super(ResidualBlock, self).__init__()
        self.flag_downsample=downsample
        
        self.conv1 = conv(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if downsample:
            self.downsample = nn.Sequential(
                conv(in_channels, out_channels, stride=2, kernel=1, padding=0),
                nn.BatchNorm2d(out_channels)
            )
        elif in_channels != out_channels:
            self.downsample = nn.Sequential(
                conv(in_channels, out_channels, stride=1, kernel=1, padding=0),
                nn.BatchNorm2d(out_channels)
            )
            self.flag_downsample = True
            
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.flag_downsample:
            residual = self.downsample(x)
        return F.relu(residual.add(out),True)

class CNN(nn.Module):#32673
    def __init__(self, num_classes=751, reid=False): #Person only
        super(CNN, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)

        self.conv = nn.Sequential(
            # Original Version
            conv(3,32,stride=1),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            conv(32,32,stride=1),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.MaxPool2d(3,2,padding=1),

            #Shorter Version (32 change into 64)
            # nn.Conv2d(3,32,3,stride=1,padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(3,2,padding=1),
        )

        # Based on https://arxiv.org/pdf/1703.07402.pdf
        # Residual 4 In 32 64 32
        self.res4 = nn.Sequential(ResidualBlock(32, 32, False))

        # Residual 5 In 32 64 32
        self.res5 = nn.Sequential(ResidualBlock(32, 32, False))

        # Residual 6 In 64 32 16
        self.res6 = nn.Sequential(ResidualBlock(32, 64,  True, stride=2))

        # Residual 7 In 64 32 16
        self.res7 = nn.Sequential(ResidualBlock(64, 64, False))

        # Residual 8 In 128 16 32
        self.res8 = nn.Sequential(ResidualBlock(64, 128, True, stride=2))

        # Residual 9 In 32 64 32
        self.res9 = nn.Sequential(ResidualBlock(128, 128, False))

        self.res4_5 = nn.Sequential(
            ResidualBlock(32, 32, False),
            ResidualBlock(32, 32, False))
        
        self.res6_7 = nn.Sequential(
            ResidualBlock(32, 64, True, stride=2),
            ResidualBlock(64, 64, False))
        
        self.res8_9 = nn.Sequential(
            ResidualBlock(64,128, True, stride=2),
            ResidualBlock(128, 128, False))
        # Dense
        self.dense = nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Linear(in_features=128*16*8, out_features=128),
            nn.BatchNorm1d(128),
            nn.ELU(inplace=True)
        ) 
        self.reid = reid
        # Batch + norm
        self.bn = nn.BatchNorm1d(128)
        self.out = nn.Sequential(
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        out = self.conv(x.cuda())
        out = self.res4(out)
        out = self.res5(out) 
        out = self.res6(out)
        out = self.res7(out)
        out = self.res8(out)
        out = self.res9(out)

        # From shape [4, 128, 16, 8] Into 4, 128*16*8
        # print(f" Before dense {out.shape}")
        out = out.view(out.size(0),-1)
        if self.reid:
            out = self.dense[0](out)
            out = self.dense[1](out)
            out = out.div(out.norm(p=2,dim=1,keepdim=True))
            # print(f" Reid dense {out.shape}")
            return out

        out = self.dense(out)
        out = self.out(out)
        return out
        

if __name__ == '__main__':
    network = CNN(reid=True)
    # Random 4 frame image, 3 channel, 128*64
    x = torch.randn(4,3,128,64)
    y = network(x)   

    # print(y)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print('Using {} device'.format(device))
    model = CNN().to(device)
    # print(model) 