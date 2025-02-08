from torch import nn
import torch
from torch.nn.functional import softmax
from torchvision.transforms.functional import crop
import logging

logger = logging.getLogger("unet")

class Unet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv11 = nn.Conv2d(1,64,3)
        self.conv12 = nn.Conv2d(64,64,3)
        self.maxPool = nn.MaxPool2d(2,stride = 2)
        self.conv21 = nn.Conv2d(64,128,3)
        self.conv22 = nn.Conv2d(128,128,3)
        self.conv31 = nn.Conv2d(128,256,3)
        self.conv32 = nn.Conv2d(256,256,3)
        self.conv41 = nn.Conv2d(256,512,3)
        self.conv42 = nn.Conv2d(512,512,3)
        self.conv51 = nn.Conv2d(512,1024,3)
        self.conv52 = nn.Conv2d(1024,1024,3)
        self.upConv1 = nn.ConvTranspose2d(1024, 512, 2, stride = 2)
        self.convU11 = nn.Conv2d(1024,512,3)
        self.convU12 = nn.Conv2d(512,512,3)
        self.upConv2 = nn.ConvTranspose2d(512, 256, 2, stride = 2)
        self.convU21 = nn.Conv2d(512,256,3)
        self.convU22 = nn.Conv2d(256,256,3)
        self.upConv3 = nn.ConvTranspose2d(256, 128, 2, stride = 2)
        self.convU31 = nn.Conv2d(256, 128,3)
        self.convU32 = nn.Conv2d(128,128,3)
        self.upConv4 = nn.ConvTranspose2d(128, 64, 2, stride = 2)
        self.convU41 = nn.Conv2d(128, 64,3)
        self.convU42 = nn.Conv2d(64,64,3)
        self.convLast = nn.Conv2d(64,2,1)
        self.activation = nn.LeakyReLU()

    def unetCrop(self, xdown, xup) :
        upsize = xup.size()[-1]
        downsize = xdown.size()[-1]
        top = int((downsize - upsize)/2)
        return crop(xdown, top, top, upsize, upsize) 

    def forward(self, x):
        logger.debug(f"x: {x.std()}, {x.mean()}")
        x1 = self.activation(self.conv12(self.activation(self.conv11(x))))
        logger.debug(f"layer0: {x1.std()}, {x1.mean()}")
        x = self.maxPool(x1)
        x2 = self.activation(self.conv22(self.activation(self.conv21(x))))
        logger.debug(f"layer1: {x2.std()}, {x2.mean()}")
        x = self.maxPool(x2)
        x3 = self.activation(self.conv32(self.activation(self.conv31(x))))
        logger.debug(f"layer2: {x3.std()}, {x3.mean()}")
        x = self.maxPool(x3)
        x4 = self.activation(self.conv42(self.activation(self.conv41(x))))
        logger.debug(f"layer3: {x4.std()}, {x4.mean()}")
        x = self.maxPool(x4)
        x = self.activation(self.conv52(self.activation(self.conv51(x))))

        xup = self.upConv1(x)
        x = torch.cat([self.unetCrop(x4, xup), xup], dim = 1)
        logger.debug(f"layer5: {x.std()}, {x.mean()}")
        x = self.activation(self.convU11(x))
        logger.debug(f"layer6: {x.std()}, {x.mean()}")
        x = self.activation(self.convU12(x))
        logger.debug(f"layer7: {x.std()}, {x.mean()}")

        xup = self.upConv2(x)
        x = torch.cat([self.unetCrop(x3, xup), xup], dim = 1)
        logger.debug(f"layer8: {x.std()}, {x.mean()}")
        x = self.activation(self.convU21(x))
        x = self.activation(self.convU22(x))
        logger.debug(f"layer9: {x.std()}, {x.mean()}")

        xup = self.upConv3(x)
        x = torch.cat([self.unetCrop(x2, xup), xup], dim = 1)
        x = self.activation(self.convU31(x))
        x = self.activation(self.convU32(x))
        logger.debug(f"layer10: {x.std()}, {x.mean()}")

        xup = self.upConv4(x)
        x = torch.cat([self.unetCrop(x1, xup), xup], dim = 1)
        x = self.activation(self.convU41(x))
        x = self.activation(self.convU42(x))
        logger.debug(f"layer11: {x.std()}, {x.mean()}")

        x = self.convLast(x)
        logger.debug(f"layer8final: {x.std()}, {x.mean()}")
        x = softmax(x)
        return x
    

