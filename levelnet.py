from torch import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import CrossEntropyLoss
import config
import cv2
from torchvision.transforms import ToTensor
from torch.autograd import Variable
from torchsummary import summary

cfg = config.get_cfg_defaults()


class reduce(nn.Module):
    def __init__(self, lvl):
        super(reduce, self).__init__()
        self.reduce_plans = nn.Conv2d(cfg[lvl].out, cfg[lvl].reduce_to, (1, 1), stride=(1, 1))

    def forward(self, x):
        return self.reduce_plans(x)


class layer_ds(nn.Module):
    def __init__(self, lvl):
        super(layer_ds, self).__init__()
        self.conv7 = nn.Conv2d(cfg[lvl].inplans, cfg[lvl].conv7plans, (7, 7), stride=(2, 2), padding=(3, 3))
        self.conv5 = nn.Conv2d(cfg[lvl].inplans, cfg[lvl].conv5plans, (5, 5), stride=(2, 2), padding=(2, 2))
        self.conv3 = nn.Conv2d(cfg[lvl].inplans, cfg[lvl].conv3plans, (3, 3), stride=(2, 2), padding=(1, 1))
        self.pool = nn.AvgPool2d((2, 2), stride=(2, 2))
        self.relu = nn.ReLU()
        self.batchNorm = nn.BatchNorm2d(cfg[lvl].out)
        self.reduce = reduce(lvl)

    def forward(self, x):
        conv7 = self.conv7(x)
        conv5 = self.conv5(x)
        conv3 = self.conv3(x)
        pool = self.pool(x)
        out_plans = torch.cat([conv7, conv5, conv3, pool], 1)
        out_plans = self.reduce.forward(self.batchNorm(self.relu(out_plans)))
        return out_plans



class upsample(nn.Module):
    def __init__(self, lvl):
        super(upsample, self).__init__()
        self.conv7t = nn.ConvTranspose2d(cfg[lvl].inplans, cfg[lvl].conv7plans, (7, 7), stride=(2, 2), padding=(3, 3), output_padding= (1, 1))
        self.conv5t = nn.ConvTranspose2d(cfg[lvl].inplans, cfg[lvl].conv5plans, (5, 5), stride=(2, 2), padding=(2, 2), output_padding= (1, 1))
        self.conv3t = nn.ConvTranspose2d(cfg[lvl].inplans, cfg[lvl].conv3plans, (3, 3), stride=(2, 2), padding=(1, 1), output_padding= (1, 1))
        self.relu = nn.ReLU()
        self.batchNorm = nn.BatchNorm2d(cfg[lvl].out)
        self.reduce = reduce(lvl)


    def forward(self, x):
        conv7t = self.conv7t(x)
        conv5t = self.conv5t(x)
        conv3t = self.conv3t(x)
        out_plans = torch.cat([conv7t, conv5t, conv3t], 1)
        out_plans = self.reduce.forward(self.batchNorm(self.relu(out_plans)))
        return out_plans




class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.conv = nn.Conv2d(3, 18, (3, 3), stride=(1, 1), padding=(1, 1))
        self.relu = nn.ReLU()
        self.batchNorm = nn.BatchNorm2d(18)
        self.layer_64 = layer_ds('lvl_64')
        self.layer_32 = layer_ds('lvl_32')
        self.layer_16 = layer_ds('lvl_16')
        self.layer_8 = layer_ds('lvl_8')
        self.layer_16up = upsample('lvl_16up')
        self.layer_32up = upsample('lvl_32up')
        self.layer_64up = upsample('lvl_64up')
        self.layer_128up =  nn.ConvTranspose2d(364, 364, (3, 3), stride=(2, 2), padding=(1, 1), output_padding= (1, 1))
        self.pred_out = nn.Conv2d(364, cfg.out_features,(3,3),stride=(1, 1),padding=(1, 1))

    def forward(self, x):
        x = self.batchNorm(self.relu(self.conv(x)))
        x64 = self.layer_64.forward(x)
        x32 = self.layer_32.forward(x64)
        x16 = self.layer_16.forward(x32)
        x8 = self.layer_8.forward(x16)
        x16up = torch.cat([self.layer_16up.forward(x8), x16], 1)
        x32up = torch.cat([self.layer_32up.forward(x16up), x32], 1)
        x64up = torch.cat([self.layer_64up.forward(x32up), x64], 1)
        x128up = self.layer_128up(x64up)
        pred_out = self.pred_out(x128up)


        return pred_out


#if __name__ == '__main__':
#    img = cv2.imread("C:/Users/Abhi/Desktop/facial_landmarks/helen_dataset/helen-master/data/img/10406776_1.jpg")
#    img = cv2.resize(img, (128, 128), cv2.INTER_LINEAR)
#    ##print(img)
 #   img = ToTensor()(img).unsqueeze(0)  # unsqueeze to add artificial first dimension
  #  ima = Variable(img)
   # model = model()
    #print(img.shape)
    #model.forward(img)
    #summary(model, (3, 128, 128), 32)

