import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import math
import os
import re
import torch
import random

def makefiles(filenames):
    for filename in filenames:
        if not os.path.exists(filename):
            os.makedirs(filename)

# facades Dataset
class FacadesDataset(Dataset):
    def __init__(self, allfilenames, guiyihua = 274,root = 'E:\\读博项目\\基准工况\\NN\\dataset\\',imagesize=256,device = torch.device("cuda:0"),type = 0):
        self.Twb1,self.Twb2,self.Tair,self.shidu,self.Vair,self.R1,self.R2,self.R3 = np.zeros(len(allfilenames)),np.zeros(len(allfilenames)),\
        np.zeros(len(allfilenames)),np.zeros(len(allfilenames)),np.zeros(len(allfilenames)),np.zeros(len(allfilenames)),np.zeros(len(allfilenames)),np.zeros(len(allfilenames))
        self.file_names = []
        D,m = guiyihua[1],guiyihua[0]
        self.type = type
        for i,canshu in enumerate(allfilenames):
            data = canshu.split()
            for ii in range(8):
                data[ii] = float(data[ii])
            self.Twb1[i], self.Twb2[i], self.Tair[i], self.shidu[i], self.Vair[i], self.R1[i], self.R2[i], self.R3[i] = 0+(data[0]-m)/D,0+(data[1]-m)/D,\
            data[2]/guiyihua[2],data[3]/guiyihua[3],data[4]/guiyihua[4],data[5]/guiyihua[5],data[6]/guiyihua[6],data[7]/guiyihua[7]
            temp = os.path.join(root,data[8])
            self.file_names.append(temp)
        self.Tmin = guiyihua[0]
        self.Tmax = guiyihua[0] + guiyihua[1]

    def __getitem__(self, index):
        img = self.readimg(self.file_names[index]) / 127.5 - 1

        inputdata = np.array(
            [self.Tair[index], self.shidu[index], self.Vair[index], self.R1[index], self.R2[index], self.R3[index],
             (float(self.file_names[index][-7:-5]) - 10) / 10])
        wenbiao = np.array([self.Twb1[index], self.Twb2[index]])
        name = self.file_names[index][-15:]
        index2 = random.randint(0, len(self.file_names)-1)
        if not self.type == 0:
            index2 = 11 * 3 - 1 + 3
        img2 = self.readimg(self.file_names[index2]) / 127.5 - 1

        inputdata2 = np.array(
            [self.Tair[index2], self.shidu[index2], self.Vair[index2], self.R1[index2], self.R2[index2],
             self.R3[index2],
             (float(self.file_names[index2][-7:-5]) - 10) / 10])
        wenbiao2 = np.array([self.Twb1[index2], self.Twb2[index2]])

        return [inputdata2, img2, wenbiao2], [inputdata, img, wenbiao], name


    def __len__(self):
        return len(self.file_names)

    def readimg(self,path,imgsize = 256):
        C, H, W = 3,imgsize, imgsize
        img = Image.open(path)
        # img = img.resize((2 * W, H), Image.ANTIALIAS)
        img = img.resize((W, H), Image.Resampling.LANCZOS)
        # plt.imshow(img)
        # plt.show()
        img = np.array(img)
        return img

if __name__=='__main__':
    pass