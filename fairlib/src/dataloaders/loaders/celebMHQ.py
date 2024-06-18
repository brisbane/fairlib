import numpy as np
from ..utils import BaseDataset
from pathlib import Path
import pandas as pd
import os
import torch
from torchvision import transforms
from PIL import Image
import random

class celebMHQDataset(BaseDataset):
    #pretransformed
    if 0:
      transform=transforms.Compose([
      transforms.Resize( int(256) ),
      transforms.CenterCrop( int(224) ),
      transforms.ToTensor(),
      #note these numbers differ from soruce paper
      #https://pytorch.org/hub/pytorch_vision_resnet/ - torch numbers
      # paper bffhq numbers https://github.com/kakaoenterprise/Learning-Debiased-Disentangled/blob/master/data/util.py
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_data(self):
        if self.args.seed !=1:
            self.data_file = os.path.join(self.args.data_dir,self.args.seed, "celebMHQ_Blond_HairMale_{}_pre.pt".format(self.split))
        else:
            self.data_file = os.path.join(self.args.data_dir, "celebMHQ_Blond_HairMale_{}_pre.pt".format(self.split))


        print("loading:" , self.data_file)
        data = torch.load(self.data_file)
        print(type(data['img_features'][0]), len(data['img_features'][0]))
        self.X = data['img_features']# [self.transform(_img) for _img in data[0]]
        print(self.X[0].shape)
        print(len(self.X))
        self.y = data['target']
        self.protected_label = data['protected']
