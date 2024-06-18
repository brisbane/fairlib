import numpy as np
from ..utils import BaseDataset
from pathlib import Path
import pandas as pd
import os
import torch
from torchvision import transforms
from PIL import Image
import random

class bffhqDataset(BaseDataset):

    def load_data(self):
        if self.args.seed != 1:
            self.data_file = os.path.join(self.args.data_dir,self.args.seed, "bffhq_{}.pt".format(self.split))
        else:
            self.data_file = os.path.join(self.args.data_dir, "bffhq_{}.pt".format(self.split))



        data = torch.load(self.data_file).set_index('index')
        print (data.head())
        print(type(data['img_features'].iloc[0]), len(data['img_features'].iloc[0]))

        self.X = data['img_features'] 
        #[self.transform(_img) for _img in data[0]]
        print(self.X[0].shape)
        print(len(self.X))
        self.y = data['target']
        self.protected_label = data['protected']
