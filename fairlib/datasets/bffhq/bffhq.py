from fairlib.src.utils import seed_everything
import numpy as np
import os
from fairlib.datasets.utils.download import download
from PIL import Image
import torch
from torchvision import datasets as tv_datasets
import zipfile
from PIL import Image 
import torchvision.transforms as transforms 
import random
from math import floor as floor
from sklearn.model_selection import train_test_split
import pandas as pd

global processcount
processcount =0
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
from tqdm import tqdm

def get_filelist(split_directory, format):
    if format == "base":
        return [os.path.join(split_directory,file) for file in os.listdir(split_directory) if file.endswith('.png')] 
    else:
        allfiles=[]
       
        for prefix_1 in ["align", "conflict"]:
           for prefix_2 in ["0", "1"]:
               split_sub_dir = os.path.join(split_directory,
                                            prefix_1, 
                                            prefix_2)
               allfiles = allfiles + [os.path.join(split_sub_dir,file) for file in os.listdir(split_sub_dir) if file.endswith('.png')]
        return allfiles
           
def transform(df):
     class QuickDataset(torch.utils.data.Dataset):
        def __init__(self, df, key='Image'):
          self.X=df[key]
        def __len__(self):
           'Denotes the total number of samples'
           return len(self.X)
        def __getitem__(self, index):  
           return self.X[index]
    # Thanks to Saeid for this
    # https://stackoverflow.com/questions/52548174/how-to-remove-the-last-fc-layer-from-a-resnet-model-in-pytorch
     class Identity(nn.Module):
       def __init__(self):
         super().__init__()
        
       def forward(self, x):
         return x
     m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
     m.fc=Identity()
     #necessary to turn off batch norm
     m.eval()
     print(df.columns) 

     ds=QuickDataset(df)   
     dl=torch.utils.data.DataLoader(ds,batch_size=4)
     results=[]
     with torch.no_grad():
       for it, batch in enumerate(dl):
         #batchsize*1000 for resnet
         results+=m(batch)

     df.drop(columns=['Image'], inplace=True)
     df['img_features']=results

def process_bffhq(split_dir, sep='_', format="base"):
    im_list, target_class, protected_class, indexrange, img_features = [], [], [], [], []
    # Iterate over all images
    #file format is split/imageid_ageclass_gender.png
    #age = 0/young 1/old
    #gender 0/female, 1/male
    weights = ResNet18_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()
    filelist= get_filelist(split_dir, format)
    pd.set_option('display.max_columns', None)
    count=0
    for file in tqdm(filelist):
        print (count)

        ageclass = os.path.basename(file).split(sep)[1]  
        gender = os.path.basename(file).split(sep)[2].split('.')[0]

        with Image.open(file) as im_file:
           count+=1

           img_transformed = preprocess(im_file)
           
           im_list.append(img_transformed)
           target_class.append(int(ageclass))
           protected_class.append(int(gender))
           indexrange.append(file)

           if count==40:
               _df=pd.DataFrame({'Image': im_list})
               transform(_df) 
               img_features+=list(_df['img_features'])
               im_list=[]
               count=0


    return pd.DataFrame({ 'img_features': img_features, 'target': target_class, 'protected' : protected_class}, index=indexrange)
       


class bffhq:

    _NAME = "bffhq"
    _SPLITS = ["train", "dev", "test"]

    def __init__(self, dest_folder, batch_size):
        self.dest_folder = dest_folder
        self.batch_size = batch_size
        self.url="https://drive.usercontent.google.com/download?id=1Y4y4vYz6sRJRqS9jJyD06cUSR618g0Rp&export=download&confirm=t&uuid=a410fd39-bec8-4140-b374-fc158d3ac480"
        self.train_dir="0.5pct"
        self.test_dir="test"
        self.dev_dir="valid"
        filename=self.url.split('/')[-1].replace(" ", "_")  # be careful with file names
        self.abs_path=os.path.abspath(os.path.join(dest_folder, filename) )
        self.abs_dir=os.path.abspath(dest_folder)
        

    def download_files(self): 
        
        download(
                url= self.url,
                dest_folder = self.dest_folder
                ) 
        self.unzip(self.abs_path)
     
    def mixup(self, train, dev, test):
        trainl=train.shape[0]
        devl=dev.shape[0]
        testl=test.shape[0]
        tot = float(trainl+devl+testl)
        seed=2020

        indices_train=( range(trainl))
        sample_train = random.Random(seed).sample(indices_train,trainl)
        indices_dev=( range(devl))
        sample_dev = random.Random(seed).sample(indices_dev,devl)
        indices_test=( range(testl))
        sample_test = random.Random(seed).sample(indices_test,testl)
        
        indices_train_newtrain = (0, floor((trainl*trainl)/tot) )
        indices_train_newdev = ( floor((trainl*trainl)/tot)+1, floor((trainl*trainl)/tot) +1+ floor((devl*trainl)/tot) )
        indices_train_newtest = ( floor((trainl*trainl)/tot)+1+ floor((devl*trainl)/tot)+1, trainl )
        
        print ( indices_train_newtrain, indices_train_newdev, indices_train_newtest)
        indices_dev_newtrain = (0, floor((trainl*devl)/tot) )
        indices_dev_newdev = ( floor((trainl*devl)/tot)+1, floor((trainl*devl)/tot) +1+ floor((devl*devl)/tot) )
        indices_dev_newtest = ( floor((trainl*devl)/tot)+1+ floor((devl*devl)/tot)+1, devl )

        indices_test_newtrain = (0, floor((trainl*testl)/tot) )
        indices_test_newdev = ( floor((trainl*testl)/tot)+1, floor((trainl*testl)/tot) +1+ floor((devl*testl)/tot) )
        indices_test_newtest = ( floor((trainl*testl)/tot)+1+ floor((devl*testl)/tot)+1, devl )
        #shuffle the final df
        a = pd.concat( (
                train.iloc[ sample_train [indices_train_newtrain[0]: indices_train_newtrain[1] ]],
                dev.iloc[ sample_dev[ indices_dev_newtrain[0]: indices_dev_newtrain[1] ]],
                test.iloc[ sample_test [ indices_test_newtrain[0]: indices_test_newtrain[1]]]
                )
                ).sample(frac=1)
        b = pd.concat( (
                train.iloc[ sample_train[ indices_train_newdev[0]: indices_train_newdev[1]]],
                dev.iloc[ sample_dev[ indices_dev_newdev[0]: indices_dev_newdev[1]]],
                test.iloc[sample_test[ indices_test_newdev[0]: indices_test_newdev[1]]]
                )
                ).sample(frac=1)
        c = pd.concat( (
                train.iloc[ sample_train [indices_train_newtest[0]: indices_train_newtest[1]]],
                dev.iloc[sample_dev[ indices_dev_newtest[0]: indices_dev_newtest[1]]],
                test.iloc[sample_test[ indices_test_newtest[0]: indices_test_newtest[1]]]
                )
                ).sample(frac=1)
        return (a, b, c) 
        
    def unzip(self, abs_path):
        if os.path.exists(os.path.join(self.dest_folder, "bffhq", "0.5pct")) : return 
        with zipfile.ZipFile(abs_path, 'r') as zip_ref:
           zip_ref.extractall(self.dest_folder)

    def processing(self, mix=True):
       # num_train = len(self.train_mnist)
       # indices = list(range(num_train))
        seed_everything(2020)


        train = process_bffhq(os.path.join(self.abs_dir, "bffhq", self.train_dir), format="train")
        dev = process_bffhq(os.path.join(self.abs_dir,"bffhq", self.dev_dir))
        test = process_bffhq(os.path.join(self.abs_dir, "bffhq", self.test_dir))



        #colored_MNIST_train = process_bffhq(torch.utils.data.Subset(self.train_mnist, train_idx), ratio = 0.8)
        #colored_MNIST_dev = process_bffhq(torch.utils.data.Subset(self.train_mnist, valid_idx), ratio = 0.5)
        #colored_MNIST_test = process_bffhq(self.test_mnist, ratio = 0.5)
        if mix:
            (train, dev, test)= self.mixup(train.reset_index(), dev.reset_index(), test.reset_index())
        torch.save(train, os.path.join(self.dest_folder, "bffhq_train.pt"))
        torch.save(dev, os.path.join(self.dest_folder, "bffhq_dev.pt"))
        torch.save(test, os.path.join(self.dest_folder, "bffhq_test.pt"))


    def prepare_data(self):
        self.download_files()
        self.processing()


