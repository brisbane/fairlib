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
from sklearn.model_selection import train_test_split
import pandas as pd

global processcount
processcount =0
from torchvision.models import resnet18, ResNet18_Weights
import random
import torch.nn as nn

from tqdm import tqdm

def get_filelist(split_directory):
        return [os.path.join(split_directory,file) for file in os.listdir(split_directory) if file.endswith('.jpg')] 
           
def preprocess(df):
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

def process_celeba(img_dir, annotations, sampleset, target_name, protected_name):
    global processcount

    processcount+=1
    im_list, target_class, protected_class = [], [], []

    indexrange=[f"{s}.jpg" for s in sampleset]

    weights = ResNet18_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()

    cols=['Image']
    annotations_df=pd.read_csv(annotations,  delim_whitespace=True, skiprows=1)#, names=cols)
    pd.set_option('display.max_columns', None)
    print (annotations_df.head())
    protecteds=annotations_df[protected_name]
    targets=annotations_df[target_name]

    count=0
    print(f"Iteration {processcount} of 6")

    for bfile in tqdm(indexrange):
        file=os.path.join(img_dir,bfile)
        protected = protecteds[bfile]
#        gender = os.path.basename(file).split(sep)[2].split('.')[0]
        target = targets[bfile]
        with Image.open(file) as im_file:

             # Apply it to the input image
           img_transformed = preprocess(im_file)
           im_list.append(img_transformed)

           target_class.append(int(target==1))
           protected_class.append(int(protected==1))

        count+=1
    return pd.DataFrame({ 'Image': im_list, 'target': target_class, 'protected' : protected_class}, index=indexrange)

class celeba:

    _NAME = "celeba"
    _SPLITS = ["train", "dev", "test"]

    def __init__(self, dest_folder, batch_size, target_name="Smiling", protected_name="Young", ):
        self.dest_folder = dest_folder
        self.batch_size = batch_size
        self.url="https://drive.usercontent.google.com/download?id=1badu11NqxGf6qM3PTTooQDJvQbejgbTv&export=download&authuser=0"
        self.train_dir="train"
        self.test_dir="test"
        self.dev_dir="valid"
        filename="CelebA.zip"
        #self.url.split('/')[-1].replace(" ", "_")  # be careful with file names
        self.abs_path=os.path.abspath(os.path.join(dest_folder, filename) )
        self.abs_dir=os.path.abspath(os.path.join(dest_folder, "CelebAMask-HQ"))
        self.img_dir = os.path.join(self.abs_dir, "CelebA-HQ-img")
        self.annotations = os.path.join(self.abs_dir, "CelebAMask-HQ-attribute-anno.txt")
        self.feature_extractor = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.target_name = target_name        
        self.protected_name = protected_name

    def download_files(self): 
        if os.path.exists(self.annotations): return
        if not os.path.exists(self.abs_path):
            print (f"Please download the file from {self.url} to {self.abs_path} first")
            raise(f"Please download the file from {self.url} to {self.abs_path} first")
        else:
            self.unzip(self.abs_path)
     
    
    def unzip(self, abs_path):
        with zipfile.ZipFile(abs_path, 'r') as zip_ref:
           zip_ref.extractall(self.dest_folder)

    def processing(self):
       # num_train = len(self.train_mnist)
       # indices = list(range(num_train))
        seed_everything(2020)
        datasettaskprotect=f"celebMHQ_{self.target_name}{self.protected_name}"
        indices=range(30000)
        seed=2020
        
        sample = random.Random(seed).sample(indices,30000)
        print (f"processing {datasettaskprotect}")

        dev_df = process_celeba(self.img_dir, self.annotations, sample[19500:22500], 
                                target_name=self.target_name, protected_name=self.protected_name)
        torch.save(dev_df, os.path.join(self.dest_folder, f"{datasettaskprotect}_dev.pt"))
        preprocess (dev_df)
        torch.save(dev_df, os.path.join(self.dest_folder, f"{datasettaskprotect}_dev_pre.pt"))
        del dev_df

        df0 = process_celeba(self.img_dir, self.annotations, sample[0:5000],
                                target_name=self.target_name, protected_name=self.protected_name)
        df1 = process_celeba(self.img_dir, self.annotations, sample[5000:10000],
                             target_name=self.target_name, protected_name=self.protected_name)
        df2 = process_celeba(self.img_dir, self.annotations, sample[10000:15000],
                             target_name=self.target_name, protected_name=self.protected_name)
        df3 = process_celeba(self.img_dir, self.annotations, sample[15000:19500],
                             target_name=self.target_name, protected_name=self.protected_name)
        torch.save(pd.concat([df0,df1,df2,df3]), os.path.join(self.dest_folder, f"{datasettaskprotect}_train.pt"))
        for df in [df0,df1,df2,df3]:
            preprocess (df)
        torch.save(pd.concat([df0, df1, df2, df3]), os.path.join(self.dest_folder, f"{datasettaskprotect}_train_pre.pt"))
        del df1, df2, df3, df0


        

        test_df = process_celeba(self.img_dir, self.annotations, sample[22500:30000],
                                 target_name=self.target_name, protected_name=self.protected_name)

        torch.save(test_df, os.path.join(self.dest_folder, f"{datasettaskprotect}_test.pt"))
        preprocess (test_df)
        torch.save(test_df, os.path.join(self.dest_folder, f"{datasettaskprotect}_test_pre.pt"))

        del test_df





    def prepare_data(self):
        self.download_files()
        self.processing()
