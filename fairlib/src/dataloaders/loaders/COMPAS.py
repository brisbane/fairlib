import numpy as np
from ..utils import BaseDataset
from pathlib import Path
import pandas as pd
import os
from sklearn.preprocessing import  OneHotEncoder
class COMPASDataset(BaseDataset):

    def load_data(self):
        if self.args.seed !=1:
            self.data_file = os.path.join(self.args.data_dir,self.args.seed, "COMPAS_{}.pkl".format(self.split))
        else:
            self.data_file = os.path.join(self.args.data_dir, "COMPAS_{}.pkl".format(self.split))

        data = pd.read_pickle(self.data_file)
        print ("datafile:", self.data_file )

        basis=['sex', 'race', 'is_recid', 'age']
        use_sex_predictor=False
        use_race_predictor=False
        if use_sex_predictor :
           basis=['is_recid', 'age','race']
        #other=0, black=1, caucasian=2
        #include race as predictor
        if use_race_predictor:
            data = pd.concat( [pd.concat( {'_': pd.get_dummies(data['race'], prefix='race', drop_first=True)}.values(), axis=1), data], axis=1)

        han=basis 
        #print (data.columns)
        self.X = data.drop(han, axis=1)
        self.columns=self.X.columns
        self.X=self.X.to_numpy().astype(np.float32)
#       self.X = data.drop(['age'], axis=1).to_numpy().astype(np.float32)
        self.y = list(data["is_recid"].astype(int))

        if self.args.protected_task == "gender":
            self.protected_label =np.array(list(data["sex"])).astype(np.int32) # Gender
        elif self.args.protected_task == "race":
            self.protected_label = np.array(list(data["race"])).astype(np.int32) # Race
        elif self.args.protected_task == "intersection":
            self.protected_label = np.array(
                [_r+_s*3 for _r,_s in zip(list(data["race"]), list(data["sex"]))]
                ).astype(np.int32) # Intersectional
