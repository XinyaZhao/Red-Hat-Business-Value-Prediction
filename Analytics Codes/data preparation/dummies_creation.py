import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
data = pd.read_csv('merge_sample_double.csv')
for c in data.columns:
    data[c].unique
dummies = []
for c in data.columns:
    if len(data[c].unique()) != 2 and len(data[c].unique())<100:
        dummies.append(c)
non_dummies = [d for d in data.columns if d not in dummies]
ds = []
for c in dummies:
    le = LabelEncoder()
    labels = le.fit_transform(data[c])
    enc = OneHotEncoder()
    dummy = enc.fit_transform(labels. reshape(-1,1))
    ds.append(dummy)
ds_frames = [pd.DataFrame(d.todense()) for d in ds]
dummy_data = pd.concat(ds_frames,axis=1)
concat = pd.concat([data[non_dummies],dummy_data],axis=1)
concat.to_csv('double sample with dummies.csv',index= False)
