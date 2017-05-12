
# coding: utf-8

# In[18]:

import pandas as pd
import numpy as np


# In[19]:

# Get Random Sample
act_data = pd.read_csv('act_train.csv')
act_sample = act_data.sample(frac=0.05)
pID = act_sample['people_id'].to_frame()
pID = pID.drop_duplicates()
people_data = pd.read_csv('people.csv')
people_sample = pd.merge(people_data, pID, on='people_id', how= 'inner')


# In[20]:

a = pd.Series(list(pID['people_id'].sort_values()))
p = pd.Series(list(people_sample['people_id'].sort_values()))
print(a.equals(p)) #True


# In[25]:

print([len(act_sample),len(people_sample)]) #(109865, 49629)


# In[26]:

act_sample.to_csv('act_sample.csv',index=False)
people_sample.to_csv('people_sample.csv', index=False)

