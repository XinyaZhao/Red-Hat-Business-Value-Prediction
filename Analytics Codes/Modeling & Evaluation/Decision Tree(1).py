
# coding: utf-8

# In[7]:

import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

raw_data = pd.read_csv('merge_sample.csv')
X = raw_data.drop('act_outcome',1)
Y = raw_data['act_outcome']
X_train, X_test,Y_train,Y_test = train_test_split(X,Y,train_size = 0.8)


# In[8]:

dt = DecisionTreeClassifier(criterion='entropy')
dt = dt.fit(X_train,Y_train)

train_ac = dt.score(X_train,Y_train)
test_ac = dt.score(X_test,Y_test)


# In[9]:

#feature importance 
feature_mi = dt.feature_importances_
feature_mi_dict = dict(zip(X_train.columns.values, feature_mi))


# In[10]:

def plotUnivariateROC(preds, truth, label_string):
    fpr, tpr, thresholds = roc_curve(truth, preds)
    roc_auc = auc(fpr, tpr)
    if roc_auc < 0.5:
        fpr, tpr, thresholds = roc_curve(truth, -1 * preds)
        roc_auc = auc(fpr, tpr)
    #chooses a random color for plotting
    c = (np.random.rand(), np.random.rand(), np.random.rand())
    #create a plot and set some options
    plt.plot(fpr, tpr, color = c, label = label_string + ' (AUC = %0.3f)' % roc_auc) 
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC')
    plt.legend(loc="lower right")    
    return roc_auc


# In[11]:

fig = plt.figure(figsize = (12, 6))
ax = plt.subplot(111)

feature_auc_dict = {}
for col in X_train.columns:
    feature_auc_dict[col] = plotUnivariateROC(X_train[col], Y_train, col)
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.0 , box.width, box.height * 1])
ax.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.15), fancybox = True, 
              shadow = True, ncol = 4, prop = {'size':10})


# In[14]:

# See the relationship between feature importance and auc 
df_auc = pd.DataFrame(pd.Series(feature_auc_dict), columns = ['auc'])
df_mi = pd.DataFrame(pd.Series(feature_mi_dict), columns = ['mi'])
feat_imp_df = df_auc.merge(df_mi, left_index = True, right_index = True)


# In[15]:

#Now create a df that holds the ranks of auc and mi 
feat_ranks = feat_imp_df.rank(axis = 0, ascending = False)
#Plot the two ranks
plt.plot(feat_ranks.auc, feat_ranks.mi, '.')
#Plot a y=x reference line
plt.plot(feat_ranks.auc, feat_ranks.auc)


# In[28]:

fig = plt.figure(figsize = (12, 6))
ax = plt.subplot(111)
box = ax.get_position()
#Now create lists of top 31 features for both auc and mi
top_auc = list(feat_ranks[(feat_ranks.auc <= 31)].index.values)
top_mi = list(feat_ranks[(feat_ranks.mi <= 31)].index.values)
# top 31 features and auc is better 

fsets = [top_auc, top_mi]
fset_descr = ['auc', 'mi']

mxdepths = [5,10,15,20,25]

raw_data = pd.read_csv('merge_sample.csv')
X = raw_data.drop('act_outcome',1)
Y = raw_data['act_outcome']
X_tr, X_te,Y_tr,Y_te = train_test_split(X,Y,train_size = 0.8)

for i, fset in enumerate(fsets):
    descr = fset_descr[i]
    #set training and testing data
    Y_train = Y_tr
    X_train = X_tr[fset]
    Y_test = Y_te
    X_test = X_te[fset]    
    for d in mxdepths:
        dt = DecisionTreeClassifier(criterion = 'entropy', max_depth = d)
        dt.fit(X_train, Y_train)
        score = dt.score(X_test,Y_test)
        preds_dt = dt.predict_proba(X_test)[:, 1]       
        plotUnivariateROC(preds_dt, Y_test,'{}:Tree_max_depth={}:(SCORE={})'.format(descr, d,round(score,3)))

# Put a legend below current axis

ax.set_position([box.x0, box.y0 + box.height * 0.0 , box.width, box.height * 1])
ax.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.15), fancybox = True, 
              shadow = True, ncol = 4, prop = {'size':10})
plt.show()


# In[27]:

#improve  split and leaf
min_samples_split_values = (100,1000,2000,3000,4000,5000,6000,7000,8000,10000)
min_samples_leaf_values = (5,10,15,20,25,30,35,40,45,50)

# fig = plt.figure()
# ax=fig.add_subplot(111)
res = []
for leaf in min_samples_leaf_values:
    clf = DecisionTreeClassifier(criterion='entropy', min_samples_leaf = leaf)
    clf = clf.fit(X_train, Y_train)
    res.append(clf.score(X_test, Y_test))    
plt.plot(min_samples_leaf_values,res)

plt.show()


# In[25]:

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
# Code here
fig = plt.figure()
ax=fig.add_subplot(111)
max_ac = 0
best_leaf = 1
best_split = 1
for leaf in min_samples_leaf_values:
    x_ls = []
    y_val = []
    for split in min_samples_split_values:
        fdt = DecisionTreeClassifier(criterion='entropy',min_samples_split = split,min_samples_leaf = leaf)
        fdt = fdt.fit(X_train,Y_train)
        x_ls.append(split)
        ac = fdt.score(X_test,Y_test)
        y_val.append(ac)
        if(max_ac < ac):
            max_ac = ac
            best_leaf = leaf
            best_split = split
    plt.plot(x_ls,y_val,label='leaf = {}'.format(leaf))
plt.legend() 
ax.set_xlabel('Min Split Size')
ax.set_ylabel('Test Set Accuarcy')
plt.show()


# In[ ]:



