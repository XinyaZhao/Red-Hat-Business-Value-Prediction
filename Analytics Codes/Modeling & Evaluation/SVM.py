
# coding: utf-8

# In[38]:

import numpy as np
import pandas as pd
import os
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
import matplotlib
import matplotlib.pyplot as plt


get_ipython().magic('matplotlib inline')
cwd = os.getcwd()
datafile = '/'.join(cwd.split('/')[0:-1]) + '/data/processed data/dummy_data.csv'
data = pd.read_csv(datafile,sep = ',')
#print(data.shape) 


# In[39]:

# Feature selection using regularization
def selection(data):
    Y = data['act_outcome']
    X = data.drop('act_outcome',1)
    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, Y)
    model = SelectFromModel(lsvc, prefit=True)
    new_X = model.transform(X)
    columns = []
    for i in range(len(model.get_support())):
        if model.get_support()[i]:
            columns.append(X.columns[i])
    new_data = pd.DataFrame(new_X,columns=columns)
    new_data['act_outcome']=Y
    return new_data


# In[40]:

#feature selection
new_data=selection(data)
sampled_data=new_data.sample(frac=0.2)
#sampled_data.shape


# In[41]:

#define a function to print ROC curves. 
#It should take in only arrays/lists of predictions and outcomes

def plotUnivariateROC(preds, truth, label_string):
    '''
    preds is an nx1 array of predictions
    truth is an nx1 array of truth labels
    label_string is text to go into the plotting label
    '''
    
    fpr, tpr, thresholds = roc_curve(truth, preds)
    roc_auc = auc(fpr, tpr)
    
    #we are doing this as a special case because we are sending unfitted predictions
    #into the function
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


# In[42]:

data_train=sampled_data.sample(frac=0.8)
data_test= sampled_data.loc[~sampled_data.index.isin(data_train.index)]
#data_train.shape,data_test.shape


# In[43]:

Y_train = data_train['act_outcome']
X_train = data_train.drop('act_outcome',1)
Y_test = data_test['act_outcome']
X_test = data_test.drop('act_outcome',1)


# In[44]:

tuned_parameters = [{'kernel': ['rbf'], 'gamma':[1e-6,1e-5,1e-4,1e-3,1e-2],'C': [1,10,100,1000,10000]}]
clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=3)
clf.fit(X_train, Y_train)
clf.best_params_


# In[47]:

svm = SVC(probability=True,C=1000,gamma=0.0001)
svm.fit(X_train,Y_train)


# In[48]:

# SVM results with feature selection and hyper-parameter tuning
svm_predict = svm.predict_proba(X_test)[:,1]
plotUnivariateROC(svm_predict, Y_test, "SVM") #Auc = 0.910


# In[49]:

svm.score(X_test,Y_test) #accuracy : 0.860


# In[ ]:



