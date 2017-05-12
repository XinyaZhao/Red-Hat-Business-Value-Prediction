##### Import Pacakges
import pandas as pd
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import numpy as np
import os
from IPython.display import Image
import sklearn
from sklearn import tree
import math
#from dstools import data_tools

import matplotlib.pylab as plt
import matplotlib.pylab as pylab


data_dept = pd.DataFrame.from_csv('processed_data.csv')
#Set X and Y
X = data_dept.drop('Target_Variable',1)
Y = data_dept['Target_Variable']
#Create Date Splits
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, train_size =.75)

#Test Initial Decision Tree Classifer

dt_model = DecisionTreeClassifier()
dt_model.fit(X_train,Y_train)
print('baseline model: %.4f' % metrics.accuracy_score(Y_test,dt_model.predict(X_test)))

# Tune DT by finding the best parameters with grid search


min_samples_leaf= [1,3,5,10,20, 50, 75, 100]
max_depth_list = [1,2,3,4,5,6,7,8,9,10]

num_splits_list = []

for i in range(1,20):
    num_splits = np.power(2, i)
    num_splits_list.append(num_splits)


grid_tree = {'min_samples_split': num_splits_list, 'min_samples_leaf': min_samples_leaf, 'max_depth': max_depth_list}
    
Tree_model = DecisionTreeClassifier(criterion="entropy")
Tree_model_clf = GridSearchCV(Tree_model, grid_tree)
Tree_model_clf.fit(X,Y)

print(Tree_model_clf.best_estimator_.min_samples_split)
print(Tree_model_clf.best_estimator_.min_samples_leaf)
print(Tree_model_clf.best_estimator_.max_depth)


