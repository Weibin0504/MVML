#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
18.10.30
chi feature
'''

import numpy as np
import scipy as sp
import pandas as pd
from scipy.io import mmread
import time
import gc
from sklearn.feature_selection import chi2, SelectKBest

label_num = 5
sample = 19241
path = '/home/17-wb/wb/mvml/reutersdata/'


labels = np.loadtxt(path+'nNone_label.csv',delimiter=',')
sp = mmread(path+'sp_cont_tfidf.mtx')
tfidf = sp.toarray()
del sp
gc.collect()
y_label = np.where(labels>0,1,0)
print 'label',y_label.shape
selected_features = []
t0 = time.time()
for label in range(label_num):
    selector = SelectKBest(chi2, k='all')
    selector.fit(tfidf, y_label[:,label])
    selected_features.append(list(selector.scores_))

selected_features = np.max(np.array(selected_features), axis=0)>6.0#原来�?.5
compute_time = time.time() - t0
print 'compute_time',compute_time
print selected_features.shape
print('chi selected_features ok')
#np.savetxt('E:\\ruanian\\anaconda\\data\\result\\sift_val_chi2_1.csv', selected_features, delimiter = ',')
'''
chi_features = np.zeros((sample,1))
columns = 0
for chi in selected_features:
    if chi >0:
        chi_features = np.c_[chi_features,tfidf_title[:,columns]]
    columns += 1

chi_features = np.delete(chi_features, 0, axis=1)
'''
t1 = time.time()
feature = pd.DataFrame(selected_features).T
tfidf = pd.DataFrame(tfidf)
con_data = pd.concat([feature,tfidf],axis=0,ignore_index=True)
data = con_data.loc[:,con_data.iloc[0]>0]
data = data.drop([0])
data.to_csv(path+'chi_data/cont_chi_6.csv',header=False,index=False,sep=',')
compute_time = time.time() - t1
print 'compute_time',compute_time
print data.shape[0],data.shape[1]
#np.savetxt(path+'chi_data/cont_chi_5.csv', chi_features, delimiter = ',')
print('chi save_features ok')