#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
two view reduce V
'''

import numpy as np
#import pandas as pd
import datetime
# Initialize data.

rho = 1000.0/2
alfa_1 = 1.0
alfa_2 = 1.0
lamda = 0.1
corpus = 13052#1500
cont_feature = 5297#103
com_feature = 5039#1437
labels = 8
beta = 2.0
test_data = 3263#917
train_data = 13052
data_long = 16315
V_k = 3000
data_5fold = [data_long/5,data_long/5*2,data_long/5*3,data_long/5*4,data_long]

loadbegin_time = datetime.datetime.now()
data = np.loadtxt("/home/17-wb/wb/mvml/python/data/twoview_sample_com3_cont4_data.csv",delimiter=',')
Y_train = data[0:data_5fold[3],0:8]
X_cont_train = data[0:data_5fold[3],8:5305]
X_com_train = data[0:data_5fold[3],5305:10344]
Y_test = data[data_5fold[3]:data_long,0:8]
X_cont_test = data[data_5fold[3]:data_long,8:5305]
X_com_test =data[data_5fold[3]:data_long,5305:10344]
X1 = np.mat(X_cont_train)
X2 = np.mat(X_com_train)
Y = np.mat(Y_train)
cont = np.mat(X_cont_test)
com = np.mat(X_com_test)
te_true_label = np.mat(Y_test)
tr_true_label = Y


A1 = np.mat(np.random.randn(V_k,cont_feature))
A2 = np.mat(np.random.randn(V_k,com_feature))
P = np.mat(np.random.randn(corpus,V_k))
Z = np.mat(np.random.randn(corpus,V_k))
C = np.mat(np.random.randn(V_k,labels))
B = np.mat(np.random.randn(corpus,V_k))
W1 = np.mat(np.random.randn(cont_feature,V_k))
W2 = np.mat(np.random.randn(com_feature,V_k))

I_1 = np.mat(np.eye(V_k,V_k))
I_2 = np.mat(np.eye(cont_feature,cont_feature))
I_3 = np.mat(np.eye(com_feature,com_feature))
ITE = 0
W1_new = W1
W2_new = W2
loadend_time = datetime.datetime.now()
print 'load data time',(loadend_time-loadbegin_time)
# train data
loss_new = 0
while(ITE<10 or loss<0.0001):
	comput_begin = datetime.datetime.now()
	a_1 = np.linalg.inv((X1.T*X1).T*(X1.T*X1)+lamda*I_2)*(X1.T*X1).T
	a_2 = np.linalg.inv((X2.T*X2).T*(X2.T*X2)+lamda*I_3)*(X2.T*X2).T

	b_1 = X1.T*(alfa_1*X1*A1.T +alfa_2*X2*A2.T + Y*C.T + rho*(B-P))
	b_2 = X2.T*(alfa_1*X1*A1.T +alfa_2*X2*A2.T + Y*C.T + rho*(B-P))

	c_1 = alfa_1*A1*A1.T+alfa_2*A2*A2.T+C*C.T+rho*I_1
	W1 = a_1*b_1*(np.linalg.inv(c_1.T*c_1+lamda*I_1)*c_1.T)-np.linalg.inv(X1.T*X1+lamda*I_2)*X1.T*(X1*W1+X2*W2)
	W2 = a_2*b_2*(np.linalg.inv(c_1.T*c_1+lamda*I_1)*c_1.T)-np.linalg.inv(X2.T*X2+lamda*I_3)*X2.T*(X1*W1+X2*W2)
	#print 'W1:',W1
	#print 'W2:',W2
	W1_err = np.linalg.norm(W1-W1_new)
	W2_err = np.linalg.norm(W2-W2_new)
	W1_new = W1
	W2_new = W2
	W1_norm = np.linalg.norm(W1)
	W2_norm = np.linalg.norm(W2)
	d_1 = X1*W1+X2*W2
	d_2 = np.linalg.inv(d_1.T*d_1+lamda*I_1)*d_1.T
	A1 = d_2*X1
	A2 = d_2*X2

	C = d_2*Y
	#print 'C:',C

	Z = d_1+B
	U, s, V = np.linalg.svd(Z)
	P = np.mat(U)*np.mat(np.eye(corpus,V_k))*np.mat(V.T)
	#print 'P:',P
	B = B+d_1-P
	loss = np.linalg.norm(Y-P*C)
	err = (loss - loss_new)/loss
	loss_new = loss
	print 'loss2:',loss

	ITE+=1
	comput_end = datetime.datetime.now()
	print 'computer time',(comput_end-comput_begin)
	print "iteration",ITE
