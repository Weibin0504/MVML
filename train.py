#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
two view reduce V
'''
import sklearn.metrics as skm
import numpy as np
import datetime
from sklearn.model_selection import train_test_split

# Initialize data.
rho = 100.0/2
lamda = 10.0

alfa_1 = 1.0
alfa_2 = 1.0

corpus = 15000#1500
cont_feature = 5363#103
com_feature = 4268#1437
labels = 5
V_k = 4000

loadbegin_time = datetime.datetime.now()
path = '/home/17-wb/wb/mvml/reutersdata/dataset/'
data = np.mat(np.loadtxt(path+'train/train.csv',delimiter=','))
X1 = data[:,0:cont_feature]
X2 = data[:,cont_feature:cont_feature+com_feature]
Y = data[:,cont_feature+com_feature:5+cont_feature+com_feature]
#X_train, x_test, Y, y_ = train_test_split(X, Y_data, test_size=0.2)
corpus = X1.shape[0]
print corpus
cont = np.mat(np.loadtxt(path+'val/val_cont.csv',delimiter=','))
com = np.mat(np.loadtxt(path+'val/val_title.csv',delimiter=','))
te_true_label = np.mat(np.loadtxt(path+'val/val_y.csv',delimiter=','))


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
