import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#Load Data X, Y
X = np.load('Feature_Matrix.npy')
Y = np.load('Energy_Matrix.npy')

iAd = np.arange(X.shape[0])
random.Random(2020).shuffle(iAd)

iAd_test = iAd[:100]
iAd_validation=iAd[100: 200]
iAd_train = iAd[200:]

X_test, Y_test = X[iAd_test], Y[iAd_test]
X_validation, Y_validation = X[iAd_validation], Y[iAd_validation]
X_train, Y_train = X[iAd_train], Y[iAd_train]

#Define Structure of neural Network
n2b = 12 
n3b = 3 
nL1 = 50 #number of neurons in first layer
nL2 = 30 #number of neurons in second layer
learningRate= 0.0001

#Set up neural network
nFeat = X.shape[1]
tf_feat = tf.placeholder(tf.float32, (None, nFeat))
tf_engy = tf.placeholder(tf.float32, (None))
tf_labels = tf.placeholder(tf.float32, (None))
tf_LR = tf.placeholder(tf.float32)
w1 = tf.Variable(tf.random_normal([nFeat, nL1], stddev=0.01, seed=100))
w2 = tf.Variable(tf.random_normal([nL1, nL2], stddev=0.01, seed=200))
w3 = tf.Variable(tf.random_normal([nL2], stddev=0.01, seed=300))
b1 = tf.Variable(tf.zeros(nL1))
b2 = tf.Variable(tf.zeros(nL2))
b3 = tf.Variable(tf.zeros(1))
L1 = tf.nn.sigmoid(tf.matmul(tf_feat, w1) + b1)
L2 = tf.nn.sigmoid(tf.matmul(L1,w2) + b2)
#L1 = tf.nn.relu(tf.matmul(tf_feat, w1) + b1)
#L2 = tf.nn.relu(tf.matmul(L1,w2) + b2)
L3 = tf.tensordot(L2, w3,axes=1) + b3

engyLoss = tf.reduce_mean((L3 - tf_engy)**2)
optimizer = tf.train.AdamOptimizer(tf_LR).minimize(engyLoss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()

train_FeedDict = {tf_feat:X_train, tf_engy:Y_train, tf_LR:learningRate}
validation_FeedDict = {tf_feat:X_validation, tf_engy:Y_validation, tf_LR:learningRate}
test_FeedDict = {tf_feat:X_test, tf_engy:Y_test, tf_LR:learningRate}

epoch_list=[]
eLoss_list=[]
veLoss_list=[]
teLoss_list=[]
for epoch in range(11801):
    sess.run(optimizer, feed_dict=train_FeedDict)
    if epoch % 200 == 0:
        
        eLoss = sess.run(engyLoss, feed_dict=train_FeedDict)
        veLoss = sess.run(engyLoss, feed_dict=validation_FeedDict)
        teLoss = sess.run(engyLoss, feed_dict=test_FeedDict)
        
        epoch_list.append(epoch)
        eLoss_list.append(np.sqrt(eLoss))
        veLoss_list.append(np.sqrt(veLoss))
        teLoss_list.append(np.sqrt(teLoss))
        
        
        print(epoch, np.sqrt(eLoss))
        print(epoch, np.sqrt(veLoss))
        print(epoch, np.sqrt(teLoss))
        print(" ")
    
#save_path = saver.save(sess, "./model.ckpt")
        
###############################################################################
Y_train_pre = sess.run(L3, feed_dict=train_FeedDict)
Y_test_pre = sess.run(L3, feed_dict=test_FeedDict)
Y_validation_pre = sess.run(L3, feed_dict=validation_FeedDict)

np.save('Y_train_DFT', Y_train)
np.save('Y_train_NN', Y_train_pre)
np.save('Y_test_DFT', Y_test)
np.save('Y_test_NN', Y_test_pre)
np.save('Y_validation_DFT', Y_validation)
np.save('Y_validation_NN', Y_validation_pre)
np.save('epoch', epoch_list)
np.save('rmse_train', eLoss_list)
np.save('rmse_test', teLoss_list)
np.save('rmse_validation', veLoss_list)