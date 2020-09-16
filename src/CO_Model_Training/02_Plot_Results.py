import os
import random
import numpy as np
import matplotlib.pyplot as plt

Y_train, Y_train_pre =  np.load('Y_train_DFT.npy'), np.load('Y_train_NN.npy')
Y_test, Y_test_pre = np.load('Y_test_DFT.npy'), np.load('Y_test_NN.npy')
Y_validation, Y_validation_pre = np.load('Y_validation_DFT.npy'), np.load('Y_validation_NN.npy')
epoch_list = np.load('epoch.npy')
eLoss_list = np.load('rmse_train.npy')
teLoss_list = np.load('rmse_test.npy')
veLoss_list = np.load('rmse_validation.npy')


x_range=np.arange(-1.4,-0.2,0.01)
fig1 = plt.figure()
fig1.set_size_inches(6, 6)
plt.plot(Y_train, Y_train_pre,'ro', label='RMSE = {:.4f} eV'.format(eLoss_list[-1]))
plt.plot(x_range,x_range,'k')
plt.xlim(-1.4,-0.2)
plt.ylim(-1.4,-0.2)
plt.title('CO Prediction Results for Training Set (Size=1184)')
plt.xlabel('AdE_ReQM/eV')
plt.ylabel('AdE_Preiction/eV')
plt.legend()
fig1.savefig('Fig_Training.pdf')

x_range=np.arange(-1.4,-0.2,0.01)
fig2 = plt.figure()
fig2.set_size_inches(6, 6)
plt.plot(Y_validation, Y_validation_pre,'ro', label='RMSE = {:.4f} eV'.format(veLoss_list[-1]))
plt.plot(x_range,x_range,'k',linewidth=2)
plt.xlim(-1.4,-0.2)
plt.ylim(-1.4,-0.2)
plt.title('CO Prediction Results for Validation Set (Size=100)')
plt.xlabel('AdE_ReQM/eV')
plt.ylabel('AdE_Preiction/eV')
plt.legend()
fig2.savefig('Fig_Validation.pdf')

x_range=np.arange(-1.4,-0.2,0.01)
fig3 = plt.figure()
fig3.set_size_inches(6, 6)
plt.plot(Y_test, Y_test_pre,'ro', label='RMSE = {:.4f} eV'.format(teLoss_list[-1]))
plt.plot(x_range,x_range,'k',linewidth=3)
plt.xlim(-1.4,-0.2)
plt.ylim(-1.4,-0.2)
plt.xlabel('AdE_ReQM/eV')
plt.ylabel('AdE_Preiction/eV')
plt.title('CO Prediction Results for Testing Set (Size=100)')
plt.legend()
fig3.savefig('Fig_Testing.pdf')

