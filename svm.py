# -*- coding: utf-8 -*-
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from pandas import read_pickle
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
import sklearn as skl
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import model_from_json
from keras.regularizers import L1L2
from math import sqrt
from matplotlib import pyplot
from sklearn.svm import SVR
import numpy as np
import sys
sys.path.insert(0, 'D:\\Programming\\research\\LSTM\\Buildings\\Repo')
import repository as rp

#read the data set
dataset = read_pickle("20182019Data.pkl")
#interpolate missing samples
dataset=dataset.interpolate(method='linear')
#get aggregated energy
dataset['Aggregated'] = dataset.apply(lambda row: row.CoolE + row.HeatE, axis = 1)
#scaling the data
scaler = MinMaxScaler(feature_range=(0, 1))
first=scaler.fit_transform(dataset.values[:,0:3])
last=scaler.fit_transform(dataset.values[:,4:8])
minvect=np.ones((dataset.values.shape[0],1))*48.98
maxvect=np.ones((dataset.values.shape[0],1))*91.66
dischtempscaled=((dataset.values[:,3]-minvect.T)/(maxvect.T - minvect.T)).T
totalvect=np.hstack((first,dischtempscaled,last))
dataset = DataFrame(totalvect, columns=dataset.columns, index=dataset.index)
#separate continuous sequences (because of 15 days missing)
dataset1=dataset.loc['2018-07-22':'2018-09-30']
dataset2=dataset.loc['2018-10-16':'2019-05-28']
#get data from Spring
dataset2=dataset2.loc['2019-03-01':'2019-05-31']#spring
#drop useless variables
dataset2=dataset2.drop(['CoolE', 'HeatE'], axis=1)
#reorganize data to predict future observation [x_(t-1) y_t]
futureAgg=dataset2['Aggregated'].shift(-1)
dataset2['Aggregated']=futureAgg
dataset2 = dataset2[~np.isnan(dataset2['Aggregated'])]
#configure shape for each sample
varIndex=np.array([4])#vector containing the index of the output variable, zero-index
delays=np.array([0]) #delay in the output variable to include as input
#extract continuous sequences
sequencelength=1
inputArray,outputArray=rp.continuous_overlapping_sequences(dataset2,sequencelength,varIndex,delays);
#training/testing separation
#sequential splitting
Nosamples=inputArray.shape[0]
#proportion=round(3/4*Nosamples)
#train_X,test_X =inputArray[0:proportion,:], inputArray[proportion:Nosamples,:]
#train_y, test_y = outputArray[0:proportion,], outputArray[proportion:Nosamples,]
#random splitting
randseqorder = np.random.permutation(range(Nosamples))
proportion=round(3/4*Nosamples)
train_X,test_X =inputArray[randseqorder[0:proportion],:], inputArray[randseqorder[proportion:Nosamples],:]
train_y, test_y = outputArray[randseqorder[0:proportion],], outputArray[randseqorder[proportion:Nosamples],]

#reshaping before training
outputfeatures=1
n_features=5
input_features=n_features+delays-outputfeatures
n_steps=sequencelength*input_features[0]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], sequencelength*input_features[0]))
test_X = test_X.reshape((test_X.shape[0], sequencelength*input_features[0]))

# design model ############################################################
#svr_rbf = SVR(kernel='rbf', C=60, gamma=0.1, epsilon=0.1)
#svr_lin = SVR(kernel='linear', C=100, gamma='auto')
#svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.005,
#               coef0=1)
#
#model=svr_lin
#model.fit(train_X, train_y)

#grid search
parameters = {'kernel': ['rbf'], 'C':[1.5, 10],'gamma': [1e-7, 1e-4],'epsilon':[0.1,0.2,0.5,0.3]}
svr = SVR()
clf = skl.model_selection.GridSearchCV(svr, parameters)
clf.fit(train_X,train_y)
clf.best_params_


#noepochs=150
#history = model.fit(train_X, train_y, epochs=noepochs, validation_data=(test_X, test_y), verbose=2)

#saving the model
rp.save_sklearnmodel(model)

# loading and testing network ############################################################

#model=rp.load_sklearnmodel()

train_pred= model.predict(train_X)
# calculate RMSE training
rmsetrain = sqrt(mean_squared_error(train_y, train_pred))
print('Train RMSE: %.3f' % rmsetrain)

test_pred= model.predict(test_X)
# calculate RMSE testing
rmsetest = sqrt(mean_squared_error(test_y, test_pred))
print('Test RMSE: %.3f' % rmsetest)

# plot error distribution
mae=abs(test_y[:,].T- test_pred[:,].T)
fig1, ax1 = pyplot.subplots()
ax1.set_title('Basic Plot')
ax1.boxplot(mae.T)

test_pred_continuous=test_pred
test_y_continuous=test_y
test_y_continuoustest=test_y

#plot predicted and real variables
pyplot.figure()
pyplot.plot(test_y, 'b*-',label='original')
pyplot.plot(test_pred,'r+-', label='predicted')
pyplot.ylabel('BTUs')
pyplot.xlabel('Samples')
pyplot.legend()
pyplot.show()

#entire dataset prediction
totalX=inputArray
totaly=outputArray
total_pred= model.predict(totalX)

test_pred_continuous=total_pred
test_y_continuous=totaly

rmsetrain = sqrt(mean_squared_error(totaly, total_pred))
print('Total RMSE: %.3f' % rmsetrain)
mae=abs(totaly[:,].T- total_pred[:,].T)
print('Total MAE: %.3f' % np.mean(mae))

pyplot.figure()
pyplot.plot(totaly, 'b*-',label='original')
pyplot.plot(total_pred,'r+-', label='predicted')
pyplot.ylabel('BTUs')
pyplot.xlabel('Samples')
pyplot.legend()
pyplot.show()

