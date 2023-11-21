import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv
import math
import sklearn
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import LSTM
from keras.layers.recurrent import GRU
from sklearn.metrics import mean_squared_error
from keras.callbacks import EarlyStopping

mydata = pd.read_excel(r"D:\Shayan\Our Conference Artice\Codes & Data\SHAPNA.xls")
mydata = mydata.values #convert to numpy array

Date = mydata[:,0]
Open = mydata[:,1]
High = mydata[:,2]
Low = mydata[:,3]
Close = mydata[:,4]
Last = mydata[:,5]
Vol = mydata[:,6]


X = np.zeros([mydata.shape[0], mydata.shape[1]+4])
for i in np.arange(4, mydata.shape[0]):
    X[i, :] = np.append(mydata[i,:],[mydata[i-1, 4], mydata[i-2, 4], mydata[i-3, 4], mydata[i-4, 4]])

Data = X[4:-1, 1:]
Y = Close[5:]

# Normalizing the data
mn1 = np.min(Data, axis=0)
mx1 = np.max(Data, axis=0)
mn2 = np.min(Y, axis=0)
mx2 = np.max(Y, axis=0)

NData = (Data - mn1)/(mx1 - mn1)
NY = (Y - mn2)/(mx2 - mn2)

# Make X_train & X_test
X_train = NData[:int(0.7*NData.shape[0]),:]
X_test = NData[int(0.7*NData.shape[0]):,:]

# Making Y_train & Y_test
Y_train = NY[:int(0.7*Y.shape[0])]
Y_test = NY[int(0.7*Y.shape[0]):]

# X_train = X_train.reshape(len(X_train),len(X_train[0]),1)
# X_test = X_test.reshape(len(X_test),len(X_test[0]),1)

X_train = X_train.reshape((-1,1,10))
X_test = X_test.reshape((-1,1,10))

# -----------------------------------------------------------------------------
# Define custom loss functions for regression in Keras
# -----------------------------------------------------------------------------
# root mean squared error (rmse) for regression
def rmse(y_true, y_pred):
    from keras import backend
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

# mean squared error (mse) for regression
def mse(y_true, y_pred):
    from keras import backend
    return backend.mean(backend.square(y_pred - y_true), axis=-1)

# coefficient of determination (R^2) for regression
def r_square(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))

def r_square_loss(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - ( 1 - SS_res/(SS_tot + K.epsilon()))

# ---------------------------------------------------
# Create the model
# ---------------------------------------------------
model_name = 'stock_price_GRU'

model = Sequential()
model.add(GRU(units=64, return_sequences=True, input_shape=(1, 10)))
# model.add(Dense(32, activation='relu'))
model.add(GRU(units=32))
# model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(optimizer="Nadam", loss="mean_squared_error", metrics=["mean_squared_error", rmse, r_square])

# model = load_model("{}.h5".format(model_name))
# print("MODEL-LOADED")

earlystopping = EarlyStopping(monitor="mean_squared_error", patience=25, verbose=1, mode='auto')
result = model.fit(X_train,Y_train,batch_size=256, epochs=250, verbose=1, callbacks=[earlystopping])
model.save("{}.h5".format(model_name))
print('MODEL-SAVED')

trainPredict = model.predict(X_train)
testPredict = model.predict(X_test)

# --------------------------------------------------------
# Plotting The figures
# --------------------------------------------------------
# comparison between net and actual for test set
plt.figure(1)
plt.plot(Y_test, '-b', label='actual')
plt.plot(testPredict, '-r', label='network')
plt.xlabel('day')
plt.ylabel('Price')
plt.title('Comparison between Network and Actual for Test set')
plt.legend()
plt.show()


# comparison between net and actual for train set
plt.figure(2)
plt.plot(Y_train, '-b', label='actual')
plt.plot(trainPredict, '-r', label='network')
plt.xlabel('day')
plt.ylabel('Price')
plt.title('Comparison between Network and Actual for Train set')
plt.legend()
plt.show()


# plot training curve for R^2
plt.figure(3)
plt.plot(result.history['r_square'])
plt.title('model R^2')
plt.ylabel('R^2')
plt.xlabel('epoch')
plt.show()


# plot training curve for rmse
plt.figure(4)
plt.plot(result.history['rmse'])
plt.title('RMSE')
plt.ylabel('RMSE')
plt.xlabel('epoch')
plt.show()


# plot training curve for mse
plt.figure(5)
plt.plot(result.history['mean_squared_error'])
plt.title('MSE')
plt.ylabel('MSE')
plt.xlabel('epoch')
plt.show()

# -----------------------------------------------------------------------------
# print statistical figures of merit
# -----------------------------------------------------------------------------
print("\n")
print("Mean absolute error (MAE):      %f" % sklearn.metrics.mean_absolute_error(Y_test, testPredict))
print("Mean squared error (MSE):       %f" % sklearn.metrics.mean_squared_error(Y_test, testPredict))
print("Root mean squared error (RMSE): %f" % math.sqrt(sklearn.metrics.mean_squared_error(Y_test, testPredict)))
print("R square (R^2):                 %f" % sklearn.metrics.r2_score(Y_test, testPredict))