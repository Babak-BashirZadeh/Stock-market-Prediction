import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import xlrd
from sklearn.svm import SVR,NuSVR, LinearSVR

mydata = pd.read_excel("SHAPNA.xls")
mydata = mydata.values #convert to numpy array

Date = mydata[:,0]
Open = mydata[:,1]
High = mydata[:,2]
Low = mydata[:,3]
Close = mydata[:,4]
Last = mydata[:,5]
Vol = mydata[:,6]

Data = []
nextday = 8
for i in np.arange(4, mydata.shape[0]-nextday):
    # X[i, :] = np.append(mydata[i,:],[mydata[i-1, 4], mydata[i-2, 4], mydata[i-3, 4], mydata[i-4, 4]])
    Data.append([Date[i],
                 Open[i-1], Open[i-2], Open[i-3], Open[i-4],
                 High[i-1], High[i-2], High[i-3], High[i-4],
                 Low[i-1], Low[i-2], Low[i-3], Low[i-4],
                 Last[i-1], Last[i-2], Last[i-3], Last[i-4],
                 Vol[i-1], Vol[i-2], Vol[i-3], Vol[i-4],
                 Close[i-1], Close[i-2], Close[i-3], Close[i-4],
                 Close[i+nextday]])

Data = np.array(Data)
# Data = X[4:-1, 1:]
X = Data[:, 1:-1]
Y = X[:, -1]
# ali

# Normalizing the data
mn1 = np.min(X, axis=0)
mx1 = np.max(X, axis=0)
mn2 = np.min(Y, axis=0)
mx2 = np.max(Y, axis=0)

NX = (X - mn1)/(mx1 - mn1)
NY = (Y - mn2)/(mx2 - mn2)

# Make X_train & X_test
X_train = NX[:int(0.7*NX.shape[0]),:]
X_test =  NX[int(0.7*NX.shape[0]):,:]

# Making Y_train & Y_test
Y_train = NY[:int(0.7*Y.shape[0])]
Y_test = NY[int(0.7*Y.shape[0]):]
Y_train_unorm = Y[:int(0.7*Y.shape[0])]
Y_test_unorm = Y[int(0.7*Y.shape[0]):]


# from sklearn.preprocessing import StandardScaler
# Sc_Xtrain = StandardScaler()
# # Sc_Ytrain = StandardScaler()
#
# X_train = Sc_Xtrain.fit_transform(X_train)
# Y_train = np.ravel(Y_train)
#
# Sc_Xtest = StandardScaler()
# Sc_Ytrain = StandardScaler()

# X_test = Sc_Xtest.fit_transform(X_test)
# Y_test = np.ravel(Y_test)

# my_net = SVR(kernel='rbf', degree=3, C= 1e3, gamma= 0.1, coef0=0.0, tol=0.001, epsilon=0.1,
#                 shrinking=True, cache_size=200, verbose=True, max_iter=200)
my_net = NuSVR(nu=0.5, C=1, kernel='rbf', gamma='scale')
from sklearn.neural_network import MLPRegressor
# my_net = MLPRegressor(hidden_layer_sizes=(40, 10), activation='tanh', solver='adam', batch_size=5, learning_rate_init=0.001
#                       , max_iter=200, shuffle=True, tol=10e-7, verbose=True,
#                       early_stopping=True, validation_fraction=0.15)

my_net.fit(X_train, Y_train)

testPredict = my_net.predict(X_test)
trn_Predict = my_net.predict(X_train)
testPredict_unorm = testPredict*(mx2 - mn2) + mn2
trn_Predict_unorm = trn_Predict*(mx2 - mn2) + mn2

trn_R2_norm = my_net.score(X_train, Y_train)
tst_R2_norm = my_net.score(X_test, Y_test)

from sklearn.metrics import r2_score
tst_R2_unorm = r2_score(Y_test_unorm, testPredict_unorm)

print('train R2 Score: %f, test R2 Score: %f, test Normalized R2 Score: %f'
      %(trn_R2_norm, tst_R2_norm, tst_R2_unorm))

plt.figure(1)
plt.plot(Y_test_unorm, '-b', label='actual')
plt.plot(testPredict_unorm, '-r', label='network')
plt.xlabel('day')
plt.ylabel('Price')
plt.title('Comparison between Network and Actual for Test set')
plt.legend()
plt.show()