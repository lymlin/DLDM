from math import sqrt
from numpy import concatenate
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from keras.optimizers import Adam
import numpy as np

## convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    #dataframe
    df = DataFrame(data) 
    cols, names = list(),list()
    #list  append data
    # input sequence(t-n,... t-1)
    for i in range(n_in,0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' %(j+1,i))for j in range(n_vars)]
    #forecast sequence(t,t+1,t+2...t+n)
    for i in range(0,n_out):
        cols.append(df.shift(-i))
        if i ==0:
            names += [('var%d(t)'%(j+1))for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)'%(j+1,i))for j in range (n_vars)]
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values (which has been done by data predisposation in R - imputeTS)
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# function of standardization
def inv_MinMaxScaler(y,ma,mi,MI,MA):
    x = (y-MI)/(MA-MI)*(ma-mi)+mi
    return x

 #load and processing dataset
def main(times=1,step=1,epo=1):
    if times == 1:
        Name0 = '0xall1.csv'
    else:
        Name0 = "%d"%(times-1) + "xall%d.csv"%(epo)

    Name = "%d"%(times) + "xall%d.csv"%(epo)
    dataset = read_csv(Name0, header=0, index_col=0)
    #---- check----print (dataset)
    values = dataset.values
    # integer endode direction for labalizing data
    # encoder =  preprocessing.LabelEncoder()
    # values[:,1] = encoder.fit_transform(values[:,1])

    # ensure all data is float
    values = values.astype('float32')
    maa = max(values[:,2])
    mii = min(values[:,2])
    
    # normalize scaler
    scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
    scaled = scaler.fit_transform(values)
    
    # frame as supervised learning
    dely=step
    reframed = series_to_supervised(scaled,dely,1)
    #reframed = series_to_supervised(values,dely,1)
    
     # drop columns we do not want to predict
    for i in range(1,dely*3):
        reframed.drop(reframed.columns[3],axis=1,inplace=True)
    
    # split into train and test sets
    n_test = 576+864+864
    n_train = (len(reframed.values)-n_test) #n_test个作验证集
    values = reframed.values
    train = values[:n_train, :]
    test = values[n_train:, :]
    
    #____check____print(train)
    # split into input and out put
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:,-1]
    # reshape input to be 3D[samples,timesteps, features]
    train_X = train_X.reshape((train_X.shape[0],1,train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0],1,test_X.shape[1]))
    
    # architecture of deep learning network
    model = Sequential()
    model.add(LSTM(25,input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dropout(0.01))
    model.add(Dense(1))
    adam = Adam(lr=1*1e-04, beta_1=0.99, beta_2=0.9999, epsilon=1e-08)
    model.compile(loss='mae', optimizer=adam)
    #fit network
    history = model.fit(train_X, train_y,epochs=250, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
    #plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'],label='test')
    np.savetxt("train_loss%d.csv"%epo,history.history['loss'])
    np.savetxt("test_loss%d.csv"%epo,history.history['val_loss'])
    pyplot.legend()
    pyplot.show()
    
    ##prediction and evaluation	
    # make a prediction
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    # invert scaling for forecast
    inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
    inv_yhat = inv_MinMaxScaler(inv_yhat,maa,mii,-1,1)
    inv_yhat = inv_yhat[:,0]
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
    inv_y = inv_MinMaxScaler(inv_y,maa,mii,-1,1)
    inv_y = inv_y[:,0]
    #print out
    np.savetxt("prediction%d.csv"%(epo),inv_yhat)
    np.savetxt("realdata%d.csv"%(epo),inv_y)
    
    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    exec("RMSE_list%s.append(rmse)"%times)
    print('Test RMSE: %.3f' % rmse)
    
    #add the last timestep
    aaa = read_csv(Name0)
    for i in range(0,step):
        ppc = aaa.tail(1).values
        ppp = {'No':ppc[0,0]+1,'time':ppc[0,1]+1,'case':ppc[0,2],'glucose':inv_yhat[len(inv_yhat) -step +i]}
        aaa = aaa.append(ppp,ignore_index=True)
    
    aaa.to_csv(Name,index=False)
    model.save_weights('my_model_weights%d.h5'%(epo))
    return None

j=2
count = 10
for i in range(1,int(8/j)+1):
    exec("RMSE_list%s = []"%i)
    for k in range(1,count+1):
        main(i,j,k)
    np.savetxt("RMSE%s.csv"%i,eval("RMSE_list%s"%i))
