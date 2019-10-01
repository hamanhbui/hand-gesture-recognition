import os
import os.path as path
import pandas as pd
import numpy as np
from scipy import stats
def read_data_sets(file_path):
    column_names = ['timestamp','x-axis', 'y-axis', 'z-axis','x1-axis', 'y1-axis', 'z1-axis','x2-axis', 'y2-axis', 'z2-axis','activity']
    data = pd.read_csv(file_path,header = None, names = column_names)
    return data

def feature_normalize(dataset):
    mu = np.mean(dataset,axis = 0)
    sigma = np.std(dataset,axis = 0)
    return mu,sigma

def windows(data, size):
    start = 0
    while start < data.count():
        yield int(start), int(start + size)
        start += (size/2)
        
def segment_signal(data,window_size,num_channels):
    segments = np.empty((0,window_size,num_channels))
    labels = np.empty((0))
    for (start, end) in windows(data["timestamp"], window_size):
        x = data["x-axis"][start:end]
        y = data["y-axis"][start:end]
        z = data["z-axis"][start:end]
        x1 = data["x1-axis"][start:end]
        y1 = data["y1-axis"][start:end]
        z1 = data["z1-axis"][start:end]
        x2 = data["x2-axis"][start:end]
        y2 = data["y2-axis"][start:end]
        z2 = data["z2-axis"][start:end]
        if(len(data["timestamp"][start:end]) == window_size):
            # segments = np.vstack([segments,np.dstack([x,y,z,x1,y1,z1])])
            segments = np.vstack([segments,np.dstack([x,y,z,x1,y1,z1,x2,y2,z2])])
            labels = np.append(labels,stats.mode(data["activity"][start:end])[0][0])

    return segments, labels    

def preprocess_data(dataset,valset,testset,input_height,input_width,num_channels):
    dataset.dropna(axis=0, how='any', inplace= True)
    testset.dropna(axis=0, how='any', inplace= True)
    valset.dropna(axis=0,how='any',inplace=True)

    xmu,xsigma=feature_normalize(dataset['x-axis'])
    dataset['x-axis'] = (dataset['x-axis']-xmu)/xsigma
    valset['x-axis'] = (valset['x-axis']-xmu)/xsigma
    testset['x-axis']= (testset['x-axis']-xmu)/xsigma
    ymu,ysigma=feature_normalize(dataset['y-axis'])
    dataset['y-axis'] = (dataset['y-axis']-ymu)/ysigma
    valset['y-axis'] = (valset['y-axis']-ymu)/ysigma
    testset['y-axis']= (testset['y-axis']-ymu)/ysigma
    zmu,zsigma=feature_normalize(dataset['z-axis'])
    dataset['z-axis'] = (dataset['z-axis']-zmu)/zsigma
    valset['z-axis']= (valset['z-axis']-zmu)/zsigma
    testset['z-axis']= (testset['z-axis']-zmu)/zsigma

    xmu1,xsigma1=feature_normalize(dataset['x1-axis'])
    dataset['x1-axis'] = (dataset['x1-axis']-xmu1)/xsigma1
    valset['x1-axis']= (valset['x1-axis']-xmu1)/xsigma1
    testset['x1-axis']= (testset['x1-axis']-xmu1)/xsigma1
    ymu1,ysigma1=feature_normalize(dataset['y1-axis'])
    dataset['y1-axis'] = (dataset['y1-axis']-ymu1)/ysigma1
    valset['y1-axis'] = (valset['y1-axis']-ymu1)/ysigma1
    testset['y1-axis']= (testset['y1-axis']-ymu1)/ysigma1
    zmu1,zsigma1=feature_normalize(dataset['z1-axis'])
    dataset['z1-axis'] = (dataset['z1-axis']-zmu1)/zsigma1
    valset['z1-axis']= (valset['z1-axis']-zmu1)/zsigma1
    testset['z1-axis']= (testset['z1-axis']-zmu1)/zsigma1

    xmu2,xsigma2=feature_normalize(dataset['x2-axis'])
    dataset['x2-axis'] = (dataset['x2-axis']-xmu2)/xsigma2
    valset['x2-axis']= (valset['x2-axis']-xmu2)/xsigma2
    testset['x2-axis']= (testset['x2-axis']-xmu2)/xsigma2
    ymu2,ysigma2=feature_normalize(dataset['y2-axis'])
    dataset['y2-axis'] = (dataset['y2-axis']-ymu2)/ysigma2
    valset['y2-axis'] = (valset['y2-axis']-ymu2)/ysigma2
    testset['y2-axis']= (testset['y2-axis']-ymu2)/ysigma2
    zmu2,zsigma2=feature_normalize(dataset['z2-axis'])
    dataset['z2-axis'] = (dataset['z2-axis']-zmu2)/zsigma2
    valset['z2-axis']= (valset['z2-axis']-zmu2)/zsigma2
    testset['z2-axis']= (testset['z2-axis']-zmu2)/zsigma2

    print(xmu,' ',xsigma)
    print(ymu,' ',ysigma)
    print(zmu,' ',zsigma)
    print(xmu1,' ',xsigma1)
    print(ymu1,' ',ysigma1)
    print(zmu1,' ',zsigma1)
    print(xmu2,' ',xsigma2)
    print(ymu2,' ',ysigma2)
    print(zmu2,' ',zsigma2)
    
    train_x, train_y = segment_signal(data=dataset,window_size=input_width,num_channels=num_channels)
    val_x,val_y=segment_signal(data=valset,window_size=input_width,num_channels=num_channels)
    test_x,test_y=segment_signal(data=testset,window_size=input_width,num_channels=num_channels)
    
    train_y = np.asarray(pd.get_dummies(train_y), dtype = np.int8)
    val_y=np.asarray(pd.get_dummies(val_y), dtype = np.int8)
    test_y=np.asarray(pd.get_dummies(test_y), dtype = np.int8)
    train_x = train_x.reshape(len(train_x), input_width, num_channels, input_height)
    val_x = val_x.reshape(len(val_x), input_width, num_channels, input_height)
    test_x = test_x.reshape(len(test_x), input_width, num_channels, input_height)

    return train_x,train_y,val_x,val_y,test_x,test_y

input_height = 1
window_size = 50
num_channels = 9
# num_channels = 6

#These files are copied by all of the raw data in train_set, valid_set and test_set respectively.
dataset = read_data_sets(file_path='Datasets/TotalSet/trainSet.txt')
valset= read_data_sets(file_path='Datasets/TotalSet/validationSet.txt')
testset= read_data_sets(file_path='Datasets/TotalSet/testSet.txt')
x_train, y_train, x_val, y_val,x_test,y_test=preprocess_data(dataset=dataset,valset=valset,testset=testset,input_height=input_height,input_width=window_size,num_channels=num_channels)

idx = np.random.permutation(len(x_train))
x_train,y_train = x_train[idx], y_train[idx]

idx = np.random.permutation(len(x_test))
x_test,y_test = x_test[idx], y_test[idx]

np.save('Datasets/OutSet_AddMagnetic/x_train',x_train)
np.save('Datasets/OutSet_AddMagnetic/y_train',y_train)
np.save('Datasets/OutSet_AddMagnetic/x_valid',x_val)
np.save('Datasets/OutSet_AddMagnetic/y_valid',y_val)
np.save('Datasets/OutSet_AddMagnetic/x_test',x_test)
np.save('Datasets/OutSet_AddMagnetic/y_test',y_test)