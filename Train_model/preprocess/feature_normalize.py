import os
import os.path as path
import pandas as pd
import numpy as np
from scipy import stats

def standardization(dataset):
    mu = np.mean(dataset)
    sigma = np.std(dataset)
    return mu,sigma

def feature_normalize(train_set,valid_set,test_set):
    mu,sigma=standardization(train_set)
    train_set=(train_set-mu)/sigma
    valid_set=(valid_set-mu)/sigma
    test_set=(test_set-mu)/sigma
    return train_set,valid_set,test_set
    
def main():
    train_x=np.load("Dataset/out/train_set/x.npy")
    valid_x=np.load("Dataset/out/valid_set/x.npy")
    test_x=np.load("Dataset/out/test_set/x.npy")

    nomarlized_train_x=np.empty((train_x.shape[0],74,0))
    nomarlized_valid_x=np.empty((valid_x.shape[0],74,0))
    nomarlized_test_x=np.empty((test_x.shape[0],74,0))

    train_x=np.transpose(train_x,axes=[2,0,1])
    valid_x=np.transpose(valid_x,axes=[2,0,1])
    test_x=np.transpose(test_x,axes=[2,0,1])

    for i in range(train_x.shape[0]):
        train_set,valid_set,test_set=feature_normalize(train_x[i],valid_x[i],test_x[i])
        nomarlized_train_x=np.dstack((nomarlized_train_x,train_set))
        nomarlized_valid_x=np.dstack((nomarlized_valid_x,valid_set))
        nomarlized_test_x=np.dstack((nomarlized_test_x,test_set))

    train_x=np.save("Dataset/out/train_set/norm_x.npy",nomarlized_train_x)
    train_x=np.save("Dataset/out/valid_set/norm_x.npy",nomarlized_valid_x)
    train_x=np.save("Dataset/out/test_set/norm_x.npy",nomarlized_test_x)
    
main()