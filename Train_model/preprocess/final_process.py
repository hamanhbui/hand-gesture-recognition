import numpy as np
import pandas as pd

def main():
    train_x=np.load("Dataset/out/train_set/norm_x.npy")
    valid_x=np.load("Dataset/out/valid_set/norm_x.npy")
    test_x=np.load("Dataset/out/test_set/norm_x.npy")
    train_y=np.load("Dataset/out/train_set/y.npy")
    valid_y=np.load("Dataset/out/valid_set/y.npy")
    test_y=np.load("Dataset/out/test_set/y.npy")

    # unique, counts = np.unique(test_y, return_counts=True)
    # print(dict(zip(unique, counts)))
    
    train_y = np.asarray(pd.get_dummies(train_y), dtype = np.int8)
    valid_y=np.asarray(pd.get_dummies(valid_y), dtype = np.int8)
    test_y=np.asarray(pd.get_dummies(test_y), dtype = np.int8)

    train_x = train_x.reshape(len(train_x), 74, 9, 1)
    valid_x = valid_x.reshape(len(valid_x), 74, 9, 1)
    test_x = test_x.reshape(len(test_x), 74, 9, 1)

    idx = np.random.permutation(len(train_x))
    train_x,train_y = train_x[idx], train_y[idx]
    idx = np.random.permutation(len(valid_x))
    valid_x,valid_y = valid_x[idx], valid_y[idx]
    idx = np.random.permutation(len(test_x))
    test_x,test_y = test_x[idx], test_y[idx]

    np.save('Dataset/train_x',train_x)
    np.save('Dataset/train_y',train_y)
    np.save('Dataset/valid_x',valid_x)
    np.save('Dataset/valid_y',valid_y)
    np.save('Dataset/test_x',test_x)
    np.save('Dataset/test_y',test_y)

main()