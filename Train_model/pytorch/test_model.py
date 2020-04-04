import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sklearn as sk
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import time

def iterate_minibatches(inputs, targets, batchsize, shuffle=True):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def load_data(root='drive/My Drive/gesture_input/Dataset/3s_NML_magne/'):
    train_x=np.load(root+'train_x.npy')
    train_y=np.load(root+'train_y.npy')
    valid_x=np.load(root+'valid_x.npy')
    valid_y=np.load(root+'valid_y.npy')
    test_x=np.load(root+'test_x.npy')
    test_y=np.load(root+'test_y.npy')

    train_x=np.delete(train_x,6,axis=2)
    train_x=np.delete(train_x,6,axis=2)
    train_x=np.delete(train_x,6,axis=2)
    
    valid_x=np.delete(valid_x,6,axis=2)
    valid_x=np.delete(valid_x,6,axis=2)
    valid_x=np.delete(valid_x,6,axis=2)
    
    test_x=np.delete(test_x,6,axis=2)
    test_x=np.delete(test_x,6,axis=2)
    test_x=np.delete(test_x,6,axis=2)

    train_x=train_x.reshape(-1,74,6)
    train_x=train_x.reshape(-1,1,74,6)
    train_x=torch.from_numpy(train_x).float()
    train_y=torch.from_numpy(train_y).long()
    valid_x=valid_x.reshape(-1,74,6)
    valid_x=valid_x.reshape(-1,1,74,6)
    valid_x=torch.from_numpy(valid_x).float()
    valid_y=torch.from_numpy(valid_y).long()
    test_x=test_x.reshape(-1,74,6)
    test_x=test_x.reshape(-1,1,74,6)
    test_x=torch.from_numpy(test_x).float()
    test_y=torch.from_numpy(test_y).long()
    train_y=torch.max(train_y, 1)[1]
    valid_y=torch.max(valid_y, 1)[1]
    test_y=torch.max(test_y, 1)[1]
    return train_x,train_y,valid_x,valid_y,test_x,test_y

def test(test_x,test_y,model_1,model_2,model_3):
    model_1.eval()  
    model_2.eval()  
    model_3.eval()
    with torch.no_grad():
        total_correct_1 = 0
        total_correct_2=0
        total_correct_3=0
        total_correct_cb=0
        total_input=0
        for batch_x,batch_y in iterate_minibatches(test_x, test_y, batchsize=128):
            inputs = batch_x
            labels = batch_y

            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            outputs_1 = model_1(inputs)
            _, predicted = torch.max(outputs_1, 1)
            correct=(predicted == labels).sum()
            total_correct_1+=correct.item()

            outputs_2=model_2(inputs)
            _, predicted = torch.max(outputs_2, 1)
            correct=(predicted == labels).sum()
            total_correct_2+=correct.item()

            outputs_3=model_3(inputs)
            _, predicted = torch.max(outputs_3, 1)
            correct=(predicted == labels).sum()
            total_correct_3+=correct.item()

            outputs_cb=outputs_1+outputs_2+outputs_3
            _, predicted = torch.max(outputs_cb, 1)
            correct=(predicted == labels).sum()
            total_correct_cb+=correct.item()

            total_input+=inputs.size(0)

    accuracy_1=(100*total_correct_1/total_input)
    accuracy_2=(100*total_correct_2/total_input)
    accuracy_3=(100*total_correct_3/total_input)
    accuracy_cb=(100*total_correct_cb/total_input)
    return accuracy_1,accuracy_2,accuracy_3,accuracy_cb


def main():
    train_x,train_y,valid_x,valid_y,test_x,test_y=load_data()
    model_cnn=torch.load("base_cnn")
    model_lstm=torch.load("lstm")
    model_cnn_lstm=torch.load("cnn_lstm")

    CUDA = torch.cuda.is_available()
    if CUDA:
        model_cnn = model_cnn.cuda()
        model_lstm=model_lstm.cuda()
        model_cnn_lstm=model_cnn_lstm.cuda()
    
    test_accuracy_cnn,test_accuracy_1stm,test_accuracy_cnn_1stm,test_accuracy_combine=test(test_x,test_y,model_cnn,model_lstm,model_cnn_lstm)
    print('Test Accuracy: {:.3f} %'.format(test_accuracy_cnn))
    print('Test Accuracy: {:.3f} %'.format(test_accuracy_1stm))
    print('Test Accuracy: {:.3f} %'.format(test_accuracy_cnn_1stm))
    print('Test Accuracy: {:.3f} %'.format(test_accuracy_combine))

main()