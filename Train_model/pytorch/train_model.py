import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sklearn as sk
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score
from radam import RAdam
# from cnn_lstm import CNN_LSTM
# from baseline_cnn import Baseline_CNN
import time
import seaborn
from sklearn.metrics import precision_recall_curve
from sklearn.utils.multiclass import unique_labels


learning_rate=0.001

class CNN_LSTM(nn.Module):

    def __init__(self):
        super(CNN_LSTM, self).__init__()
        self.relu=nn.ReLU()
        self.conv1=nn.Conv2d(in_channels=1, out_channels=64, kernel_size=[5,1],stride=1,padding=0)
        self.conv2=nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[5,1],stride=1,padding=0)
        self.conv3=nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[5,1],stride=1,padding=0)
        self.conv4=nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[5,1],stride=1,padding=0)
        self.rnn = nn.LSTM(input_size=384, hidden_size=128, num_layers=2,dropout=0.5)
        self.fc3=nn.Linear(in_features=128, out_features=22)

    def forward(self, x):
        x=self.conv1(x)
        x=self.relu(x)
        x=self.conv2(x)
        x=self.relu(x)
        x=self.conv3(x)
        x=self.relu(x)
        x=self.conv4(x)
        x=self.relu(x)
        x=x.permute(2,0,1,3).contiguous()
        x = x.view(34,-1,384)
        # h0 = torch.randn(2*1, 100, 128)
        # c0 = torch.randn(2*1, 100, 128)
        # x,_ = self.rnn(x, (h0, c0))
        x,_ = self.rnn(x)
        # x=x.contiguous()
        # x = x.view(-1,128)
        # x=self.fc3(x)
        # x = x.view(58,-1,22)
        # x=x[57]

        x=x[-1]
        x=self.fc3(x)
        return x

def adjust_learning_rate(optimizer, epoch):
    global learning_rate
    if epoch in [30,60]:
        learning_rate *= 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

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

def load_data(root='drive/My Drive/My Data/segment_2s/'):
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


    train_x=train_x.reshape(-1,50,6)
    train_x=train_x.reshape(-1,1,50,6)
    train_x=torch.from_numpy(train_x).float()
    train_y=torch.from_numpy(train_y).long()
    valid_x=valid_x.reshape(-1,50,6)
    valid_x=valid_x.reshape(-1,1,50,6)
    valid_x=torch.from_numpy(valid_x).float()
    valid_y=torch.from_numpy(valid_y).long()
    test_x=test_x.reshape(-1,50,6)
    test_x=test_x.reshape(-1,1,50,6)
    test_x=torch.from_numpy(test_x).float()
    test_y=torch.from_numpy(test_y).long()
    train_y=torch.max(train_y, 1)[1]
    valid_y=torch.max(valid_y, 1)[1]
    test_y=torch.max(test_y, 1)[1]
    return train_x,train_y,valid_x,valid_y,test_x,test_y

def train(train_x,train_y,model,criterion,optimizer):
    model.train()
    total_loss=0
    total_correct=0
    total_input=0
    iterations=0
    for batch_x,batch_y in iterate_minibatches(train_x, train_y, batchsize=128):
        inputs = batch_x
        labels = batch_y
        
        CUDA = torch.cuda.is_available()
        if CUDA:
            inputs = inputs.cuda()
            labels = labels.cuda()

        outputs = model(inputs)
        loss=criterion(outputs, labels)
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum()

        total_loss+=loss.item()
        total_correct+=correct.item()
        total_input+=inputs.size(0)

        optimizer.zero_grad()  
        loss.backward()                
        # nn.utils.clip_grad_norm(model.parameters(), 5)
        optimizer.step()

        iterations+=1

    loss=total_loss/iterations
    accuracy=(100*total_correct/total_input)
    return loss,accuracy

def valid(valid_x,valid_y,model,criterion):
    model.eval()
    total_loss=0
    total_correct=0
    total_input=0
    iterations=0
    for batch_x,batch_y in iterate_minibatches(valid_x, valid_y, batchsize=128):
        inputs = batch_x
        labels = batch_y
        
        CUDA = torch.cuda.is_available()
        if CUDA:
            inputs = inputs.cuda()
            labels = labels.cuda()
        
        outputs = model(inputs)     
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs, 1)
        correct=(predicted == labels).sum()

        total_loss+=loss.item()
        total_correct+=correct.item()
        total_input+=inputs.size(0)
        iterations+=1

    loss=total_loss/iterations
    accuracy=(100*total_correct/total_input)
    return loss,accuracy

def test(test_x,test_y,model):
    model.eval()  
    with torch.no_grad():
        total_correct = 0
        total_input=0
        for batch_x,batch_y in iterate_minibatches(test_x, test_y, batchsize=128):
            inputs = batch_x
            labels = batch_y

            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct=(predicted == labels).sum()

            total_correct+=correct.item()
            total_input+=inputs.size(0)

    accuracy=(100*total_correct/total_input)
    return accuracy 


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    
    return ax
def main():
    num_epochs=100
    train_x,train_y,valid_x,valid_y,test_x,test_y=load_data()
    model=CNN_LSTM()

    CUDA = torch.cuda.is_available()
    if CUDA:
        model = model.cuda()  
    
    criterion = nn.CrossEntropyLoss()
    optimizer = RAdam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-4,momentum=0.9)
    train_loss=[]
    train_accuracy=[]
    valid_loss=[]
    valid_accuracy=[]
    for epoch in range(num_epochs): 
        start = time.time()

        # adjust_learning_rate(optimizer,epoch)
        loss_avg,accuracy_avg = train(train_x,train_y,model,criterion,optimizer)
        valid_loss_avg,valid_acc_avg=valid(valid_x,valid_y,model,criterion)

        train_loss.append(loss_avg)      
        train_accuracy.append(accuracy_avg)
        valid_loss.append(valid_loss_avg)      
        valid_accuracy.append(valid_acc_avg)
        stop = time.time()
        print('Epoch: [%d | %d] LR: %f Train:Loss_Avg=%f Accuracy_Avg=%f Valid: Loss=%f Accuracy=%f Time: %f' 
        % (epoch + 1, num_epochs, learning_rate,loss_avg,accuracy_avg,valid_loss_avg,valid_acc_avg, stop-start))

    test_accuracy=test(test_x,test_y,model)
    print('Test Accuracy: {:.3f} %'.format(test_accuracy))

    # torch.save(model,"base_cnn")

    if torch.cuda.is_available():
        inputs = test_x.cuda()
        labels = test_y.cuda()

    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)

    correct = ((predicted == labels).sum().item())/labels.size(0)
    print('Test Accuracy: {:.3f} %'.format(100 * correct))

    print ("Precision")
    print (precision_score(predicted.cpu().numpy(), labels.cpu().numpy(),average=None))
    print (precision_score(predicted.cpu().numpy(), labels.cpu().numpy(),average='weighted'))
    print ("Recall")
    print (recall_score(predicted.cpu().numpy(), labels.cpu().numpy(),average=None))
    print (recall_score(predicted.cpu().numpy(), labels.cpu().numpy(),average='weighted'))

    print ("F1 Accuracy:", sk.metrics.f1_score(predicted.cpu().numpy(), labels.cpu().numpy(),average=None))
    print ("F1 Accuracy:", sk.metrics.f1_score(predicted.cpu().numpy(), labels.cpu().numpy(),average='weighted'))

    np.save("predicted",predicted.cpu().numpy())
    np.save("labels",labels.cpu().numpy())

    print("Confustion Matrix:")
    class_names=['Start_gesture','Unknown']
    class_names=np.array(class_names)
    plot_confusion_matrix(predicted.cpu().numpy(), labels.cpu().numpy(), classes=class_names,
                         normalize=True, title='Normalized confusion matrix')
    plt.figure(figsize=(10,10))

    plt.savefig('foo.png')
    plt.show()
    print()

    # Loss
    f = plt.figure(figsize=(10, 10))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(valid_loss, label='Valid Loss')
    plt.legend()
    plt.show()

    # Accuracy
    f = plt.figure(figsize=(10, 10))
    plt.plot(train_accuracy, label='Training Accuracy')
    plt.plot(valid_accuracy, label='Valid Accuracy')
    plt.legend()
    plt.show()

    train_loss=np.asarray(train_loss)
    train_accuracy=np.asarray(train_accuracy)
    valid_loss=np.asarray(valid_loss)
    valid_accuracy=np.asarray(valid_accuracy)
    np.save('train_loss',train_loss)
    np.save('train_accuracy',train_accuracy)
    np.save('valid_loss',valid_loss)
    np.save('valid_accuracy',valid_accuracy)

main()