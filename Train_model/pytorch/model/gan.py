import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sklearn as sk
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from radam import RAdam
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

# Generator Code

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( 100, 64 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(True),
            # state size. (64*8) x 4 x 4
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True),
            # state size. (64*4) x 8 x 8
            nn.ConvTranspose2d( 64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True),
            # state size. (64*2) x 16 x 16
            nn.ConvTranspose2d( 64 * 2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size. (64) x 32 x 32
            nn.ConvTranspose2d( 64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.rnn=nn.LSTM(input_size=6, hidden_size=128, num_layers=1,dropout=0.5)
        self.sigmoid=nn.Sigmoid()
        self.fc3=nn.Linear(in_features=128, out_features=1)

    def forward(self, x):
        x=x.permute(2,0,1,3).contiguous()
        x = x.view(74,-1,6)
        x,_ = self.rnn(x)
        x=self.fc3(x[-1])
        x=self.sigmoid(x)
        return x

def train(train_x,train_y,gen_model,dis_model,criterion,optimizerD,optimizerG):
    G_losses = []
    D_losses = []
    iterations=0
    for batch_x,batch_y in iterate_minibatches(train_x, train_y, batchsize=64):
        inputs = batch_x
        labels = torch.full((64,), 1)
        
        CUDA = torch.cuda.is_available()
        if CUDA:
            inputs = inputs.cuda()
            labels = labels.cuda()

        dis_model.zero_grad()  
        outputs = dis_model(inputs)
        errD_real=criterion(outputs, labels)  
        errD_real.backward()                
        D_x = outputs.mean().item()
        
        noise = torch.randn(64, 100, 1, 1)
        fake=gen_model(noise)
        label.fill_(0)

        output=dis_model(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()

        errD = errD_real + errD_fake
        optimizerD.step()

        gen_model.zero_grad()
        label.fill_(1)
        output = dis_model(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        G_losses.append(errG.item())
        D_losses.append(errD.item())

        iterations+=1

    return G_losses,D_losses
    
def main():
    num_epochs=100
    train_x,train_y,valid_x,valid_y,test_x,test_y=load_data()
    gen_model=Generator()
    dis_model=Discriminator()

    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, 100, 1, 1)
    real_label = 1
    fake_label = 0

    # Setup Adam optimizers for both G and D
    optimizerD = RAdam(dis_model.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerG = RAdam(gen_model.parameters(), lr=0.0002, betas=(0.5, 0.999))

    CUDA = torch.cuda.is_available()
    if CUDA:
        gen_model = gen_model.cuda()
        dis_model=dis_model.cuda()  
    
    for epoch in range(num_epochs): 
        G_losses,D_losses=train(train_x,train_y,gen_model,dis_model,criterion,optimizerD,optimizerG)
        f = plt.figure(figsize=(10, 10))
        plt.plot(G_losses, label='G_losses')
        plt.plot(D_losses, label='D_losses')
        plt.legend()
        plt.show()

main()