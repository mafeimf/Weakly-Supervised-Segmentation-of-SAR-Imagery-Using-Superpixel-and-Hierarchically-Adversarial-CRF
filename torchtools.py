# -*- coding: utf-8 -*-
import numpy as np
import torch as t
import torch.utils.data as data
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


class MyDataset(data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __getitem__(self, index):#返回的是tensor
        img, target = self.images[index], self.labels[index]
        return img, target

    def __len__(self):
        return len(self.images)

    
def get_dataloader(X,Y,batch_size=20,shuffle=True):
    Y = np.array(Y, dtype = 'int')
    dataset = MyDataset(X, Y)
    loader = data.DataLoader(dataset, batch_size= batch_size, shuffle=shuffle)
    return loader


def testCNN(test_loader,model):
    model.eval()
    acc_count = 0
    for ii, data in enumerate(test_loader):
        test,target = data
        inputs = Variable(test)
        label = Variable(target)
        inputs = inputs.cuda()
        output=model(inputs)[0]
        acc_count += np.count_nonzero(np.argmax(output.data,axis=1)==target)
    
    return (acc_count+0.0)/test_loader.dataset.labels.size


def train_CNN(train_loader, test_loader, model, optimizer, criterion, max_epoch, test=False):
    # begin training
    model.cuda()
    print('begin training, be patient...')
    one=t.FloatTensor([1])
    one = one.cuda()
    train_loss=[]
    acc=[]

    for epoch in range(max_epoch):
        model.train()
        for ii, data in enumerate(train_loader):
            real,target = data

            inputs = Variable(real)
            label = Variable(target)

            inputs = inputs.cuda()
            label = label.cuda()

            output=model(inputs)[0]
            error_d = criterion(output.squeeze(),label)
            optimizer.zero_grad()
            error_d.backward(one)
            optimizer.step()
        if(test):
            test_acc = testCNN(test_loader,model)
            train_loss.append(error_d.data[0])
            acc.append(test_acc)
            print('Train Epoch:{}\tLoss:{:.6f},\tacc:{:.6f}'.format(epoch,error_d.data[0],test_acc))
        else:
            print('Train Epoch:{}\tLoss:{:.6f}'.format(epoch,error_d.data[0]))
            train_loss.append(error_d.data[0])
    return {'train_loss':train_loss,'test_acc':acc}


def evaluateCNN(model, test_loader, target_names):
    model.eval()
    y_true=np.array([])
    y_pred=np.array([])
    for ii, data in enumerate(test_loader):
        test,target = data
        inputs = Variable(test)
        label = Variable(target)
        inputs = inputs.cuda()
        output=model(inputs)
        y_true = np.hstack((y_true,target))
        y_pred = np.hstack((y_pred,np.argmax(output.data,axis=1)))
        
    print(classification_report(y_true, y_pred, target_names=target_names))
    print(confusion_matrix(y_true, y_pred))

    
def PredictCNN(model, test_loader):
    model.eval()
    y_true=np.array([])
    y_pred=np.array([])
    for ii, data in enumerate(test_loader):
        test,target = data
        inputs = Variable(test)
        inputs = inputs.cuda()
        output=model(inputs)
        y_pred = np.hstack((y_pred,np.argmax(output.data,axis=1)))
        
    return y_pred
