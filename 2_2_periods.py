# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 15:52:28 2019

@author: RUC
"""



# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 10:37:43 2019

@author: RUC
"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
import numpy as np
import pickle
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
#input_size = 14*14
#hidden_size = 500
num_classes = 10
num_epochs = 1000
batch_size = 100
learning_rate = 0.001
best_trainacc = 0
best_testacc =0
run_name = 'periodicity'

#Accuracy = []
# MNIST dataset 
train_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', 
                                          train=False, 
                                          transform=transforms.ToTensor())
# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=200, 
                                          shuffle=False)



def transfrom(images, ans='none'):
    images = torch.squeeze(images, dim=1) #减少一维

    [batch,h,w] = images.shape
    k = int(h/2)
    g = int(h*w/4)  #有多少个2*2
    c = torch.zeros([1, 10, g*batch]).to(device)

    if ans=='top_bottom':
        # 上下两端设置为 0
        zeros = torch.tensor(np.zeros((batch, 1, w))).to(device)
        images = torch.cat((zeros, images[:, 1:-1, :], zeros), dim=1)

    if ans=='left_right':
        zeros = torch.tensor(np.zeros((batch, h, 1))).to(device)
        images = torch.cat((zeros, images[:, :, 1:-1], zeros), dim=2)

    if ans=='all':
        zeros = torch.tensor(np.zeros((batch, 1, w))).to(device)
        images = torch.cat((zeros, images[:, 1:-1, :], zeros), dim=1)
        zeros = torch.tensor(np.zeros((batch, h, 1))).to(device)
        images = torch.cat((zeros, images[:, :, 1:-1], zeros), dim=2)

    if ans=='none':
        pass
    # print(images.shape)
    b = images.reshape([batch,k,2,k,2]).permute(2,4,1,3,0).reshape(2,2,g*batch)   #变换形式2*2*n
    c[0][0] = b[0][0]*b[0][0]
    c[0][1] = b[0][1]*b[0][1]
    c[0][2] = b[1][0]*b[1][0]
    c[0][3] = b[1][1]*b[1][1]
    c[0][4] = b[0][0]*b[0][1]
    c[0][5] = b[0][0]*b[1][0]
    c[0][6] = b[0][0]*b[1][1]
    c[0][7] = b[0][1]*b[1][0]
    c[0][8] = b[0][1]*b[1][1]
    c[0][9] = b[1][0]*b[1][1]
    a = c.reshape(1, 10, k, k, batch).permute(4, 2, 0, 3, 1).reshape(batch, 1*k, 10*k) #输出所有二次项
    a = torch.unsqueeze(a, dim=1)
    return a

def trans_all_rg(x): 
    x = torch.squeeze(x, dim=1) #减少一维
    [batch,h,w] = x.shape
    a = x.reshape([batch,h*h,1])
    b = x.reshape([batch,1,h*h])
    c = torch.matmul(a,b)  #
    c = torch.triu(c)  #取上三角
    c = torch.unsqueeze(c, dim=1)
    return c

def trans(x):
    x = x.view(x.size(0),-1)
    return x
 
def fig():
    with open(f'./metrics/train_epoch_{run_name}', 'rb') as file:
        his_dict = pickle.load(file)

    # 画出监测值图像
    train_acc_epoch = his_dict['train_acc']
    test_acc_epoch = his_dict['test_acc']
    loss_epoch = his_dict['loss']

    # val_loss_values = his_dict['val_loss']
    epochs = range(1, len(train_acc_epoch) + 1)

    plt.clf()
    plt.plot(epochs, loss_epoch, 'bo', label='Training loss')
    # plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plt.title(f'Loss {loss_epoch[-1]:.4f}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'./metrics/loss_epoch_{run_name}.png')
    plt.show()

    plt.clf()
    plt.plot(epochs, train_acc_epoch, 'bo', label='Training acc')
    plt.plot(epochs, test_acc_epoch, 'r', label='Test acc')
    plt.title(f'train_acc={train_acc_epoch[-1]:.4f}_val_acc={test_acc_epoch[-1]:.4f}')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.legend()
    plt.savefig(f'./metrics/acc_epoch_{run_name}.png')


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()   #声明可学习参数等
        self.l1_w1 = nn.Parameter(nn.init.xavier_normal_(torch.randn([1, 28, 28]))) #1_x1  一次项参数
        self.l1_w2_1 = nn.Parameter(nn.init.xavier_normal_(torch.randn([1, 1*14, 10*14]))) #1_x2  二次项参数
        self.l1_w2_2 = nn.Parameter(nn.init.xavier_normal_(torch.randn([1, 1 * 14, 10 * 14])))  # 1_x2  二次项参数
        self.l1_w2_3 = nn.Parameter(nn.init.xavier_normal_(torch.randn([1, 1 * 14, 10 * 14])))  # 1_x2  二次项参数
        self.l1_w2_4 = nn.Parameter(nn.init.xavier_normal_(torch.randn([1, 1 * 14, 10 * 14])))  # 1_x2  二次项参数
        self.l1_w3 = nn.Parameter(nn.init.xavier_normal_(torch.randn([1, 14, 14])))   #常数项参数
        
        self.l2_w1 = nn.Parameter(nn.init.xavier_normal_(torch.randn([1, 14, 14]))) #2_x1
        self.l2_w2 = nn.Parameter(nn.init.xavier_normal_(torch.randn([1, 1*7, 10*7]))) #2_x2
        #self.l2_w3 = nn.Parameter(nn.init.xavier_normal_(torch.randn([1, 28, 28])))  #1_x1
        self.l2_w4 = nn.Parameter(nn.init.xavier_normal_(torch.randn([1, 7, 7])))
    
        self.l3_w1 = nn.Linear(7*7,10)  #3_x1
        self.l3_w2 = nn.Linear(49*49,10)  #3_x2
        # self.l3_w3 = nn.Linear(28*28,10)  #1 _x1
        # self.l3_w4 = nn.Linear(14*10*14,10) #1_x2
        # self.l3_w5 = nn.Linear(14*14,10) #2_x1
        # self.l3_w6 = nn.Linear(7*10*7,10) #2_x2
        
        self.pool_2_2 = nn.AvgPool2d((2, 2), stride=(2, 2))
        self.pool_4_4 = nn.AvgPool2d((4, 4), stride=(4, 4))
        self.pool_1_10 = nn.AvgPool2d((1, 10), stride=(1, 10))

        self.bn = nn.BatchNorm2d(1)
    
    def forward(self, x): #[100,28,28]   #构造神经网络
        # 1st rg layer
        x1 = x
        l1_x1 = torch.mul(x, self.l1_w1)
        l1_x1 = 4*self.pool_2_2(l1_x1)

        l1_x2_1 = transfrom(x, ans='left_right')
        l1_x2_1 = torch.mul(l1_x2_1, self.l1_w2_1)
        l1_x2_1 = 10*self.pool_1_10(l1_x2_1)

        l1_x2_2 = transfrom(x, ans='top_bottom')
        l1_x2_2 = torch.mul(l1_x2_2, self.l1_w2_2)
        l1_x2_2 = 10 * self.pool_1_10(l1_x2_2)

        l1_x2_3 = transfrom(x, ans='all')
        l1_x2_3 = torch.mul(l1_x2_3, self.l1_w2_3)
        l1_x2_3 = 10 * self.pool_1_10(l1_x2_3)

        l1_x2_4 = transfrom(x, ans='none')
        l1_x2_4 = torch.mul(l1_x2_4, self.l1_w2_4)
        l1_x2_4 = 10 * self.pool_1_10(l1_x2_4)

        l1_out = l1_x1 + l1_x2_1 + l1_x2_2 + l1_x2_3 + l1_x2_4 + self.l1_w3
        x = self.bn(l1_out)
        # 2cd rg layer
        x3 = x
        l2_x1 = torch.mul(x, self.l2_w1)
        l2_x1 = 4*self.pool_2_2(l2_x1) 
        l2_x2 = transfrom(x)
        x4 = l2_x2
        l2_x2 = torch.mul(l2_x2, self.l2_w2)
        l2_x2 = 10*self.pool_1_10(l2_x2)
        # l2_x3 = torch.mul(x1, self.l2_w3)
        # l2_x3 = 16*self.pool_4_4(l2_x3)
        l2_out = l2_x1 + l2_x2 +self.l2_w4
        x = self.bn(l2_out)
        # 3rd rg_all layer
        x5 = x
        l3_x1 = self.l3_w1(trans(x5))
        l3_x2 = trans_all_rg(x)
        l3_x2 = self.l3_w2(trans(l3_x2))
        # l3_x3 = self.l3_w3(trans(x1))
        # l3_x4 = self.l3_w4(trans(x2))
        # l3_x5 = self.l3_w5(trans(x3))
        # l3_x6 = self.l3_w6(trans(x4))
        x = l3_x1 + l3_x2
        return x


start = time.time()
model = NeuralNet().to(device)
#print(model)
for i in model.parameters():
    print(i.size())
total_num = sum(p.numel() for p in model.parameters())
print(total_num)
# Loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
#criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5) 
 
train_epoch = {}
train_acc_epoch = []
test_acc_epoch = []
loss_epoch = []

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for step, (images, labels) in enumerate(train_loader):  
        # Move tensors to the configured device
        images = images.reshape(-1, 28, 28).to(device)
#        pad_dims = (2,2,2,2)
#        images=F.pad(images,pad_dims,"constant") #填充到[32,32]
        images = torch.unsqueeze(images, dim=1)
#        labels = labels.type(torch.FloatTensor).to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
#        outputs = model(images).reshape([batch_size])
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
#       
        
#        if (step+1) % 100 == 0:
#    print ('Epoch [{}/{}],  Loss: {:.4f}' 
#               .format(epoch+1, num_epochs,  loss.item()))


    with torch.no_grad():
        train_acc = 0
        test_acc = 0
        train_correct = 0
        test_correct = 0
        train_total = 0
        test_total = 0
        for images, labels in train_loader:
            images = images.reshape(-1, 28, 28).to(device)
            images = torch.unsqueeze(images, dim=1) 
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            train_acc = 100*train_correct/train_total
            
        for images, labels in test_loader:
            images = images.reshape(-1, 28, 28).to(device)
            images = torch.unsqueeze(images, dim=1) 
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            test_acc = 100*test_correct/test_total
#        Accuracy.append(test_acc)
        if best_trainacc < train_acc:
            best_trainacc = train_acc
        if best_testacc < test_acc:
            best_testacc = test_acc
            
        print('epoch={},   loss_value={:.4f},   train_acc={:.2f} %,   test_acc = {:.2f} %'
              .format(epoch+1, loss.item(),train_acc, test_acc))
        loss_epoch.append(loss.item())
        train_acc_epoch.append(train_acc)
        test_acc_epoch.append(test_acc)

    train_epoch['train_acc'] = train_acc_epoch
    train_epoch['test_acc'] = test_acc_epoch
    train_epoch['loss'] = loss_epoch

    with open(f'./metrics/train_epoch_{run_name}', 'wb') as file:
        pickle.dump(train_epoch, file)

    fig()

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
print('best_trainacc is: ',best_trainacc)
print('best_testacc is : ',best_testacc)
print('time span:', time.time() - start)
#plt.figure()
#plt.plot(Accuracy)
#plt.show()


