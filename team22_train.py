import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import os.path
import pickle


# mode regarding to Unique Methods
mode = 0
while (mode < 1 or mode > 4):
    mode = input("##########\nmode 1: with all Unique Methods(normalization, Adam optimizer)\nmode 2: with only Adam optimizer\nmode 3: with only normalization, optimizer is Adagrad\nmode 4: without all Unique Methods, optimizer is Adagrad\n##########\nchoose mode: ")
    mode = int(mode)
    if mode == 1:
        do_normalization = True
        is_adam = True
    elif mode == 2:
        do_normalization = False
        is_adam = True
    elif mode == 3:
        do_normalization = True
        is_adam = False
    elif mode == 4:
        do_normalization = False
        is_adam = False
    else:
        print("wrong answer. choose 1~4")
        mode = 0

### you can change dataset file here
trainfile = 'data_train.csv'
trainfile_processed = trainfile[:-4]+'_processed.csv'


class MyDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, X, y):
        self.data = X
        self.target = y
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        return x, y

    def __len__(self):
        return len(self.data)

# date's year to float
def date_to_numeric(dat):
    return float(dat[0:4])

# if preprocessed file exist, just read
if os.path.isfile(trainfile_processed):
    dataset = pd.read_csv(trainfile_processed, header=None)
    
# if not, preprocess col 0
else:
    print('preprocessing...')
    dataset = pd.read_csv(trainfile, header=None)
    N = dataset.shape[0]
    a=dataset[0].tolist()
    for i in range(N):
        if type(a[i]) == str: 
            a[i] = date_to_numeric(a[i])
    dataset[0] = a
    dataset.to_csv(trainfile_processed, mode='w', header=False, index=False)
    
# pick column number. we removed col 18 because there's built year in col 19.
pick = list(range(24))
pick.remove(18)
dataset = dataset[pick]

num_inputs = len(pick) - 1

# neural net
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_inputs,100)
        self.fc2 = nn.Linear(100,100)
        self.fc3 = nn.Linear(100,100)
        self.fc4 = nn.Linear(100,1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# fill nan
means = dataset.mean()
for i in pick:
    dataset[i].fillna(means[i], inplace=True)

N = dataset.shape[0]
ranmat = np.random.uniform(0,1,N)
train = ranmat < 0.9
test = ranmat >= 0.9

train_set = dataset[:][train]
train_set = np.array(train_set)
train_inputs = train_set[:,:num_inputs]
train_prices = train_set[:,num_inputs].reshape([-1,1])
# price log
train_prices = np.log(train_prices)

test_set = dataset[:][test]
test_set = np.array(test_set)
test_inputs = test_set[:,:num_inputs]
test_prices = test_set[:,num_inputs].reshape([-1,1])
# price log
test_prices = np.log(test_prices)

if do_normalization:
    # scaling inputs
    input_scaler = StandardScaler()
    input_scaler.fit(train_inputs)
    train_inputs = input_scaler.transform(train_inputs)
    test_inputs = input_scaler.transform(test_inputs)
    
    # scaling prices
    price_scaler = StandardScaler()
    price_scaler.fit(train_prices)
    train_prices = price_scaler.transform(train_prices)
    test_prices = price_scaler.transform(test_prices)

# numpy array to torch tensor
train_inputs = torch.from_numpy(train_inputs).type(torch.float32)
test_inputs = torch.from_numpy(test_inputs).type(torch.float32)
train_prices = torch.from_numpy(train_prices).type(torch.float32)
test_prices = torch.from_numpy(test_prices).type(torch.float32)

train_set = MyDataset(train_inputs, train_prices)
test_set = MyDataset(test_inputs, test_prices)

trainloader = torch.utils.data.DataLoader(train_set, batch_size=256,
                                          shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                         shuffle=False, num_workers=0)

# loss function
criterion = nn.MSELoss()

net = Net()
for epoch in range(20):  # loop over the dataset multiple times
    # learning rate decay
    if epoch == 0:
        lr = 1e-2
    else:
        lr /= 1.3
    
    # optimizer
    if is_adam:
        optimizer = optim.Adam(net.parameters(), lr=lr)
    else:
        optimizer = optim.Adagrad(net.parameters(), lr=lr)
    
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        labels = labels.view(-1,1)
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()      

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # print every 200 mini-batches
            print('[epoch %d, batch %5d] loss: %.6f' %(epoch + 1, i + 1,
                    running_loss / 1000))
            running_loss = 0.0

print('Finished Training')

# validation loss - original loss
with torch.no_grad():
    total_loss = 0
    
    for i, data in enumerate(testloader, 0):
        # get the inputs
        inputs, labels = data
        labels = labels.view(-1,1)

        # predict
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        
        # print validation set loss
        total_loss += loss.item()
        if i % 1000 == 999:    # print every 1000 mini-batches
#                print("inputs is")
#                print(inputs)
#                print("outputs is")
#                print(outputs.view(1,-1))
#                print("labels is")
#                print(labels.view(1,-1))
#                print("loss is")
#                print(loss.item())
            pass
            
    print('validation loss: %.6f' %(total_loss / (i+1)))

def percent_loss(predicted, actual):
    return abs((predicted-actual)/actual)

# validation loss - percentage loss
with torch.no_grad():
    total_loss = 0
    
    for i, data in enumerate(testloader, 0):
        # get the inputs
        inputs, labels = data
        labels = labels.view(-1)

        # predict
        outputs = net(inputs).view(-1)
        
        if do_normalization:
            # inverse scaling
            real_outputs = np.exp(price_scaler.inverse_transform(outputs.detach().numpy()))[0]
            real_labels = np.exp(price_scaler.inverse_transform(labels.detach().numpy()))[0]
        else:
            real_outputs = np.exp(outputs.detach().numpy())[0]
            real_labels = np.exp(labels.detach().numpy())[0]

        # percentage loss
        loss = percent_loss(real_outputs, real_labels)
        
        # print validation set loss
        total_loss += loss
        if i % 1000 == 999:    # print every 1000 mini-batches
#                print(i)
#                print("inputs is")
#                print(inputs)
            print("outputs is")
            print(real_outputs)
            print("labels is")
            print(real_labels)
#                print("loss is")
#                print(loss.item())
            pass
            
    print('percentage validation loss: %.6f' %(total_loss / (i+1)))

# save neural net, scalers
nn_file = 'team22_nn_' + str(mode) + '.sav'
mean_file = 'means_' + str(mode) + '.sav'
torch.save(net.state_dict(), nn_file)
pickle.dump(means, open(mean_file, 'wb'))
if do_normalization:
    input_scalerfile = 'input_scaler_' + str(mode) + '.sav'
    price_scalerfile = 'price_scaler_' + str(mode) + '.sav'
    pickle.dump(input_scaler, open(input_scalerfile, 'wb'))
    pickle.dump(price_scaler, open(price_scalerfile, 'wb'))
