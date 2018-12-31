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
import time


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

start_time = time.time()
    
### you can change dataset file here
testfile = 'data_test.csv'
testfile_processed = testfile[:-4]+'_processed.csv'


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

    
# preprocess col 0
print('preprocessing...')
dataset_test = pd.read_csv(testfile, header=None)
N = dataset_test.shape[0]
a=dataset_test[0].tolist()
for i in range(N):
    if type(a[i]) == str: 
        a[i] = date_to_numeric(a[i])
dataset_test[0] = a
dataset_test.to_csv(testfile_processed, mode='w', header=False, index=False)

# pick column number. we removed col 18 because there's built year in col 19.
pick = list(range(23))
pick.remove(18)
dataset_test = dataset_test[pick]

num_inputs = len(pick)

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

# load neural net, scalers
nn_file = 'team22_nn_' + str(mode) + '.sav'
mean_file = 'means_' + str(mode) + '.sav'
net = Net()
net.load_state_dict(torch.load(nn_file))
net.eval()
means = pickle.load(open(mean_file, 'rb'))
if do_normalization:
    input_scalerfile = 'input_scaler_' + str(mode) + '.sav'
    price_scalerfile = 'price_scaler_' + str(mode) + '.sav'
    input_scaler = pickle.load(open(input_scalerfile, 'rb'))
    price_scaler = pickle.load(open(price_scalerfile, 'rb'))

# fill nan
for i in pick:
    dataset_test[i].fillna(means[i], inplace=True)

# test set
test_set = np.array(dataset_test)
test_inputs = test_set
test_prices = np.zeros(test_inputs.shape[0]).reshape([-1,1])

if do_normalization:
    # scaling inputs
    test_inputs = input_scaler.transform(test_inputs)

# numpy array to torch tensor
test_inputs = torch.from_numpy(test_inputs).type(torch.float32)
test_prices = torch.from_numpy(test_prices).type(torch.float32)

test_set = MyDataset(test_inputs, test_prices)
testloader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                         shuffle=False, num_workers=0)

with torch.no_grad():
    predict=[]
    
    for i, data in enumerate(testloader, 0):
        # get the inputs
        inputs, labels = data
        labels = labels.view(-1)

        # predict
        outputs = net(inputs)
        
        if do_normalization:
            # inverse scaling
            real_outputs = np.exp(price_scaler.inverse_transform(outputs.detach().numpy()))[0]
        else:
            real_outputs = np.exp(outputs.detach().numpy())[0]
        
        predict.append(int(real_outputs))
        
    # save predict.csv
    predict = np.array(predict)
    predict_file = "predict_" + str(mode) + ".csv"
    np.savetxt(predict_file, predict, delimiter = "\n")
    print('result saved. "{}"'.format(predict_file))
    
end_time = time.time()
print("execution time is {} sec".format(end_time-start_time))