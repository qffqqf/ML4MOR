import pickle
from data_set_utils import Dataset
from sklearn.model_selection import train_test_split
import rff
import numpy
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from scipy.io import savemat

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
# Parameters
params = {'batch_size': 5*1,
          'shuffle': True,
          'num_workers': 1}
max_epochs = 50
learning_rate = 2e-3
order = 30

# dataset
outfile = open('./data/training_data.p', 'rb')
training_data = pickle.load(outfile)
data_p = training_data["X_train"]
label_Tr = training_data["Y_train"]
print(data_p.shape)
X_train, X_test, y_train, y_test = train_test_split( \
                       data_p, label_Tr, test_size=0.1, random_state=42)
nDt, nParam = data_p.shape
nDt, nDOFs = label_Tr.shape


# Load validation
outfile = open('./data/ref_data.p', 'rb')
ref_data = pickle.load(outfile)
X_ref = ref_data["X_ref"]
y_ref = ref_data["Y_ref"]

# Data loader
training_set = Dataset(X_train, y_train)
training_generator = torch.utils.data.DataLoader(training_set, **params)
test_set = Dataset(X_test, y_test)
test_generator = torch.utils.data.DataLoader(test_set, **params)
validation_set = Dataset(X_ref, y_ref)
validation_generator = torch.utils.data.DataLoader(validation_set, **params)

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.lin = nn.Linear(in_channels, out_channels)
        self.gelu = nn.GELU()
        self.sigm = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        residual = x
        out = self.lin(x)
        out = self.gelu(out)
        out = self.lin(out)
        out += residual
        out = self.gelu(out)
        return out

# ResNet
class ResNet(nn.Module):
    def __init__(self, block, nInput, nOutput):
        super(ResNet, self).__init__()
        self.gelu = nn.GELU()
        self.nInput = nInput
        self.in1 = nn.Linear(nInput, 140)
        self.in2 = nn.Linear(140, 60)
        self.inr2 = self.make_layer(block, 60)
        self.in3 = nn.Linear(60, 20)
        self.in1_3 = nn.Linear(140, 20)
        self.inr3 = self.make_layer(block, 20)
        self.in4 = nn.Linear(20, 3)
        self.in1_4 = nn.Linear(140, 3)
        self.in2_4 = nn.Linear(60, 3)
        self.inr4 = self.make_layer(block, 3)

        self.outr4 = self.make_layer(block, 3)
        self.out4 = nn.Linear(3, 20)
        self.out4_2 = nn.Linear(3, 60)
        self.out4_1 = nn.Linear(3, 140)
        self.outr3 = self.make_layer(block, 20)
        self.out3 = nn.Linear(20, 60)
        self.out3_1 = nn.Linear(20, 140)
        self.outr2 = self.make_layer(block, 60)
        self.out2 = nn.Linear(60, 140)
        self.out1 = nn.Linear(140, nOutput)
        self.outr1 = self.make_layer(block, 140)
        self.output = nn.Linear(nOutput, nOutput)
        
    def make_layer(self, block, width):
        layers = []
        layers.append(block(width, width))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x1 = self.gelu(self.in1(x))
        x2 = self.gelu(self.in2(x1))
        for iRes in range(1):
            x2 = self.inr2(x2) + self.gelu(self.in2(x1))
        x3 = self.gelu(self.in3(x2))
        for iRes in range(1):
            x3 = self.inr3(x3) + self.gelu(self.in1_3(x)) + self.gelu(self.in1_3(x1))
        x4 = self.gelu(self.in4(x3))
        for iRes in range(1):
            x4 = self.inr4(x4) + self.gelu(self.in1_4(x)) + self.gelu(self.in1_4(x1)) + self.gelu(self.in2_4(x2))

        for iRes in range(1):
            y4 = self.outr4(x4)

        y3 = self.gelu(self.out4(y4))
        for iRes in range(1):
            y3 = self.outr3(y3) + self.gelu(self.out4(y4))
        y2 = self.gelu(self.out3(y3)) + self.gelu(self.out4_2(y4))
        for iRes in range(1):
            y2 = self.outr2(y2) + self.gelu(self.out3(y3)) + self.gelu(self.out4_2(y4))
        y1 = self.gelu(self.out2(y2)) + self.gelu(self.out3_1(y3)) + self.gelu(self.out4_1(y4))
        for iRes in range(1):
            y1 = self.outr1(y1) + self.gelu(self.out2(y2)) + self.gelu(self.out3_1(y3)) + self.gelu(self.out4_1(y4))
        y = self.output(y1)
        return y, y4
    
model = ResNet(ResidualBlock, 140, 140).to(device)


# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

# For updating learning rate
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# Opt
def closure():
    optimizer.zero_grad()
    out, midout = model(local_labels) 
    loss_regression = criterion(out, local_labels) #+ criterion(midout, local_batch)
    loss_regression.backward()   
    return loss_regression

# Train the model
error_all = []
curr_lr = learning_rate
for epoch in range(max_epochs):
    print("epoch = ", epoch)
    for local_batch, local_labels in training_generator:
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)
        optimizer.step(closure)   
    print("curr_lr = ", curr_lr)
    # Decay learning rate
    if (epoch+1) % 20 == 0:
        if curr_lr > 5e-5:
            curr_lr /= 1.5
            update_lr(optimizer, curr_lr)

    correct = 0
    total = 0
    sqr_err = 0
    sqr_sum = 0

    with torch.no_grad():
        for data in training_generator:
            parameters, labels = data
            parameters, labels = parameters.to(device), labels.to(device)
            outputs, midout = model(labels)
            sqr_err += torch.sum((outputs.data-labels.data)**2)
            sqr_sum += torch.sum(labels.data**2)

    print(f'Relative error (training): {torch.sqrt(sqr_err/sqr_sum)}')

    with torch.no_grad():
        for data in test_generator:
            parameters, labels = data
            parameters, labels = parameters.to(device), labels.to(device)
            outputs, midout = model(labels)
            sqr_err += torch.sum((outputs.data-labels.data)**2)
            sqr_sum += torch.sum(labels.data**2)

    print(f'Relative error (test): {torch.sqrt(sqr_err/sqr_sum)}')

    with torch.no_grad():
        for data in validation_generator:
            parameters, labels = data
            parameters, labels = parameters.to(device), labels.to(device)
            #parameters_ec = encoding(parameters, order, rand_)
            outputs, midout = model(labels)
            sqr_err += torch.sum((outputs.data-labels.data)**2)
            sqr_sum += torch.sum(labels.data**2)


    print(f'Relative error (validation): {torch.sqrt(sqr_err/sqr_sum)}')
    error_all.append(torch.sqrt(sqr_err/sqr_sum).item())   
        
max_epochs = 100
params = {'batch_size': 8,
          'shuffle': True,
          'num_workers': 1}
# Data loader
training_generator = torch.utils.data.DataLoader(training_set, **params)
test_generator = torch.utils.data.DataLoader(test_set, **params)
validation_generator = torch.utils.data.DataLoader(validation_set, **params)


for param in model.parameters():
    param.requires_grad = False



class TinyNet(nn.Module):
    def __init__(self, block):
        super(TinyNet, self).__init__()
        self.gelu = nn.GELU()
        self.res3 = self.make_layer(block, 3)
        self.res100 = self.make_layer(block, 500)
        self.res500 = self.make_layer(block, 1000)
        self.lin3_100 = nn.Linear(3, 500)
        self.lin3_500 = nn.Linear(3, 1000)
        self.lin100_500 = nn.Linear(500, 1000)
        self.lin500_100 = nn.Linear(1000, 500)
        self.lin100_3 = nn.Linear(500, 3)
        self.lin500_3 = nn.Linear(1000, 3)
        self.output = nn.Linear(3, 3)
        
    def make_layer(self, block, width):
        layers = []
        layers.append(block(width, width))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x1 = self.gelu(self.lin3_100(x))
        for iRes in range(1):
            x1 = self.res100(x1) + self.gelu(self.lin3_100(x))
        x2 = self.gelu(self.lin100_500(x1))
        for iRes in range(1):
            x2 = self.res500(x2) + self.gelu(self.lin100_500(x1)) + self.gelu(self.lin3_500(x))
        x3 = self.gelu(self.lin500_100(x2))
        for iRes in range(1):
            x3 = self.res100(x3) + self.gelu(self.lin500_100(x2)) + self.gelu(self.lin3_100(x))
        x4 = self.gelu(self.lin100_3(x3))
        for iRes in range(1):
            x4 = self.res3(x4) + self.gelu(self.lin100_3(x3)) + self.gelu(self.lin500_3(x2)) + self.res3(x)
        """
        x3 = self.res(x2)
        for iRes in range(5):
            x3 = self.res(x3) + self.res(x2) + self.res(x1) + self.res(x)
        x4 = self.res(x3)
        for iRes in range(2):
            x4 = self.res(x4) + self.res(x3) + self.res(x2) + self.res(x1) + self.res(x)
        x5 = self.res(x4)
        for iRes in range(1):
            x5 = self.res(x5) + self.res(x4) + self.res(x3) + self.res(x2) + self.res(x1) + self.res(x)
        """
        y = self.output(x4) 
        return y
    
tinynet = TinyNet(ResidualBlock).to(device)
# Loss and optimizer
learning_rate = 1e-3
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(tinynet.parameters(), lr = learning_rate)
def closure_tiny():
    optimizer.zero_grad()
    out_ae, midout = model(local_labels) 
    print(local_batch.shape)
    out = tinynet(local_batch) 
    loss_regression = criterion(out, midout)
    loss_regression.backward()   
    return loss_regression



# Train the model
error_all = []
curr_lr = learning_rate
for epoch in range(max_epochs):
    print("epoch = ", epoch)
    for local_batch, local_labels in training_generator:
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)
        optimizer.step(closure_tiny)   
    print("curr_lr = ", curr_lr)
    # Decay learning rate
    if (epoch+1) % 20 == 0:
        if curr_lr > 5e-5:
            curr_lr /= 1.5
            update_lr(optimizer, curr_lr)

    sqr_err = 0
    sqr_sum = 0

    with torch.no_grad():
        for data in training_generator:
            parameters, labels = data
            parameters, labels = parameters.to(device), labels.to(device)
            out_ae, midout = model(labels)
            out = tinynet(parameters) 
            sqr_err += torch.sum((out.data-midout.data)**2)
            sqr_sum += torch.sum(midout.data**2)

    print(f'Relative error (training): {torch.sqrt(sqr_err/sqr_sum)}')

    with torch.no_grad():
        for data in test_generator:
            parameters, labels = data
            parameters, labels = parameters.to(device), labels.to(device)
            out_ae, midout = model(labels)
            out = tinynet(parameters) 
            sqr_err += torch.sum((out.data-midout.data)**2)
            sqr_sum += torch.sum(midout.data**2)

    print(f'Relative error (test): {torch.sqrt(sqr_err/sqr_sum)}')

    with torch.no_grad():
        for data in validation_generator:
            parameters, labels = data
            parameters, labels = parameters.to(device), labels.to(device)
            out_ae, midout = model(labels)
            out = tinynet(parameters) 
            sqr_err += torch.sum((out.data-midout.data)**2)
            sqr_sum += torch.sum(midout.data**2)

    print(f'Relative error (validation): {torch.sqrt(sqr_err/sqr_sum)}')
    error_all.append(torch.sqrt(sqr_err/sqr_sum).item())  



import matplotlib.pyplot as plt
plt.figure()
plt.plot(error_all)
plt.xlabel("# Epochs")
plt.ylabel("Relative error")
plt.show()
