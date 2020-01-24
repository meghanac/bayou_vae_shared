import torch
# import torch.utils.data
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

'''

Steps:
1. Load Dataset
2. Make Dataset Iterable
3. Create Model Class
4. Instantiate Model Class
5. Instantiate Loss Class
6. Instantiate Optimizer Class
7. Train Model

'''

# # Step 1: Load Dataset
#
# import torch.nn as nn
# import torchvision.transforms as transforms
# import torchvision.datasets as dsets
# from torch.autograd import Variable
#
# train_dataset = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
# test_dataset = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
#
# print(train_dataset.train_data.size())
# print(train_dataset.train_labels.size())
# print(test_dataset.test_data.size())
# print(test_dataset.test_labels.size())
#
# # ---------------------------------------------------------------------------
#
# # Step 2: Make Dataset Iterable
#
# batch_size = 100
# n_iters = 3000
# num_epochs = n_iters / (len(train_dataset) / batch_size)
# num_epochs = int(num_epochs)
#
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
#
#
# # ---------------------------------------------------------------------------
#
# # Step 3: Create Model Class
#
# class RNNModel(nn.Module):
#     def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
#         super(RNNModel, self).__init__()
#
#         # Hidden dimensions
#         self.hidden_dim = hidden_dim
#
#         # Number of hidden layers
#         self.layer_dim = layer_dim
#
#         # Building your RNN
#         # batch_first = True caseuse input/output tensors to be of shape
#         # (batch_dim, seq_dim, input_dim)
#         self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
#
#         # Readout layer
#         self.fc = nn.Linear(hidden_dim, output_dim)
#
#     def forward(self, x):
#         # initialize hidden state with zeros
#         # (layer_dim, batch_size, hidden_dim)
#         h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
#
#         # One time step
#         out, hn = self.rnn(x, h0)
#
#         # Index hidden state of last time step
#         # out.size() --> 100, 28, 100 (28 is num of time steps)
#         # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
#         out = self.fc(out[:, -1, :])
#         # out.size() --> 100, 10
#
#
# # ---------------------------------------------------------------------------
#
# # Step 4: Instantiate Model Class
#
# '''
# 28 times steps
#     Each time step: input dimension = 28
# 1 hidden layer
# MNIST 1-9 digits -> output dimension = 10
# '''
#
# input_dim = 28
# hidden_dim = 100
# layer_dim = 1
# output_dim = 10
#
# model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim)
#
# # ---------------------------------------------------------------------------
#
# # Step 5: Instantiate Loss Class
#
# '''
# RNNs, CNNs, FNNs, Logistic Regression: Cross Entropy Loss
# Linear Regression: MSE
#
# '''
#
# criterion = nn.CrossEntropyLoss()
#
# # ---------------------------------------------------------------------------
#
# # Step 6: Instantiate Optimizer Class
#
# # parameters = parameters - learning_rate * parameters_gradients
#
# learning_rate = 0.1
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#
# # parameters in-depth
# print(len(list(model.parameters())))
#
# '''
# Parameters
#
# Input to Hidden Layer Linear Function
#     A1, B1
#
# Hidden Layer to Output Linear Function
#     A2, B2
#
# Hidden Layer to Hidden Layer Linear Function
#     A3, B3
#
# '''
#
# # Input --> Hidden (A1)
# print(list(model.parameters())[0].size())
#
# # Input --> Hidden Bias (B1)
# print(list(model.parameters())[2].size())
#
# # Hidden --> Hidden (A3)
# print(list(model.parameters())[1].size())
#
# # Hidden --> Hidden Bias (B3)
# print(list(model.parameters())[3].size())
#
# # Hidden --> Output (A2)
# print(list(model.parameters())[4].size())
#
# # Hidden --> Output Bias (B2)
# print(list(model.parameters())[5].size())
#
# # ---------------------------------------------------------------------------
#
# # Step 7: Train Model
#
# '''
# Process
#     1. Convert inputs/labels to variables
#         - RNN Input: (1, 28)
#         - CNN Input: (1, 28, 28)
#         - FNN Input: (1, 28*28)
#     2. Clear gradient buffets
#     3. Get output given inputs
#     4. Get loss
#     5. Get gradients w.r.t. parameters
#     6. Update parameters using gradients
#         - parameters = parameters - learning_rate * parameters_gradients
#     7. REPEAT
#
# '''
#
# seq_dim = 28
#
# iter = 0
# for epoch in range(num_epochs):
#     for i, (images, labels) in enumerate(train_loader):
#         # Load images as Variable
#         images = Variable(images.view(-1, seq_dim, input_dim))
#         labels = Variable(labels)
#
#         # Clear gradients w.r.t. parameters
#         optimizer.zero_grad()
#
#         # Forward pass to get output/logits
#         # outputs.size() --> 100, 10
#         outputs = model(images)
#
#         print(outputs)
#         print(labels)
#
#         # Calculate Loss: softmax --> cross entropy loss
#         loss = criterion(outputs, labels)
#
#         # Getting gradients w.r.t. parameters
#         loss.backward()
#
#         # Updating parameters
#         optimizer.step()
#
#         iter += 1
#
#         if iter % 500 == 0:
#             # Calculate Accuracy
#             correct = 0
#             total = 0
#             # Iterate through test dataset
#             for images, labels in test_loader:
#                 # Load images to a Torch Variable
#                 images = Variable(images.view(-1, seq_dim, input_dim))
#
#                 # Forward pass only to get logits/output
#                 outputs = model(images)
#
#                 # Get predictions from the maximum value
#                 _, predicted = torch.max(outputs.data, 1)
#
#                 # Total number of labels
#                 total += labels.size(0)
#
#                 # Total correct predictions
#                 correct += (predicted == labels).sum()
#
#             accuracy = 100 * correct / total
#
#             # Print Loss
#             print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.data[0], accuracy))


import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable

'''
STEP 1: LOADING DATASET
'''
train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor())

'''
STEP 2: MAKING DATASET ITERABLE
'''


batch_size = 100
n_iters = 3000
num_epochs = n_iters / (len(train_dataset) / batch_size)
num_epochs = int(num_epochs)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

'''
STEP 3: CREATE MODEL CLASS
'''


class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Building your RNN
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='tanh')

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        #######################
        #  USE GPU FOR MODEL  #
        #######################
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))

        # One time step
        out, hn = self.rnn(x, h0)

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        out = self.fc(out[:, -1, :])
        # out.size() --> 100, 10
        return out


'''
STEP 4: INSTANTIATE MODEL CLASS
'''
input_dim = 28
hidden_dim = 100
layer_dim = 2  # ONLY CHANGE IS HERE FROM ONE LAYER TO TWO LAYER
output_dim = 10

model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim)

#######################
#  USE GPU FOR MODEL  #
#######################

if torch.cuda.is_available():
    model.cuda()

'''
STEP 5: INSTANTIATE LOSS CLASS
'''
criterion = nn.CrossEntropyLoss()

'''
STEP 6: INSTANTIATE OPTIMIZER CLASS
'''
learning_rate = 0.1

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

'''
STEP 7: TRAIN THE MODEL
'''

# Number of steps to unroll
seq_dim = 28

iter = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Load images as Variable
        #######################
        #  USE GPU FOR MODEL  #
        #######################
        if torch.cuda.is_available():
            images = Variable(images.view(-1, seq_dim, input_dim).cuda())
            labels = Variable(labels.cuda())
        else:
            images = Variable(images.view(-1, seq_dim, input_dim))
            labels = Variable(labels)

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Forward pass to get output/logits
        # outputs.size() --> 100, 10
        outputs = model(images)

        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels)

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()

        iter += 1

        if iter % 500 == 0:
            # Calculate Accuracy
            correct = 0
            total = 0
            # Iterate through test dataset
            for images, labels in test_loader:
                #######################
                #  USE GPU FOR MODEL  #
                #######################
                if torch.cuda.is_available():
                    images = Variable(images.view(-1, seq_dim, input_dim).cuda())
                else:
                    images = Variable(images.view(-1, seq_dim, input_dim))

                # Forward pass only to get logits/output
                outputs = model(images)

                # Get predictions from the maximum value
                _, predicted = torch.max(outputs.data, 1)

                # Total number of labels
                total += labels.size(0)

                # Total correct predictions
                #######################
                #  USE GPU FOR MODEL  #
                #######################
                if torch.cuda.is_available():
                    correct += (predicted.cpu() == labels.cpu()).sum()
                else:
                    correct += (predicted == labels).sum()

            accuracy = 100 * correct / total

            # Print Loss
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.data.item(), accuracy))




