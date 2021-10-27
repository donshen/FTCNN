import os
import h5py
import argparse
import numpy as np
from tqdm import tqdm
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.optim import *
from cnnmodel.model import CNNModel

parser = argparse.ArgumentParser()
parser.add_argument('--proc_path',
                    type=str,
                    help='path that contain preprocessed training and test data splitted from FT and shifted voxilized point clouds, in .h5 format')
parser.add_argument('--save_path',
                    type=str,
                    help='path that the trained CNN model is trained')
parser.add_argument('-n',
                    '--num_epochs',
                    type=int,
                    default=100,
                    help='number of epochs to train')
parser.add_argument('-b',
                    '--batch_size',
                    type=int,
                    default=128)
opt = parser.parse_args()

def load_data(data_path):
    # Load Data
    with h5py.File(data_path, "r") as hf:    
        # Split the data into training/test features/targets
        X_train = hf["X_train"][:]
        y_train = hf["y_train"][:]
        X_test = hf["X_test"][:]
        y_test = hf["y_test"][:]

    # Resolution of the voxels
    res = X_train[0].shape[0]

    np.array(X_train).reshape(len(X_train), 1, res, res, res)
    np.array(X_test).reshape(len(X_test), 1, res, res, res)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    train_x = torch.from_numpy(np.array(X_train)).float()
    train_y = torch.from_numpy(np.array(y_train)).long()

    test_x = torch.from_numpy(np.array(X_test)).float()
    test_y = torch.from_numpy(np.array(y_test)).long()

    # Pytorch train and test sets
    train = torch.utils.data.TensorDataset(train_x, train_y)
    test = torch.utils.data.TensorDataset(test_x, test_y)
    return train, test

# Select device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load data from preprocessed path
train, test = load_data(opt.proc_path)

# data loader
train_loader = torch.utils.data.DataLoader(train, opt.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, opt.batch_size, shuffle=True)

#Definition of hyperparameters
n_iters = (len(train) / opt.batch_size) * opt.num_epochs

# Load CNNModel
model = CNNModel().to(device)
#model.cuda()
print(model)

# Cross Entropy Loss 
loss_fn = nn.CrossEntropyLoss()

# Adam
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

# Train
count = 0
loss_list = []
iteration_list = []
accuracy_list = []
for epoch in tqdm(range(opt.num_epochs)):
    scheduler.step()
    for i, (voxels, voxel_labels) in enumerate(train_loader):
        res_x, res_y, res_z = voxels.shape[-3:]
        voxels, voxel_labels = voxels.to(device), voxel_labels.to(device)
        if voxels.shape[0] != opt.batch_size:
            continue
        train = Variable(voxels.view(opt.batch_size, 1, res_x, res_y, res_z))
        voxel_labels = Variable(voxel_labels)
        # Clear gradients
        optimizer.zero_grad()
        # Forward propagation
        outputs = model(train)
        # Calculate softmax and ross entropy loss
        loss = loss_fn(outputs, voxel_labels)
        # Calculating gradients
        loss.backward()
        # Update parameters
        optimizer.step()
        
        count += 1
        if count % 100 == 0:
            # Calculate Accuracy         
            correct = 0
            total = 0
            # Iterate through test dataset
            for voxels, voxel_labels in test_loader:
                voxels, voxel_labels = voxels.to(device), voxel_labels.to(device)
                if voxels.shape[0] != opt.batch_size:
                    continue
                test = Variable(voxels.view(opt.batch_size, 1, res_x, res_y, res_z))
                # Forward propagation
                outputs = model(test)
                # Get predictions from the maximum value
                predicted = torch.max(outputs.data, 1)[1]
                # Total number of labels
                total += len(voxel_labels)
                correct += (predicted == voxel_labels).sum()
            
            accuracy = 100 * correct / float(total)
            
            # store loss and iteration
            iteration_list.append(count)
            loss_list.append(loss.data.cpu())
            accuracy_list.append(accuracy.cpu())
 
            # Print Loss
            print(f'Iteration: {count}  Loss: {loss.data}  Accuracy: {accuracy}')
            
        # Save plots for loss and accuracy
        np.savetxt(os.path.join(opt.save_path, 'loss_acc.txt'),
                   np.array((iteration_list, loss_list, accuracy_list)).T, fmt='%10.5f')
              
    # Save the trained model
    if epoch % 100 == 0:
        torch.save(model, opt.save_path + '/' + str(epoch) + '.pth')
        print(f'Model saved to {opt.save_path}.')