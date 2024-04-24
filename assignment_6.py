import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import random
import numpy as np

###################################################################################### Functions from previous sections

def train(model, optimizer, loss_fn, num_epochs, train_dl, valid_dl):
    '''
    Main function to train and test the model
    '''
    # lists to strore losses and accuracies
    loss_hist_train = [0] * num_epochs
    accuracy_hist_train = [0] * num_epochs
    loss_hist_valid = [0] * num_epochs
    accuracy_hist_valid = [0] * num_epochs
    # main loop through epochs
    for epoch in range(num_epochs):
        # training mode
        model.train()
        for x_batch, y_batch in train_dl:
            # core of the learning process: predict and fit
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # compute train loss and accuracy
            loss_hist_train[epoch] += loss.item()*y_batch.size(0)
            is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
            accuracy_hist_train[epoch] += is_correct.sum()
        # compute average loss per epoch
        loss_hist_train[epoch] /= len(train_dl.dataset)
        accuracy_hist_train[epoch] /= len(train_dl.dataset)
        # we also put the model in evaluation mode, so that specific layers such as dropout or batch normalization layers behave correctly.
        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in valid_dl:
                # predict
                pred = model(x_batch)
                loss = loss_fn(pred, y_batch)
                loss_hist_valid[epoch] += loss.item()*y_batch.size(0)
                is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
                accuracy_hist_valid[epoch] += is_correct.sum()
                if epoch==0:
                    preds,actuals=torch.argmax(pred, dim=1),y_batch
                else:
                    preds=torch.cat((preds,torch.argmax(pred, dim=1)),dim=0)
                    actuals=torch.cat((actuals,y_batch),dim=0)
        # compute average loss per epoch
        loss_hist_valid[epoch] /= len(valid_dl.dataset)
        accuracy_hist_valid[epoch] /= len(valid_dl.dataset)
        # print accuracy
        if (epoch+1) % 100==0:
            print(f'Epoch {epoch+1} accuracy: {accuracy_hist_train[epoch]:.4f} val_accuracy: {accuracy_hist_valid[epoch]:.4f}')
    return loss_hist_train, loss_hist_valid, accuracy_hist_train, accuracy_hist_valid, preds,actuals

def plot_losses(hist):
    ''' plots train and test loss
    Input
    ------
    history, the output of function train()
    '''
    x_arr = np.arange(len(hist[0])) + 1
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(x_arr, hist[0], '-o', label='Train loss')
    ax.plot(x_arr, hist[1], '--<', label='Test loss')
    ax.set_xlabel('Epoch', size=15)
    ax.set_ylabel('Loss', size=15)
    ax.legend(fontsize=15)
    ax.set_ylim([0, 20])
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(x_arr, hist[2], '-o', label='Train acc.')
    ax.plot(x_arr, hist[3], '--<', label='Test acc.')
    ax.legend(fontsize=15)
    ax.set_xlabel('Epoch', size=15)
    ax.set_ylabel('Accuracy', size=15)
    plt.show()

def plot_accuracy_from_predictions(hist):
    ''' Creates and prints confusion matrix from a model and a set of examples
    Inputs
    ------
    hist: tuple
        where hist[4] is the list of predicted values for test and hist[5] are the actual labels
    '''
    pred=hist[4].numpy()
    actual=hist[5].numpy()
    labels = np.unique(actual)
    disp = ConfusionMatrixDisplay.from_predictions(actual,pred,labels=labels)
    # print global accuracy
    accuracy=np.sum(np.diagonal(disp.confusion_matrix))/np.sum(disp.confusion_matrix)
    print(f'Accuracy on test set: {accuracy:.4f}')
    plt.show()

################################################################################ CIFAR10 Transform to Standard Scale

cifar_trainset = torchvision.datasets.CIFAR10(root='./', train=True, download=False, transform=transforms.ToTensor())

imgs = [item[0] for item in cifar_trainset] # item[0] and item[1] are image and its label
imgs = torch.stack(imgs, dim=0).numpy()

# calculate mean over each channel (r,g,b)
mean_r = imgs[:,0,:,:].mean()
mean_g = imgs[:,1,:,:].mean()
mean_b = imgs[:,2,:,:].mean()
print(f"Red channel mean = {mean_r}\nGreen channel mean = {mean_g}\nBlue channel mean = {mean_b}\n")

# calculate std over each channel (r,g,b)
std_r = imgs[:,0,:,:].std()
std_g = imgs[:,1,:,:].std()
std_b = imgs[:,2,:,:].std()
print(f"Red channel SD = {std_r}\nGreen channel SD = {std_g}\nBlue channel SD = {std_b}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((mean_r, mean_g, mean_b), (std_r, std_g, std_b))
])

################################################################################ Data and parameters

# load CIFAR10 data set
train_set = torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=transform) # 50000 images in training set
test_set = torchvision.datasets.CIFAR10(root='./', train=False, download=True, transform=transform) # 10000 images in test set

# parameter constants
hidden_size = 64
batch_size= 512
num_epochs = 30
# Optimizer specific options
learning_rate=0.0005
regularization_param=0.001
# Dropout: if p>0
dropout_p=0.4

########################################################################### train and test, pre-processing
X_train, y_train  = train_set.data, train_set.targets
X_test, y_test = test_set.data, test_set.targets

# select 1000 train images
num_images_train = len(X_train)
random_select_train = random.sample(range(num_images_train), 1000)
X_train = np.array([X_train[i] for i in random_select_train])
y_train = np.array([y_train[i] for i in random_select_train])
# select 200 test images
num_images_test = len(X_test)
random_select_test = random.sample(range(num_images_test), 200)
X_test = np.array([X_test[i] for i in random_select_test])
y_test = np.array([y_test[i] for i in random_select_test])

# Convert numpy arrays to PyTorch tensors of the right shape (labels do not need to be reshaped)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).permute(0, 3, 1, 2) # originally was NHWC
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).permute(0, 3, 1, 2)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Instantiate the model
input_size = X_train_tensor.shape[1]
output_size = len(train_set.classes)

# Create dataloader where batch size is set (note: batchsize is the first parameter in NCHW)
train_dl=DataLoader(TensorDataset(X_train_tensor,y_train_tensor), batch_size, shuffle=True)
test_dl=DataLoader(TensorDataset(X_test_tensor,y_test_tensor), batch_size, shuffle=True)

###################################################################################### CNN  model
model=nn.Sequential(
    nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,padding=1),
    nn.BatchNorm2d(32),
    nn.LeakyReLU(0.1),
    nn.MaxPool2d(kernel_size=2),
    nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1),
    nn.BatchNorm2d(64),
    nn.LeakyReLU(0.1),
    nn.MaxPool2d(kernel_size=2),
    nn.Flatten(),
    nn.Linear(64*8*8, hidden_size),
    nn.LeakyReLU(0.1),
    nn.Dropout(p=dropout_p),
    nn.Linear(hidden_size, hidden_size),
    nn.LeakyReLU(0.1),
    nn.Dropout(p=dropout_p),
    nn.Linear(hidden_size, output_size)
)

# model description
summary(model,(3,32,32)) # C, H, W

# Define loss function and optimizer
# Either torch.nn.NLLLoss or torch.nn.CrossEntropyLoss can be used: CrossEntropyLoss (https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) implements softmax internally
loss_fn = nn.CrossEntropyLoss()

# Optimizer: optimizer object that will hold the current state and will update the parameters based on the computed gradients
# for param in model.parameters(): print(param.data)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=regularization_param)

# Train the model and predict on test samples to estimate accuracy
# history stores losses, accuracy, actual labels and predictions
history = train(model, optimizer, loss_fn, num_epochs, train_dl, test_dl)

# plot losses along epochs
plot_losses(history)
# plot confusion matrix
plot_accuracy_from_predictions(history)
