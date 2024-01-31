import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import scipy.stats as stats
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

def experiment_with_minibatch_sizes(batch_sizes, num_epochs=700):
   
    

    class ANNwine(nn.Module):
        def __init__(self):
            super().__init__()

            ### input layer
            self.input = nn.Linear(11, 16)

            ### hidden layers
            self.fc1 = nn.Linear(16, 32)
            self.fc2 = nn.Linear(32, 32)

            ### output layer
            self.output = nn.Linear(32, 1)

        # forward pass
        def forward(self, x):
            x = F.relu(self.input(x))
            x = F.relu(self.fc1(x))  # fully connected
            x = F.relu(self.fc2(x))
            return self.output(x)

    # import the data
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    data = pd.read_csv(url, sep=';')
    data = data[data['total sulfur dioxide'] < 200]  # drop a few outliers

    # z-score all columns except for quality
    cols2zscore = data.keys()
    cols2zscore = cols2zscore.drop('quality')
    data[cols2zscore] = data[cols2zscore].apply(stats.zscore)

    # create a new column for binarized (boolean) quality
    data['boolQuality'] = 0
    data.loc[data['quality'] > 5, 'boolQuality'] = 1  # fix this line

    # convert from pandas dataframe to tensor
    dataT = torch.tensor(data[cols2zscore].values).float()
    labels = torch.tensor(data['boolQuality'].values).float()
    labels = labels[:, None]  # transform to matrix

    # use scikit-learn to split the data
    train_data, test_data, train_labels, test_labels = train_test_split(dataT, labels, test_size=.1)

    # convert them into PyTorch Datasets (note: already converted to tensors)
    train_dataDataset = TensorDataset(train_data, train_labels)
    test_dataDataset = TensorDataset(test_data, test_labels)

    # initialize output results matrices
    accuracyResultsTrain = np.zeros((num_epochs, len(batch_sizes)))
    accuracyResultsTest = np.zeros((num_epochs, len(batch_sizes)))
    comptime = np.zeros(len(batch_sizes))

    # test data doesn't vary by training batch size
    test_loader = DataLoader(test_dataDataset, batch_size=test_dataDataset.tensors[0].shape[0])



    # Time taking process (15 min)


    # loop over batch sizes
    for bi in range(len(batch_sizes)):
        # start the clock!
        starttime = time.process_time()

        # create dataloader object
        train_loader = DataLoader(train_dataDataset,
                                  batch_size=int(batch_sizes[bi]), shuffle=True, drop_last=True)

        # create and train a model
        winenet = ANNwine()
        trainAcc, testAcc, losses = trainTheModel(winenet, train_loader, test_loader, num_epochs=num_epochs)

        # store data
        accuracyResultsTrain[:, bi] = trainAcc
        accuracyResultsTest[:, bi] = testAcc

        # check the timer
        comptime[bi] = time.process_time() - starttime

    # plot some results
    fig, ax = plt.subplots(1, 2, figsize=(17, 7))

    ax[0].plot(accuracyResultsTrain)
    ax[0].set_title('Train accuracy')
    ax[1].plot(accuracyResultsTest)
    ax[1].set_title('Test accuracy')

    # common features
    for i in range(2):
        ax[i].legend(batch_sizes)
        ax[i].set_xlabel('Epoch')
        ax[i].set_ylabel('Accuracy (%)')
        ax[i].set_ylim([50, 100])
        ax[i].grid()

    plt.show()

    # bar plot of computation time
    plt.bar(range(len(comptime)), comptime, tick_label=batch_sizes)
    plt.xlabel('Mini-batch size')
    plt.ylabel('Computation time (s)')
    plt.show()


# a function that trains the model
def trainTheModel(model, train_loader, test_loader, num_epochs=700):
    # loss function and optimizer
    lossfun = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)

    # initialize losses
    losses = torch.zeros(num_epochs)
    trainAcc = []
    testAcc = []

    # loop over epochs
    for epochi in range(num_epochs):
        # switch on training mode
        model.train()

        # loop over training data batches
        batchAcc = []
        batchLoss = []
        for X, y in train_loader:
            # forward pass and loss
            y_hat = model(X)
            loss = lossfun(y_hat, y)

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # loss from this batch
            batchLoss.append(loss.item())

            # compute training accuracy for this batch
            batchAcc.append(100 * torch.mean(((y_hat > 0) == y).float()).item())
        # end of batch loop...

        # now that we've trained through the batches, get their average training accuracy
        trainAcc.append(np.mean(batchAcc))

        # and get average losses across the batches
        losses[epochi] = np.mean(batchLoss)

        # test accuracy
        model.eval()
        X, y = next(iter(test_loader))  # extract X,y from test dataloader
        with torch.no_grad():  # deactivates autograd
            y_hat = model(X)
        testAcc.append(100 * torch.mean(((y_hat > 0) == y).float()).item())

    # function output
    return trainAcc, testAcc, losses
 

# Run the function with specified batch sizes
experiment_with_minibatch_sizes([2, 4, 8, 16, 32, 64, 128, 256, 512])
