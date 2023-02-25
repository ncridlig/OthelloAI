import os.path
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm as progress_bar
import argparse
from torch.cuda import is_available
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.datasets as dset
import torchvision.transforms as T
import pdb
from model import Conv8

def compute_accuracy(logits, labels, args):

    softmax_logits = F.softmax(logits, dim=1)
    if args.sample_softmax:
        predictions = torch.multinomial(softmax_logits, num_samples=1)
    else:
        predictions = torch.argmax(softmax_logits, dim=1) # list of batchsize

    sorted_softmax = torch.argsort(softmax_logits, dim=1)

    # Accuracy 1
    num_correct = torch.sum(predictions == labels)
    accuracy_1 = num_correct.item()

    # Accuracy 3
    sorted_confidence_3 = sorted_softmax[:,:3] # provides a batchsize x 3
    top_3 = [labels[i] in sorted_confidence_3[i] for i in range(len(labels))]
    accuracy_3 = sum(top_3)

    # Accuracy 5
    sorted_confidence_5 = sorted_softmax[:,:5] # provides a batchsize x 5
    top_5 = [labels[i] in sorted_confidence_5[i] for i in range(len(labels))]
    accuracy_5 = sum(top_5)

    return predictions, [accuracy_1, accuracy_3, accuracy_5]

def save_model(file_folder, model):

    path = os.path.join(file_folder, 'best_model.pth')
    torch.save(model.state_dict(), path)


def validate_normal_model(val_loader, model, args, criterion):

    """
    Evaluates the model on a given dataset
    """

    model.eval()
    losses = 0
    running_accuracy = [0, 0, 0]
    num_examples = 0
    for step, (inputs, labels) in progress_bar(enumerate(val_loader), total=len(val_loader)):

        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
            inputs, labels = inputs.to(torch.float32), labels.to(torch.float32)
            inputs, labels = inputs.to(device), labels.to(device)
            
        
        num_examples += inputs.size()[0]

        logits = model(inputs)
        prediction, batch_accuracies = compute_accuracy(logits, labels, args)
        running_accuracy = [running_accuracy[i] + batch_accuracies[i] for i in range(len(running_accuracy))]
        loss = criterion(logits, labels)
        loss.backward()
        losses += loss.item()

    avg_accuracy = [100*running_accuracy[i]/num_examples for i in range(len(running_accuracy))]
    avg_loss = losses/num_examples
    return avg_accuracy, avg_loss







def normal_train(model, train_loader, val_loader, args):
    """
    Performs training without SWA
    :param train_loader: training dataset
    :param val_loader: validation dataset
    """


    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    epochs = args.epochs

    for epoch_count in range(epochs):
        losses = 0
        model.train()
        running_accuracy = [0, 0, 0]
        num_examples = 0
        for step, (inputs, labels) in progress_bar(enumerate(train_loader), total=len(train_loader)):

            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()
            elif torch.backends.mps.is_available():
                device = torch.device('mps')
                inputs, labels = inputs.to(torch.float32), labels.to(torch.float32)
                inputs, labels = inputs.to(device), labels.to(device)
            
            num_examples += inputs.size()[0]

            logits = model(inputs)
            prediction, batch_accuracies = compute_accuracy(logits, labels, args)
            running_accuracy = [running_accuracy[i] + batch_accuracies[i] for i in range(len(running_accuracy))]
            labels = labels.to(torch.int32)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()  # backprop to update the weights
            optimizer.zero_grad()
            losses += loss.item()

        avg_accuracy = [100 * running_accuracy[i] / num_examples for i in range(len(running_accuracy))]
        avg_loss = losses/num_examples
        val_accuracy, val_loss = validate_normal_model(val_loader, model, args, criterion)
        print('----------------------------  Epoch ' + str(epoch_count) + ' ----------------------------')
        print('Training Loss: ' + avg_loss)
        print('Training Accuracies: ' + avg_accuracy)
        print('Validation Loss: ' + val_loss)
        print('Validation Accuracies: ' + val_accuracy)
        print('-----------------------------------------------------------------')

    


def params():
    """
    Loads the hyperparameters passed into the command line
    :return: argparser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning-rate", default=1e-3, type=float,
                        help="Model learning rate starting point.")
    parser.add_argument("--batch-size", default=1024, type=int,
                        help="Batch size per GPU/CPU for training and evaluation.")
    parser.add_argument("--weight-decay", default=1e-8, type=float,
                        help="L2 Regularization")
    parser.add_argument("--epochs", default=100, type=int,
                        help="Number of epochs to train for")
    parser.add_argument("--swa-enabled", action='store_true',
                        help="Enables Stochastic Weighting Average")
    parser.add_argument("--sample-softmax", action='store_true',
                        help="Samples from Move Probability Distribution")
    args = parser.parse_args()
    return args

def load_data():

    # Load data and labels from .npy files
    train_data = np.load('OthelloData/train_data.npy')
    train_labels = np.load('OthelloData/train_labels.npy')

    test_data = np.load('OthelloData/test_data.npy')
    test_labels = np.load('OthelloData/test_labels.npy')

    val_data = np.load('OthelloData/val_data.npy')
    val_labels = np.load('OthelloData/val_labels.npy')

    # Convert to PyTorch Tensors
    train_data = torch.from_numpy(train_data)
    train_labels = torch.from_numpy(train_labels)

    test_data = torch.from_numpy(test_data)
    test_labels = torch.from_numpy(test_labels)

    val_data = torch.from_numpy(val_data)
    val_labels = torch.from_numpy(val_labels)

    # Create a PyTorch Dataset from the data and labels
    train_dataset = TensorDataset(train_data, train_labels)
    test_dataset = TensorDataset(test_data, test_labels)
    val_dataset = TensorDataset(val_data, val_labels)

    return train_dataset, val_dataset, test_dataset


if __name__ == '__main__':


    args = params()
    train_dataset, val_dataset, test_dataset = load_data()

    # Create a PyTorch Dataloader for the datasets
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize Othello model
    model = Conv8()

    if torch.cuda.is_available():
        model = model.cuda()
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        model = model.to(device)

    normal_train(model=model, train_loader=train_loader, val_loader=val_loader, args=args)

