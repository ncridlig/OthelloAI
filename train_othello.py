import os.path

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

def save_model(file_folder, model):

    path = os.path.join(file_folder, 'best_model.pth')
    torch.save(model.state_dict(), path)

def normal_train(train_loader, val_loader, args):
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

        for step, (inputs, labels) in progress_bar(enumerate(train_loader), total=len(train_loader)):

            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()

            logits = model(inputs)
            softmax_logits = F.softmax(logits, dim=1)
            predictions = torch.argmax(softmax_logits, dim=1)
            loss = criterion(logits, labels)
            loss.backward()

            optimizer.step()  # backprop to update the weights
            model.scheduler.step()  # Update learning rate schedule
            optimizer.zero_grad()
            losses += loss.item()

        # Run on validation set
        run_eval(args, model, datasets, tokenizer, split='validation')
        print('epoch', epoch_count, '| losses:', losses)


def params():
    """
    Loads the hyperparameters passed into the command line
    :return: argparser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", default=1, type=int,
                        help="Batch size per GPU/CPU for training and evaluation.")
    parser.add_argument("--learning-rate", default=1e-3, type=float,
                        help="Model learning rate starting point.")
    parser.add_argument("--batch-size", default=1, type=int,
                        help="Batch size per GPU/CPU for training and evaluation.")
    parser.add_argument("--weight-decay", default=1e-8, type=float,
                        help="L2 Regularization")
    parser.add_argument("--swa-enabled", action='store_true',
                        help="Enables Stochastic Weighting Average")
    parser.parse_args()
    return parser

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
    model = None

    if torch.cuda.is_available():
        model = model.cuda()


