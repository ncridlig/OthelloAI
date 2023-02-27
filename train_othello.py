import os.path
import os
import pdb

import pandas as pd

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm as progress_bar
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from model import Conv8
from copy import deepcopy
import time

def compute_accuracy(logits, labels, args):

    softmax_logits = F.softmax(logits, dim=1)
    if args.sample_softmax:
        predictions = torch.multinomial(softmax_logits, num_samples=1).flatten()
    else:
        predictions = torch.argmax(softmax_logits, dim=1) # list of batchsize

    sorted_softmax = torch.argsort(softmax_logits, dim=1, descending=True)

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

    # Save the model into the designated folder
    path = os.path.join(file_folder, timestr + '.pth')
    torch.save(model, path)


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
            inputs, labels = inputs.float(), labels.float()
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
            inputs, labels = inputs.to(torch.float32), labels.to(torch.float32)
            inputs, labels = inputs.to(device), labels.to(device)

        num_examples += inputs.size()[0]
        logits = model(inputs)
        prediction, batch_accuracies = compute_accuracy(logits, labels, args)
        running_accuracy = [running_accuracy[i] + batch_accuracies[i] for i in range(len(running_accuracy))]
        labels = labels.to(torch.int64)
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

    # Initialize the cross entropy and Adam optimizer to update weights
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    epochs = args.epochs

    # Used to save the best model
    best_valid_accuracy = [0.0, 0.0, 0.0]
    best_training_accuracy = [0.0, 0.0, 0.0]
    best_model = None

    for epoch_count in range(epochs):

        # Initializing running accuracy and loss statistics
        losses = 0
        model.train()
        running_accuracy = [0, 0, 0]
        num_examples = 0
        for step, (inputs, labels) in progress_bar(enumerate(train_loader), total=len(train_loader)):

            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()
                inputs, labels = inputs.float(), labels.float()

            elif torch.backends.mps.is_available():
                device = torch.device('mps')
                inputs, labels = inputs.to(torch.float32), labels.to(torch.float32)
                inputs, labels = inputs.to(device), labels.to(device)
            
            num_examples += inputs.size()[0]
            logits = model(inputs)
            prediction, batch_accuracies = compute_accuracy(logits, labels, args)
            running_accuracy = [running_accuracy[i] + batch_accuracies[i] for i in range(len(running_accuracy))]
            labels = labels.to(torch.int64)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()  # backprop to update the weights
            optimizer.zero_grad()
            losses += loss.item()

        avg_accuracy = [100 * running_accuracy[i] / num_examples for i in range(len(running_accuracy))]
        avg_loss = losses/num_examples
        val_accuracy, val_loss = validate_normal_model(val_loader, model, args, criterion)
        print()
        print('----------------------------  Epoch ' + str(epoch_count) + ' ----------------------------')
        print('Training Loss: ' + str(avg_loss))
        print('Training Accuracies: ' + str(avg_accuracy))
        print('Validation Loss: ' + str(val_loss))
        print('Validation Accuracies: ' + str(val_accuracy))
        print('------------------------------------------------------------------')

        # Update the model if it achieves a higher
        # accuracy then the previous model
        if val_accuracy[0] > best_valid_accuracy[0]:
            best_valid_accuracy[0] = val_accuracy[0]
            best_model = deepcopy(model.state_dict())

        # Update calidation accuracy statistics
        best_valid_accuracy[1] = max(val_accuracy[1], best_valid_accuracy[1])
        best_valid_accuracy[2] = max(val_accuracy[2], best_valid_accuracy[2])

        # Update training accuracy statistics
        best_training_accuracy[0] = max(avg_accuracy[0], best_training_accuracy[0])
        best_training_accuracy[1] = max(avg_accuracy[1], best_training_accuracy[1])
        best_training_accuracy[2] = max(avg_accuracy[2], best_training_accuracy[2])


    # Save the best model from training
    save_model('models', best_model)
    return best_training_accuracy, best_valid_accuracy


def params():
    """
    Loads the hyperparameters passed into the command line
    :return: argparser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning-rate", default=5e-4, type=float,
                        help="Model learning rate starting point.")
    parser.add_argument("--batch-size", default=4096, type=int,
                        help="Batch size per GPU/CPU for training and evaluation.")
    parser.add_argument("--weight-decay", default=1e-4 , type=float,
                        help="L2 Regularization")
    parser.add_argument("--epochs", default=8, type=int,
                        help="Number of epochs to train for")
    parser.add_argument("--swa-enabled", action='store_true',
                        help="Enables Stochastic Weighting Average")
    parser.add_argument("--sample-softmax", action='store_true',
                        help="Samples from Move Probability Distribution")
    args = parser.parse_args()
    return args

def save_experiment(args, statistics):
    """
    Saves the experiment results to a csv
    :param args: The hyperparameters used
    :param statistics: The accuracies for the training, validation, and test sets
    """
    trial_dict = {
        'Model name': [timestr],
        'Learning rate': [args.learning_rate],
        'Batch size': [args.batch_size],
        'Weight decay': [args.weight_decay],
        'Epochs': [args.epochs],
        'Stochastic Weighting Averaging': [args.swa_enabled],
        'Softmax Sampling': [args.sample_softmax],
        'Max Training Accuracy': [statistics[0][0]],
        'Max Training Accuracy3': [statistics[0][1]],
        'Max Training Accuracy5': [statistics[0][2]],
        'Max Validation Accuracy': [statistics[1][0]],
        'Max Validation Accuracy3': [statistics[1][1]],
        'Max Validation Accuracy5': [statistics[1][2]],
        'Testing Accuracy': [statistics[2][0]],
        'Testing Accuracy3': [statistics[2][1]],
        'Testing Accuracy5': [statistics[2][2]],
        'Testing Loss': [statistics[3]]
    }

    trial_dict = pd.DataFrame(trial_dict)
    need_header = not os.path.exists('results.csv')

    if need_header:
        trial_dict.to_csv('results.csv', index=False, header=need_header)
    else:
        trial_dict.to_csv('results.csv', mode='a', index=False, header=need_header)



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

    # Load the hyperparameters and the dataset
    args = params()
    train_dataset, val_dataset, test_dataset = load_data()

    # Create a PyTorch Dataloader for the datasets
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize Othello model
    model = Conv8()
    timestr = time.strftime("%Y%m%d-%H%M%S")

    # Convert to mps or cuda
    if torch.cuda.is_available():
        model = model.cuda()
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        model = model.to(device)

    # Training the model on the training set
    best_training_acc, best_val_acc = normal_train(model=model, train_loader=train_loader,
                                                   val_loader=val_loader, args=args)

    # Load the best model and evaluate it on the test set
    best_model = Conv8()
    best_model.load_state_dict(torch.load(os.path.join('models', timestr, 'best_model.pth')))

    if torch.cuda.is_available():
        best_model = best_model.cuda()
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        best_model = best_model.to(device)

    test_accuracy, test_loss = validate_normal_model(val_loader=test_loader, model=best_model,args=args,
                                         criterion=torch.nn.CrossEntropyLoss())
    statistics = [best_training_acc, best_val_acc, test_accuracy, test_loss]
    save_experiment(args, statistics)

