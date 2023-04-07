
import os
import torch
import argparse

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


from models.vgg import VGG
from utils import get_training_dataloader, get_test_dataloader, save_checkpoint


def parse_opt():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
    parser.add_argument('--dataset', type=str, default='cifar100', help='training dataset (default: cifar100)')
    parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true', help='train with channel sparsity regularization')
    parser.add_argument('--s', type=float, default=0.0001, help='scale sparse rate (default: 0.0001)')
    parser.add_argument('--refine', default='', type=str, metavar='PATH', help='path to the pruned model to be fine tuned')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=256, metavar='N', help='input batch size for testing (default: 256)')
    parser.add_argument('--epochs', type=int, default=160, metavar='N', help='number of epochs to train (default: 160)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--save', default='./logs', type=str, metavar='PATH', help='path to save prune model (default: current directory)')
    parser.add_argument('--arch', default='vgg', type=str,  help='architecture to use')
    parser.add_argument('--depth', default=19, type=int, help='depth of the neural network')

    args = parser.parse_args()
    return args


# additional subgradient descent on the sparsity-induced penalty term
def updateBN():
    for m in model.modules():
        # Check if the module is a BatchNorm2d layer
        if isinstance(m, nn.BatchNorm2d):
            # Calculate the L1 regularization term and add it to the weight gradients
            # args.s is a scalar value that determines the strength of the regularization
            # torch.sign(m.weight.data) returns the sign of the weight parameters
            m.weight.grad.data.add_(args.s*torch.sign(m.weight.data))  # L1


def train(epoch):
    # Set the model to training mode
    model.train()
    # Loop through the batches in the training data
    for batch_idx, (data, target) in enumerate(train_loader):
        # Move the data and target tensors to the GPU if args.cuda is True
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        # Zero out the gradients in the optimizer
        optimizer.zero_grad()
        # Forward pass: compute the output of the model on the input data
        output = model(data)
        # Compute the loss between the output and target labels
        loss = F.cross_entropy(output, target)
        # Backward pass: compute the gradients of the loss w.r.t. the model parameters
        loss.backward()
        # If args.sr is True, apply L1 regularization to the Batch Normalization layers
        if args.sr:
            updateBN()
        # Update the model parameters using the optimizer
        optimizer.step()
        # Print the training loss and progress at regular intervals
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), 
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader), 
                loss.item()))


def test():
    # Set the model to evaluation mode
    model.eval()
    # Initialize test loss and correct predictions
    test_loss = 0
    correct = 0
    # Turn off gradient calculation during inference
    with torch.no_grad():    
        # Loop through the test data
        for data, target in test_loader:
            # Move the data and target tensors to the GPU if args.cuda is True
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            # Compute the output of the model on the input data
            output = model(data)
            # Compute the test loss and add it to the running total
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            # Compute the predictions from the output using the argmax operation
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            # Compute the number of correct predictions and add it to the running total
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    # Compute the average test loss and accuracy
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    # Print the test results
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, 
        len(test_loader.dataset),
        accuracy))
    return accuracy / 100.


if __name__ == '__main__':
    # Parse command line arguments
    args = parse_opt()
    # Check if CUDA is available and set args.cuda flag accordingly
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    # Set the random seed for PyTorch and CUDA if args.cuda is True
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    # Create the save directory if it does not exist
    if not os.path.exists(args.save):
        os.makedirs(args.save)
        
    # Set kwargs to num_workers=1 and pin_memory=True if args.cuda is True, 
    # otherwise kwargs is an empty dictionary
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    
    # Create data loaders for the CIFAR10 dataset 
    # using the get_training_dataloader() and get_test_dataloader() functions
    if args.dataset == 'cifar10':
        train_loader = get_training_dataloader(batch_size=args.batch_size, **kwargs)
        test_loader  = get_test_dataloader(batch_size=args.test_batch_size, **kwargs)
        
    # Load a pre-trained VGG model if args.refine is not None, 
    # otherwise create a new VGG model
    if args.refine:
        checkpoint = torch.load(args.refine)
        model = VGG(depth=args.depth, cfg=checkpoint['cfg'])
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model = VGG(depth=args.depth)
    # Move the model to the GPU if args.cuda is True
    if args.cuda:
        model.cuda()
    # Set up the optimizer with Stochastic Gradient Descent (SGD) 
    # and the specified learning rate, momentum, and weight decay
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, 
        momentum=args.momentum, weight_decay=args.weight_decay)

    if args.resume:
        # Check if the checkpoint file exists
        if os.path.isfile(args.resume):
            # If the checkpoint file exists, print a message indicating that it's being loaded
            print("=> loading checkpoint '{}'".format(args.resume))
            # Load the checkpoint file
            checkpoint = torch.load(args.resume)
            # Update the start epoch and best precision variables from the checkpoint
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            
            # Load the model state dictionary and optimizer state dictionary from the checkpoint
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            # Print a message indicating that the checkpoint has been loaded
            print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
                .format(args.resume, checkpoint['epoch'], best_prec1))
        else:
            # If the checkpoint file does not exist, print an error message
            print("=> no checkpoint found at '{}'".format(args.resume))
            
    # Initialize the best test accuracy to 0
    best_prec1 = 0.
    # Loop through the epochs, starting from args.start_epoch and continuing until args.epochs
    for epoch in range(args.start_epoch, args.epochs):
        # If the current epoch is at 50% or 75% of the total epochs, 
        # reduce the learning rate by a factor of 10
        if epoch in [args.epochs*0.5, args.epochs*0.75]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        # Train the model on the training data for the current epoch
        train(epoch)
        # Evaluate the model on the test data and compute the top-1 test accuracy
        prec1 = test()
        # Check if the current test accuracy is better than the previous best test accuracy
        is_best = prec1 > best_prec1
        # Update the best test accuracy and save a checkpoint of the model and optimizer state
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, filepath=args.save)
    # Print the best test accuracy achieved during training
    print("Best accuracy: "+str(best_prec1))