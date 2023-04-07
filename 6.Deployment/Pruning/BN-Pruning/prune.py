import os
import argparse
import numpy as np
import torch
import torch.nn as nn

from models.vgg import VGG
from utils import get_test_dataloader


def parse_opt():
    # Prune settings
    parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
    parser.add_argument('--dataset', type=str, default='cifar100', help='training dataset (default: cifar10)')
    parser.add_argument('--test-batch-size', type=int, default=256, metavar='N', help='input batch size for testing (default: 256)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--depth', type=int, default=19, help='depth of the vgg')
    parser.add_argument('--percent', type=float, default=0.5, help='scale sparse rate (default: 0.5)')
    parser.add_argument('--model', default='', type=str, metavar='PATH', help='path to the model (default: none)')
    parser.add_argument('--save', default='logs/', type=str, metavar='PATH', help='path to save pruned model (default: none)')
    args = parser.parse_args()
    return args

# simple test model after Pre-processing prune (simple set BN scales to zeros)
# Define a function named test that takes a PyTorch model as input
def test(model):
    # Set kwargs to num_workers=1 and pin_memory=True if args.cuda is True, 
    # otherwise kwargs is an empty dictionary
    # kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    
    # Create a test data loader for the CIFAR10 dataset if args.dataset is 'cifar10'
    if args.dataset == 'cifar10':
        # test_loader = get_test_dataloader(batch_size=args.test_batch_size, **kwargs)
        test_loader = get_test_dataloader(batch_size=args.test_batch_size)
        
    else:
        raise ValueError("No valid dataset is given.")
    # Set the model to evaluation mode
    model.eval()
    # Initialize the number of correct predictions to 0
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
            # Compute the predictions from the output using the argmax operation
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            # Compute the number of correct predictions and add it to the running total
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    # Compute the test accuracy and print the result
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
        correct, len(test_loader.dataset), accuracy))
    # Return the test accuracy as a float
    return accuracy / 100.



if __name__ == '__main__':
    # Parse command line arguments using the parse_opt() function
    args = parse_opt()
    # Check if CUDA is available and set args.cuda flag accordingly
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    # Create the save directory if it does not exist
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    # Create a new VGG model with the specified depth
    model = VGG(depth=args.depth)
    # Move the model to the GPU if args.cuda is True
    if args.cuda:
        model.cuda()
    # If args.model is not None, 
    # attempt to load a checkpoint from the specified file
    if args.model:
        if os.path.isfile(args.model):
            print("=> loading checkpoint '{}'".format(args.model))
            checkpoint = torch.load(args.model)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
                .format(args.model, checkpoint['epoch'], best_prec1))
        else:
            print("=> no checkpoint found at '{}'".format(args.model))
            
    # Print the model to the console
    print(model)
    # Initialize the total number of channels to 0
    total = 0
    # Loop through the model's modules and count the number of channels in each BatchNorm2d layer
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            total += m.weight.data.shape[0]
    # Create a new tensor to store the absolute values of the weights of each BatchNorm2d layer
    bn = torch.zeros(total)
    # Initialize an index variable to 0
    index = 0
    # Loop through the model's modules again and 
    # store the absolute values of the weights of each BatchNorm2d layer in the bn tensor
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            bn[index:(index+size)] = m.weight.data.abs().clone()
            index += size
    # Sort the bn tensor and compute the threshold value for pruning
    y, i = torch.sort(bn)
    thre_index = int(total * args.percent)
    thre = y[thre_index]
    
    # Initialize the number of pruned channels to 0 and 
    # create lists to store the new configuration and mask for each layer
    pruned = 0
    cfg = []
    cfg_mask = []
    # Loop through the model's modules a third time and 
    # prune each BatchNorm2d layer that falls below the threshold value
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d):
            # Compute a mask indicating which weights to keep and which to prune
            weight_copy = m.weight.data.abs().clone()
            # tensor.gt(): greater-than operator
            mask = weight_copy.gt(thre).float().cuda()
            # calculate the number of parameters to be pruned
            pruned = pruned + mask.shape[0] - torch.sum(mask)
            # Apply the mask to the weight and bias tensors of the BatchNorm2d layer
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            # Record the new configuration and mask for this layer
            cfg.append(int(torch.sum(mask)))
            cfg_mask.append(mask.clone())
            # Print information about the pruning for this layer
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                format(k, mask.shape[0], int(torch.sum(mask))))
        elif isinstance(m, nn.MaxPool2d):
            # If the module is a MaxPool2d layer, 
            # record it as an 'M' in the configuration list
            cfg.append('M')
    # Compute the ratio of pruned channels to total channels
    pruned_ratio = pruned/total
    # Print a message indicating that the pre-processing was successful
    print('Pre-processing Successful!')
    # Evaluate the pruned model on the test set and 
    # store the accuracy in the acc variable
    acc = test(model)
    
# ============================ Make real prune ============================

    # Print the new configuration to the console
    print(cfg)
    # Initialize a new VGG model with the pruned configuration
    newmodel = VGG(cfg=cfg)
    # Move the new model to the GPU if available
    if args.cuda:
        newmodel.cuda()
    # Compute the number of parameters in the new model 
    num_parameters = sum([param.nelement() for param in newmodel.parameters()])
    # Save the configuration above, number of parameters, and test accuracy to a file
    savepath = os.path.join(args.save, "prune.txt")
    with open(savepath, "w") as fp:
        fp.write("Configuration: \n"+str(cfg)+"\n")
        fp.write("Number of parameters: "+str(num_parameters)+"\n")
        fp.write("Test accuracy: "+str(acc))

    # Initialize variables for the masks corresponding to the start and end of each pruned layer
    layer_id_in_cfg = 0
    start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]
    
    # Loop through the modules of the original and new models
    # Copy the weights and biases of each layer from the original model to the new model
    # Applying the appropriate masks to the weights and biases of the pruned layers
    for [m0, m1] in zip(model.modules(), newmodel.modules()):
        
        # ============================ Conv2d Layers ============================
        # If the module is a Conv2d layer, 
        # compute the indices of the non-zero weights in the input and output channels and 
        # copy them from the original model
        if isinstance(m0, nn.Conv2d):
            # Get the indices of input and output channels that are not pruned for this convolutional layer, 
            # by converting the start and end masks from the previous and current layers into numpy arrays, 
            # finding the non-zero elements, and removing the extra dimensions
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            # Print the number of input and output channels that are not pruned
            print('In channels: {:d}, Out channels {:d}.'.format(idx0.size, idx1.size))
            # If either idx0 or idx1 has a size of 1, 
            # resize it to (1,) to avoid a broadcasting error.
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            # Extract the weight tensor for this layer from the original model (m0) 
            # by selecting the input and output channels that are not pruned, 
            # and clone it to create a new tensor (w1)
            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
            w1 = w1[idx1.tolist(), :, :, :].clone()
            m1.weight.data = w1.clone()
        
        # ============================ BatchNorm Layers ============================
        # If the module is a BatchNorm2d layer, 
        # compute the indices of the non-zero weights and biases in the new model and 
        # copy them from the original model
        elif isinstance(m0, nn.BatchNorm2d):
            # Compute the list of indices of the remaining channels in the current BatchNorm2d layer
            # np.argwhere: return the indices of non-zero elements
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            # Resize the index list if it has only one element
            if idx1.size == 1:
                idx1 = np.resize(idx1,(1,))
            # Compute the weight of the current layer 
            # by copying only the weights of the remaining channels using the index list
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            # Compute the bias of the current layer 
            # by copying the bias values of the original layer and then cloned
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            # Compute the running mean of the current layer by 
            # copying the mean values of the original layer and then cloned
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            # Compute the running variance of the current layer by 
            # copying the variance values of the original layer and then cloned
            m1.running_var = m0.running_var[idx1.tolist()].clone()
            # Update the masks for the next pruned layer
            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                end_mask = cfg_mask[layer_id_in_cfg]
                
        # ============================ Linear Layers ============================
        # If the module is a Linear layer, 
        # compute the indices of the non-zero weights in the input channels and 
        # copy them from the original model
        elif isinstance(m0, nn.Linear):
            # Compute the list of indices of the remaining neurons/channels 
            # of the previous layer that connect to this current linear layer
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            # Resize the index list if it has only one element
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            # Compute the weight of the current layer 
            # by copying only the weights of the remaining channels of the previous layer 
            # using the index list
            m1.weight.data = m0.weight.data[:, idx0].clone()
            m1.bias.data   = m0.bias.data.clone()

    torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, os.path.join(args.save, 'pruned.pth'))

    print(newmodel)
    model = newmodel
    test(model)
