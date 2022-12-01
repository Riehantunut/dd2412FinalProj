import os
import torch
from torch import nn
from torchvision.datasets import CIFAR10
from torchvision import transforms
import matplotlib.pyplot as plt
import scipy
import numpy as np
from resnet20_training import *


# Returns list of models. First list is model A, last is model B, with interpolated models in between.
def naive_interpolation(model_A, model_B, num_steps):
    weights = []
    biases = []

    # Data can be accessed and changed for a ResNet if each arg is specified,
    # so for one weight the args can be ['['layer1', '0', 'conv1']
    for name, param in model_A.named_parameters():
        if 'weight' in name:
            s = name.replace('.weight', '')
            s = s.split('.')
            weights.append(s)
        if 'bias' in name:
            s = name.replace('.bias', '')
            s = s.split('.')
            biases.append(s)

    list_of_interpolated_models = []
    for i in range(num_steps):
        delta = 1/num_steps * i

        # instantiate empty model
        interpolated_model = ResNet(ResidualBlock, [3, 3, 3])

        # In order to access data / change data of ResNet we to acces each "_module"
        # Example: weight_of_specific_layer = model_A_w._modules['layer1']._modules['0']._modules['conv1'].weight.data
        for w in weights:
            model_A_w = model_A
            model_B_w = model_B
            interpolated_w = interpolated_model
            for arg in w:
                model_A_w = model_A_w._modules[arg]
                model_B_w = model_B_w._modules[arg]
                interpolated_w = interpolated_w._modules[arg]
            model_A_w = model_A_w.weight.data
            model_B_w = model_B_w.weight.data
            interpolated_w.weight.data = (
                (1-delta) * model_A_w) + ((delta) * model_B_w)

        for b in biases:
            model_A_b = model_A
            model_B_b = model_B
            interpolated_b = interpolated_model
            for arg in b:
                model_A_b = model_A_b._modules[arg]
                model_B_b = model_B_b._modules[arg]
                interpolated_b = interpolated_b._modules[arg]
            model_A_b = model_A_b.bias.data
            model_B_b = model_B_b.bias.data
            interpolated_b.bias.data = (
                (1-delta) * model_A_b) + ((delta) * model_B_b)

        list_of_interpolated_models.append(interpolated_model)

    list_of_interpolated_models.append(model_B)  # Append model B at end

    return list_of_interpolated_models


# Identic to the one in interpolation_mlp
def create_naive_plot(naive_list):
    torch.manual_seed(42)
    # Prepare CIFAR-10 dataset
    dataset = CIFAR10(os.getcwd(), download=True,
                      transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(
        dataset, batch_size=9000, shuffle=False, num_workers=1)  # Take whole dataset in one batch

    loss_function = nn.CrossEntropyLoss()
    loss_list = []
    for naive_model in naive_list:
        for data in trainloader:  # This loop should only go once, I know this is non-ideal
            inputs, targets = data
            outputs = naive_model(inputs)

            # Compute loss
            loss = loss_function(outputs, targets)
            print(float(loss))
            loss_list.append(float(loss))

            break

    plt.plot(loss_list)
    plt.show()
    print(loss_list)


# Based on the algorithm described in the paper
def permutation_coordinate_descent(model_weights_A, model_weights_B, layer_list):
    pi = {}  # What dimensions?

    # JUST PRINTING SOME SHAPES TO SEE WHAT HAPPENS:
    for i in range(len(layer_list)):
        # WE GET THE LAYER (ex: 'layer1.0.conv1')
        layer = layer_list[i]

        # IF WE ARE AT THE LAST LAYER, WE CANNOT GET next_layer
        if layer == layer_list[-1]:
            print('Last layer:', layer, model_weights_A[layer].shape)
        else:
            next_layer = layer_list[i+1]
            print(layer, model_weights_A[layer].shape,
                  model_weights_B[next_layer].shape)


def matching_weights_interpolation(model_A, model_B):
    weights_model_A = {}
    weights_model_B = {}
    layer_list = []

    # Iterates through all parameters in models A and B and stores the ones that include "weight" in respective dict
    # Example: "layer3.1.conv1.weight"
    for name, param in model_A.named_parameters():
        if 'weight' in name:
            layer_list.append(name)
            weights_model_A[name] = param.data

    for name, param in model_B.named_parameters():
        if 'weight' in name:
            weights_model_B[name] = param.data

    return permutation_coordinate_descent(weights_model_A, weights_model_B, layer_list)


if __name__ == "__main__":
    model_A = ResNet(ResidualBlock, [3, 3, 3])
    model_A.load_state_dict(torch.load("resnet20_model.pth"))
    model_B = ResNet(ResidualBlock, [3, 3, 3])
    model_B.load_state_dict(torch.load("resnet20_model_second.pth"))

    # CREATE NAIVE PLOT
    #interpolated_model_list = naive_interpolation(model_A, model_B, 7)
    # create_naive_plot(interpolated_model_list)

    # CREATE WEIGHT MATCHING
    pi_B = matching_weights_interpolation(model_A, model_B)
    #weights_interpolation_model_list = naive_interpolation(model_A, pi_B, 7)
    # create_naive_plot(weights_interpolation_model_list)
