import os
import torch
from torch import nn
from torchvision.datasets import CIFAR10
from torchvision import transforms
import matplotlib.pyplot as plt
from resnet20_training import *


# Returns list of models. First list is model A, last is model B, with interpolated models in between.
def naive_interpolation(model_A, model_B, num_steps):

    list_of_interpolated_models = []
    for i in range(num_steps):
        delta = 1/num_steps * i

        # instantiate empty model
        interpolated_model = ResNet(ResidualBlock, [3, 3, 3])

        # Loop to go through linear layers
        for layer_num in range(1, 10, 2):
            model_A_weight = model_A.layers[layer_num].weight
            model_A_bias = model_A.layers[layer_num].bias

            model_B_weight = model_B.layers[layer_num].weight
            model_B_bias = model_B.layers[layer_num].bias

            # We are walking from model A to B, as delta start=0
            interpolated_layer_weight = (
                model_A_weight * (1-delta) + model_B_weight * (delta))
            interpolated_layer_bias = (
                model_A_bias * (1-delta) + model_B_bias * (delta))

            interpolated_model.layers[layer_num].weight = torch.nn.Parameter(
                interpolated_layer_weight)
            interpolated_model.layers[layer_num].bias = torch.nn.Parameter(
                interpolated_layer_bias)

        list_of_interpolated_models.append(interpolated_model)

    list_of_interpolated_models.append(model_B)  # Append model B at end

    return list_of_interpolated_models


if __name__ == "__main__":
    model_A = ResNet(ResidualBlock, [3, 3, 3])
    model_A.load_state_dict(torch.load("resnet20_model.pth"))
    model_B = ResNet(ResidualBlock, [3, 3, 3])
    model_B.load_state_dict(torch.load("resnet20_model_second.pth"))
    #model_A = torch.load_state_dict("resnet20_model.pth")
    #model_B = torch.load("resnet20_model_second.pth")

    # print(model_A.layers[1].weight)
    # print(model_B.layers[1].weight)

    # print(naive_interpolation(model_A, model_B, 10))
    naively_interpolated_model_list = naive_interpolation(model_A, model_B, 7)

    # print(naively_interpolated_model_list[0].layers[1].weight)
    # print(naively_interpolated_model_list[5].layers[1].weight)
    # print(naively_interpolated_model_list[10].layers[1].weight)

    # Evaluate models in list

    torch.manual_seed(42)
    # Prepare CIFAR-10 dataset
    dataset = CIFAR10(os.getcwd(), download=True,
                      transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(
        dataset, batch_size=9000, shuffle=False, num_workers=1)  # Take whole dataset in one batch

    loss_function = nn.CrossEntropyLoss()

    loss_list = []
    for naive_model in naively_interpolated_model_list:

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
