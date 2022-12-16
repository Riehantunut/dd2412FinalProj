import os
import torch
from torch import nn
from torchvision.datasets import KMNIST
from torchvision import transforms
import matplotlib.pyplot as plt
from scipy import optimize
import scipy
import numpy as np
import json

torch.set_printoptions(precision=8)

# Define MLP3 model


class MLP(nn.Module):
    '''
      Multilayer Perceptron.
    '''

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)


NUMBER_OF_LAYERS = 14
# 1. CHeck the identity of P. DONE, YOU GET THE IDENTITY
# 2. Debug on 2 level network.


# Returns list of models. First list is model A, last is model B, with interpolated models in between.
def naive_interpolation(model_A, model_B, num_steps):

    list_of_interpolated_models = []
    for i in range(num_steps):
        delta = 1/num_steps * i

        interpolated_model = MLP()  # instantiate empty model

        # Loop to go through linear layers
        for layer_num in range(1, NUMBER_OF_LAYERS, 2):
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


def create_naive_plot(naive_list, need_dtype_change):
    torch.manual_seed(42)
    # Prepare KMNIST dataset
    dataset = KMNIST(os.getcwd(), download=True,
                     transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(
        dataset, batch_size=9000, shuffle=False, num_workers=1)  # Take whole dataset in one batch

    loss_function = nn.CrossEntropyLoss()
    loss_list = []
    for naive_model in naive_list:
        for data in trainloader:  # This loop should only go once, I know this is non-ideal
            inputs, targets = data
            if need_dtype_change:
                # convert to double when activation matching to solve dtype error
                outputs = naive_model(inputs.double())
            else:
                outputs = naive_model(inputs)

            # Compute loss
            loss = loss_function(outputs, targets)
            print(float(loss))
            loss_list.append(float(loss))

            break

    # plt.plot(loss_list)
    # plt.show()
    # print(loss_list)
    return loss_list


# Registers activations and returns for all layers in model as a dictionary of Z matrices
def get_Z_for_each_layer(model, dataset):

    trainloader = torch.utils.data.DataLoader(
        dataset, batch_size=1000, shuffle=False, num_workers=1)  # Use 10 datapoints

    # Get data of batch size
    for data in trainloader:
        inputs, _ = data
        break

    activation = {}
    # This function adds output from a layer to the activation dictionary

    def get_activation(name):
        # This function must have this structure, due to PyTorch "register_forward_hook" function
        def hook(model, input, output):
            # torch.t = transpose, to get correct format
            activation[name] = torch.t(output.detach())
        return hook

    # Loop to go through linear layers, and register Z for each layer
    for layer_num in range(1, NUMBER_OF_LAYERS, 2):
        model.layers[layer_num].register_forward_hook(
            get_activation(str(layer_num)))  # "1" signifies the first layer
        output = model(inputs)
        # print(model.layers[layer_num]) # Linear(in_features=3072, out_features=512, bias=True)
        # print(activation[str(layer_num)].shape)

    return activation


# Function for running activation matching. Returns new model pi(theta), that is, a new modified model_B which should
# be in the same basin as A.
def activation_matching_interpolation(model_A, model_B, num_steps, dataset):

    Z_dict_model_A = get_Z_for_each_layer(model_A, dataset)
    Z_dict_model_B = get_Z_for_each_layer(model_B, dataset)

    # print("layer 1")
    # print(Z_dict_model_A["1"])
    # print(Z_dict_model_A["1"].shape)

    # print("layer 3")
    # print(Z_dict_model_A["3"])
    # print(Z_dict_model_A["3"].shape)

    # print("layer 5")
    # print(Z_dict_model_A["5"])
    # print(Z_dict_model_A["5"].shape)

    P_l_for_all_layers = {}

    # P_l has dimensions d*d, "d" is the dimension of the layer
    # Calculate all P_l values
    for layer_num in range(1, NUMBER_OF_LAYERS, 2):
        # Use Hungarian method to get P_l
        Z_l = np.matmul(Z_dict_model_A[str(layer_num)], torch.t(
            Z_dict_model_B[str(layer_num)]))

        # print("SHAPE: ", Z_dict_model_A[str(layer_num)].shape) # MAKE SURE THIS IS d*N (d=input_dim*output_dim)

        # MANUALLY VERIFY THAT ROWS ARE SHUFFLED TO PERMUTE TOWARDS MODEL B
        # linear_sum_assignment calculates an X matrix with either 0 or 1
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(
            torch.t(Z_l), maximize=True)

        P_l = np.zeros([len(row_ind), len(row_ind)])  # d*d matrix
        for row in row_ind:
            P_l[row, col_ind[row]] = 1
        P_l = P_l.T  # transpose P_l back

        # P*Z_B is quite alike Z_A, linear_sum_assignment seems to work
        # print("Z_A: ", Z_dict_model_A["1"])
        # print("Z_B: ", Z_dict_model_B["1"])
        # print("P_l.shape", P_l.shape)
        # print("Z_dict_model_B[1].shape", Z_dict_model_B["1"].shape)
        # print("P*Z_B: ", np.matmul(P_l, Z_dict_model_B["1"]))
        # quit()

        # print("IDENTITY PLZ:", np.matmul(P_l, P_l.T)) # You get the identity matrix
        P_l_for_all_layers[layer_num] = torch.from_numpy(
            P_l)  # convert np matrix to torch matrix

    interpolated_model = MLP()  # instantiate empty model

    # Now we calculate W' and b' by modifying model_B
    for layer_num in range(1, NUMBER_OF_LAYERS, 2):
        model_B_weight = model_B.layers[layer_num].weight
        model_B_bias = model_B.layers[layer_num].bias

        # print("P_l_for_all_layers[layer_num]: ", P_l_for_all_layers[layer_num])
        # print("P_l_for_all_layers[layer_num].double(): ",  P_l_for_all_layers[layer_num].double())

        # print("model_B_weight: ", model_B_weight[0])
        # print("type(model_B_weight): ", type(model_B_weight[0]))

        model_B_weight = model_B_weight.detach().numpy()
        model_B_bias = model_B_bias.detach().numpy()

        if layer_num == 1:  # at layer=1 we will not multiply by P_{l-1}^T
            interpolated_layer_weight = np.matmul(
                P_l_for_all_layers[layer_num], model_B_weight)
            interpolated_layer_bias = np.matmul(
                P_l_for_all_layers[layer_num], model_B_bias)

            # print("interpolated_layer_weight \n ", interpolated_layer_weight)
            # print("model_B_weight \n ", model_B_weight[21:24])

            # print("interpolated_layer_weight \n ", interpolated_layer_weight.shape)
            # print("model_B_weight \n ", model_B_weight.shape)
            # quit()

        else:
            interpolated_layer_weight = np.matmul(
                np.matmul(P_l_for_all_layers[layer_num], model_B_weight),
                np.transpose(P_l_for_all_layers[layer_num-2]))
            interpolated_layer_bias = np.matmul(
                P_l_for_all_layers[layer_num], model_B_bias)

        interpolated_model.layers[layer_num].weight = torch.nn.Parameter(
            interpolated_layer_weight)
        interpolated_model.layers[layer_num].bias = torch.nn.Parameter(
            interpolated_layer_bias)

    # Create new model is the same basin as model_A
    return interpolated_model


# Based on the algorithm described in the paper
def permutation_coordinate_descent(model_weights_A, model_weights_B):
    # Initialize pi_dict
    pi = {}
    for layer_num, weights in model_weights_A.items():
        num_feat = len(weights)
        # The dimensions of P_l is decided by the first dim of the weights (num "features")
        pi[layer_num] = torch.eye(num_feat)

    layer_list = [x for x in range(1, NUMBER_OF_LAYERS, 2)]
    converged_list = [0] * NUMBER_OF_LAYERS
    iteration = 0
    max_iter = 300

    while iteration < max_iter:  # sum(converged_list) != len(layer_list)
        iteration += 1
        for layer in np.random.permutation(layer_list):

            if layer == layer_list[0]:
                WA_WB = torch.matmul(
                    model_weights_A[layer], torch.t(model_weights_B[layer]))
                WAT_P = torch.matmul(
                    torch.t(model_weights_A[layer + 2]), pi[layer + 2])
                WAT_P_WB = torch.matmul(WAT_P, model_weights_B[layer + 2])
                LAP = WA_WB + WAT_P_WB

            elif layer == layer_list[-1]:
                WA_P = torch.matmul(model_weights_A[layer], pi[layer - 2])
                WA_P_WBT = torch.matmul(WA_P, torch.t(model_weights_B[layer]))
                LAP = WA_P_WBT

            else:  # layers 3, 5 and 7
                WA_P = torch.matmul(model_weights_A[layer], pi[layer - 2])
                WA_P_WBT = torch.matmul(WA_P, torch.t(model_weights_B[layer]))
                WAT_P = torch.matmul(
                    torch.t(model_weights_A[layer + 2]), pi[layer + 2])
                WAT_P_WB = torch.matmul(WAT_P, model_weights_B[layer + 2])
                LAP = WA_P_WBT + WAT_P_WB

            n, _ = LAP.shape
            A = LAP.detach().numpy().T

            # row_ind, col_ind = scipy.optimize.linear_sum_assignment(LAP.detach().numpy(), maximize = True) # linear_sum_assignment calculates an X matrix with either 0 or 1
            ri, ci = scipy.optimize.linear_sum_assignment(A, maximize=True)

            pi_l = torch.zeros(pi[layer].shape)
            for row in range(len(pi[layer])):
                pi_l[row][ci[row]] = 1

            oldL = np.vdot(A, pi[layer])
            newL = np.vdot(A, pi_l)
            diff = newL - oldL
            if diff == 0:
                # We don't currently use the converged_list
                # only a set iteration
                converged_list[layer] = 1
            pi[layer] = torch.t(pi_l)
            # print(diff)

    interpolated_model = MLP()  # instantiate empty model

    # Now we calculate W' and b' by modifying model_B
    for layer_num in range(1, NUMBER_OF_LAYERS, 2):
        model_B_weight = model_B.layers[layer_num].weight
        model_B_bias = model_B.layers[layer_num].bias

        # print("P_l_for_all_layers[layer_num]: ", P_l_for_all_layers[layer_num])
        # print("P_l_for_all_layers[layer_num].double(): ",  P_l_for_all_layers[layer_num].double())

        # print("model_B_weight: ", model_B_weight[0])
        # print("type(model_B_weight): ", type(model_B_weight[0]))

        model_B_weight = model_B_weight.detach().numpy()
        model_B_bias = model_B_bias.detach().numpy()

        if layer_num == 1:  # at layer=1 we will not multiply by P_{l-1}^T
            interpolated_layer_weight = np.matmul(
                pi[layer_num], model_B_weight)
            interpolated_layer_bias = np.matmul(pi[layer_num], model_B_bias)
        else:
            interpolated_layer_weight = np.matmul(
                np.matmul(pi[layer_num], model_B_weight),
                np.transpose(pi[layer_num-2]))
            interpolated_layer_bias = np.matmul(pi[layer_num], model_B_bias)

        interpolated_model.layers[layer_num].weight = torch.nn.Parameter(
            interpolated_layer_weight)
        interpolated_model.layers[layer_num].bias = torch.nn.Parameter(
            interpolated_layer_bias)

    # Create new model is the same basin as model_A
    return interpolated_model, converged_list


def matching_weights_interpolation(model_A, model_B):
    weights_model_A = {}
    weights_model_B = {}

    biases_model_A = {}
    biases_model_B = {}

    # Loop to go through linear layers
    for layer_num in range(1, NUMBER_OF_LAYERS, 2):
        weights_model_A[layer_num] = model_A.layers[layer_num].weight
        weights_model_B[layer_num] = model_B.layers[layer_num].weight

        biases_model_A[layer_num] = model_A.layers[layer_num].bias
        biases_model_B[layer_num] = model_B.layers[layer_num].bias

    return permutation_coordinate_descent(weights_model_A, weights_model_B)


if __name__ == "__main__":
    print("INTERPOLATING: MLPs TRAINED ON KMNIST WITH ADAM")

    # 1,3 är på första fliken
    # 1,4 andra
    model_A = torch.load("models/mlp_KMNIST_adam_model_2.pth")
    model_B = torch.load("models/mlp_KMNIST_adam_model_4.pth")

    dataset = KMNIST(os.getcwd(), download=True,
                     transform=transforms.ToTensor())

    interpolation_steps = 14

    # CREATE WEIGHT MATCHING
    print("WEIGHT MATCHING")
    pi_B, converged_list = matching_weights_interpolation(model_A, model_B)
    while sum(converged_list) == 0:
        pi_B, converged_list = matching_weights_interpolation(model_A, model_B)

    weights_interpolation_model_list = naive_interpolation(
        model_A, pi_B, interpolation_steps)
    weight_matching_loss_list = create_naive_plot(
        weights_interpolation_model_list, False)

    # CREATE NAIVE PLOT
    print("NAIVE INTERPOLATION")
    naively_interpolated_model_list = naive_interpolation(
        model_A, model_B, interpolation_steps)
    naive_loss_list = create_naive_plot(naively_interpolated_model_list, False)

    # CREATE ACTIVATION MATCHING
    print("ACTIVATION MATCHING")
    activation_matching_model = activation_matching_interpolation(
        model_A, model_B, 10, dataset)

    activation_interpolated_model_list = naive_interpolation(
        model_A, activation_matching_model, interpolation_steps)
    activation_matching_loss_list = create_naive_plot(
        activation_interpolated_model_list, True)

    steps = list(range(interpolation_steps + 1))

    absolute_path = os.path.abspath(__file__)
    data_to_save = {}
    data_to_save['steps'] = steps
    data_to_save['naive_loss_list'] = naive_loss_list
    data_to_save['activation_matching_loss_list'] = activation_matching_loss_list
    data_to_save['weight_matching_loss_list'] = weight_matching_loss_list

    dir_name = os.path.dirname(absolute_path)
    file_name = dir_name + '/data/mlp_kmnist_adam/run_'

    go = True
    i = 0
    while go:
        try:
            f = open(file_name + str(i) + '.json')
            f.close()
            i += 1

        except IOError:
            with open(file_name + str(i) + '.json', 'w') as outfile:
                json.dump(data_to_save, outfile)
            go = False

    # # Save plot
    # plt.plot(naive_loss_list, label="naive")
    # plt.plot(activation_matching_loss_list, label="activation matching")
    # plt.plot(weight_matching_loss_list, label="weight matching")
    # plt.legend()
    # plt.ylabel("loss")
    # plt.xlabel("steps")
    # plt.title(
    #     "Interpolation between MLPs trained on CIFAR10 with the Adam optimizer")
    # plt.savefig("plots/interpolation_mlp_cifar10_adam_14_steps.pdf")
