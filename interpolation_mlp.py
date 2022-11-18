import os
import torch
from torch import nn
from torchvision.datasets import CIFAR10
from torchvision import transforms
import matplotlib.pyplot as plt
from scipy import optimize
import scipy
import numpy as np

# Define MLP3 model
class MLP(nn.Module):
  '''
    Multilayer Perceptron.
  '''

  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Flatten(),
      nn.Linear(32 * 32 * 3, 512),
      nn.ReLU(),
      nn.Linear(512, 512),
      nn.ReLU(),
      nn.Linear(512, 512),
      nn.ReLU(),
      nn.Linear(512, 32),
      nn.ReLU(),
      nn.Linear(32, 10)
    )


  def forward(self, x):
    '''Forward pass'''
    return self.layers(x)
  

# Returns list of models. First list is model A, last is model B, with interpolated models in between.
def naive_interpolation(model_A, model_B, num_steps):

    list_of_interpolated_models = []
    for i in range(num_steps):
        delta = 1/num_steps * i 

        interpolated_model = MLP() # instantiate empty model
        
        # Loop to go through linear layers
        for layer_num in range(1,10, 2):
            model_A_weight = model_A.layers[layer_num].weight
            model_A_bias = model_A.layers[layer_num].bias

            model_B_weight = model_B.layers[layer_num].weight
            model_B_bias = model_B.layers[layer_num].bias

            interpolated_layer_weight = (model_A_weight * (1-delta) + model_B_weight * (delta)) # We are walking from model A to B, as delta start=0
            interpolated_layer_bias = (model_A_bias * (1-delta) + model_B_bias * (delta))

            interpolated_model.layers[layer_num].weight = torch.nn.Parameter(interpolated_layer_weight)
            interpolated_model.layers[layer_num].bias = torch.nn.Parameter(interpolated_layer_bias)

        list_of_interpolated_models.append(interpolated_model)
    
    list_of_interpolated_models.append(model_B) #Append model B at end

    return list_of_interpolated_models

def create_naive_plot(naive_list):
    torch.manual_seed(42)
    # Prepare CIFAR-10 dataset
    dataset = CIFAR10(os.getcwd(), download=True, transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=9000, shuffle=False, num_workers=1) # Take whole dataset in one batch

    loss_function = nn.CrossEntropyLoss()
    loss_list = []
    for naive_model in naive_list:
        for data in trainloader: # This loop should only go once, I know this is non-ideal
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


# Registers activations and returns for all layers in model as a dictionary of Z matrices
def get_Z_for_each_layer(model, dataset):
    
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=1) # Use 10 datapoints

    # Get data of batch size
    for data in trainloader: 
        inputs, _ = data
        break
    
    activation = {}
    # This function adds output from a layer to the activation dictionary
    def get_activation(name):
        def hook(model, input, output): # This function must have this structure, due to PyTorch "register_forward_hook" function
            activation[name] = torch.t(output.detach())  # torch.t = transpose, to get correct format
        return hook
    
    # Loop to go through linear layers, and register Z for each layer
    for layer_num in range(1,10, 2):
        model.layers[layer_num].register_forward_hook(get_activation(str(layer_num))) # "1" signifies the first layer
        output = model(inputs)
        # print(activation[str(layer_num)])
        # print(activation[str(layer_num)].shape)
    
    return activation


# Function for running activation matching. Returns new model pi(theta), that is, a new modified model_B which should
# be in the same basin as A.
def activation_matching_interpolation(model_A, model_B, num_steps, dataset):

    Z_dict_model_A = get_Z_for_each_layer(model_A, dataset)
    Z_dict_model_B = get_Z_for_each_layer(model_B, dataset)

    P_l_for_all_layers = {}

    # P_l has dimensions d*d, "d" is the dimension of the layer
    # Calculate all P_l values
    for layer_num in range(1,10, 2):
        # Use Hungarian method to get P_l
        Z_l = np.matmul(Z_dict_model_A[str(layer_num)], torch.t(Z_dict_model_B[str(layer_num)]))
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(torch.t(Z_l), maximize = True) # linear_sum_assignment calculates an X matrix with either 0 or 1

        P_l = np.zeros([len(row_ind), len(row_ind)]) # d*d matrix
        for row in row_ind:
            P_l[row, col_ind[row]] = 1
        P_l = P_l.T # transpose P_l back
        P_l_for_all_layers[layer_num] = P_l


    # Now we calculate W' and b' by modifying model_B
    model_B_weight = model_B.layers[layer_num].weight
    model_B_bias = model_B.layers[layer_num].bias

    for layer_num in range(1,10, 2):
        if layer_num == 1: # at layer=1 we will not multiply by P_{l-1}^T

    
    # print(type(col_ind))
    # print(col_ind.shape)
    # print("Z shape: ", Z_dict_model_A["1"].shape)

    # Create new model is the same basin as model_A
    pass


# Based on the algorithm described in the paper
def permutation_coordinate_descent(model_weights_A, model_weights_B):
    # Initialize pi_dict
    pi = {}
    for layer_num, weights in model_weights_A.items():
        pi[layer_num] = torch.eye(len(weights)) # The dimensions of P_l is decided by the first dim of the weights (num "features")
    
    layer_list = [x for x in range(1,10, 2)]
    converged = False
    while not converged:
        for layer in np.random.permutation(layer_list):
            print(layer)
        


def matching_weights_interpolation(model_A, model_B, num_steps):
    weights_model_A = {}
    weights_model_B = {}

    biases_model_A = {}
    biases_model_B = {}

    # Loop to go through linear layers
    for layer_num in range(1,10, 2):
        weights_model_A[layer_num] = model_A.layers[layer_num].weight
        weights_model_B[layer_num] = model_B.layers[layer_num].weight

        biases_model_A[layer_num] = model_A.layers[layer_num].bias
        biases_model_B[layer_num] = model_B.layers[layer_num].bias

    permutation_coordinate_descent(weights_model_A, weights_model_B)

    


if __name__ == "__main__":
    model_A = torch.load("mlp_model.pth")
    model_B = torch.load("mlp_model_second.pth")

    dataset = CIFAR10(os.getcwd(), download=True, transform=transforms.ToTensor())
    
    ## CREATE WEIGHT MATCHING
    # matching_weights_interpolation(model_A, model_B, 10)
    

    ## CREATE ACTIVATION MATCHING
    activation_matching_interpolation(model_A, model_B, 10, dataset)

    

    ## CREATE NAIVE PLOT
    #naively_interpolated_model_list = naive_interpolation(model_A, model_B, 7)
    #create_naive_plot(naively_interpolated_model_list)


    # print(model_A.layers[1].weight)
    # print(model_B.layers[1].weight)

    # print(naive_interpolation(model_A, model_B, 10))
    

    # print(naively_interpolated_model_list[0].layers[1].weight)
    # print(naively_interpolated_model_list[5].layers[1].weight)
    # print(naively_interpolated_model_list[10].layers[1].weight)

    # Evaluate models in list
    


    



    
