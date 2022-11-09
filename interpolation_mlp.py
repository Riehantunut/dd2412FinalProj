import os
import torch
from torch import nn
from torchvision.datasets import CIFAR10
from torchvision import transforms
import matplotlib.pyplot as plt


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


def activation_matching_interpolation(model_A, model_B, num_steps, dataset_batch):
    pass


if __name__ == "__main__":
    model_A = torch.load("mlp_model.pth")
    model_B = torch.load("mlp_model_second.pth")

    # activation = {}
    # def get_activation(name):
    #     def hook(model, input, output):
    #         activation[name] = output.detach()
    #     return hook

    # model = MLP()
    # model.fc2.register_forward_hook(get_activation('fc2'))
    # x = torch.randn(1, 25)
    # output = model(x)
    # print(activation['fc2'])

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
    


    



    
