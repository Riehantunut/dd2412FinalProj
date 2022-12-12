import json
import os

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from seaborn.categorical import stripplot

sns.set_theme(style="whitegrid")

absolute_path = os.path.abspath(__file__)


data = []


# file_name = os.path.dirname(absolute_path) + \
#     '/data/mlp_cifar10_adam/run_'
# file_name = os.path.dirname(absolute_path) + \
#     '/data/mlp_cifar10_sgd/run_'
# file_name = os.path.dirname(absolute_path) + \
#     '/data/mlp_mnist_adam/run_'
file_name = os.path.dirname(absolute_path) + \
    '/data/mlp_mnist_sgd/run_'
go = True
i = 0

steps = 8


naive_loss_sum = np.array(steps*[0])
activation_loss_sum = np.array(steps*[0])
weight_loss_sum = np.array(steps*[0])

data = []

while go:
    plot_data_file = file_name + str(i) + '.json'
    try:
        with open(plot_data_file) as json_file:
            saved_data = json.load(json_file)
            for step, naive_loss, activation_loss, weight_loss in zip(saved_data["steps"], saved_data["naive_loss_list"], saved_data["activation_matching_loss_list"], saved_data["weight_matching_loss_list"]):
                data_point = {}
                data_point["step"] = step
                data_point["naive"] = naive_loss
                data_point["activation"] = activation_loss
                data_point["weight"] = weight_loss

                data.append(data_point)

            i += 1
    except:
        go = False


# Combine all this data into a Pandas dataframe and print to screen
columns = ["step", "naive", "activation", "weight"]
df = pd.DataFrame(data, columns=columns)

# print(df.describe())
# print(df)

plt.figure(1, figsize=(6, 6))
sns.lineplot(x="step", y="naive",
             ci=100,                 # confidence interval 100% => min-max bounds
             data=df,
             label="naive",
             )
sns.lineplot(x="step", y="activation",
             ci=100,                 # confidence interval 100% => min-max bounds
             data=df,
             label="activation",
             )
sns.lineplot(x="step", y="weight",
             ci=100,                 # confidence interval 100% => min-max bounds
             data=df,
             label="weight",
             )

plt.title(r'Comparison between activation matching and weight matching')
plt.xlabel(r'Interpolation step')
plt.ylabel(r'Loss')
plt.tight_layout()
# plt.savefig(os.path.dirname(absolute_path) +
#             '/plots/mlp_cifar10_adam.pdf', format='pdf')
# plt.savefig(os.path.dirname(absolute_path) +
#             '/plots/mlp_cifar10_sgd.pdf', format='pdf')
# plt.savefig(os.path.dirname(absolute_path) +
#             '/plots/mlp_mnist_adam.pdf', format='pdf')
plt.savefig(os.path.dirname(absolute_path) +
            '/plots/mlp_mnist_sgd.pdf', format='pdf')
plt.show()
