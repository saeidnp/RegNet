# RegNet

In this work, we focus on regularizing a deep neural network by adding a prior to the loss function based on similarities among labels. Particularly, we consider an image classification task where very few training samples are available for some classes and, study and expand a work which adds a regularizer to push weight vectors corresponding to visually similar classes, to be similar.

## Getting Started

You can clone and run the code!

### Prerequisites

We have implemented the code in Jupyter Notebook using Pytorch. The code supports GPUs for training the network.

## Running the code

You need to run *RegNet.ipynb* which contains initialization, downloading the dataset, training the network and storing the trained model. Is also contains some evaluations on the training set.

### Arguments

In order to run different experiments, the only part of the code you need to change is in the first cell of *RegNet.ipynb*. You can change the following arguments:
- `exp_type` : Specifies the experimental settings which should be one of the following
    - null: using no hierarchy
    - fixed: using the fixed hierarchy provided with CIFAR100 dataset
    - dynamic_greedy: using the greedy method for updating the hierarchy
    - dynamic_kmeans: using the k-means method for updating the hierarchy
- `dataset_class_toremove` and `dataset_portion`: These are used to shrink the dataset. `dataset_portion` specifies portion to keep and `dataset_class_toremove` specifies which classes are affected, meaning the rest of the classes will be left untouched. `dataset_class_toremove` should be either set to "all" or a single class' name.
- `experiment_name`: Shows the name of the experiment. The value of this variable indicates the name of the directory in which the code stores all the outputs.

### Outputs
- This code will save the model every few epochs in tmp/models/ under the project directory. It also stores the final trained model in results/models. The same story goes with the losses; they go to tmp/losses and results/losses.

## Code structure

- *RegNet.ipynb*: main
- cifar100_data_loader: contains functions for loading the training set (downloads it if not available), retrieving the set of classes and superclasses, class names and the fixed hierarchy.
- commons.py: imports required libraries.
- const.py: defines a few number of constants used.
- data_collector.py: contains `DataCollector` class which is used for storing and loading models and losses.
- data_loader.py: a wrapper for cifar100_data_loader. It also handles dataset shrinking.
- evaluate.py: defines top-k and top-1 evaluator functions.
- loss_monitor.py: defines a `LossMonitor` class which is responsible for plotting and printing the loss during training process.
- net.py: defines the network model.
- plotter.py: defines `Plotter` class which initializes and updates a plot.
- trainer.py: defines `Trainer` class which handles the training process.
- utils.py: contains some misc functions.

## Authors

* [**Saeid Naderiparizi**](https://github.com/saeidnp)
* [**Setareh Cohan**](https://github.com/setarehc)

