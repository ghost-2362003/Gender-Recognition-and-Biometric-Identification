import torchvision
import torch.nn as nn
from torchvision.models.alexnet import AlexNet_Weights

# Load the pre-trained AlexNet model
model = torchvision.models.alexnet(weights=AlexNet_Weights.DEFAULT)        # weights=True is deprecated since torchvision 0.13

# Access the weights of the first convolutional layer
filters = model.features[0].weight.data

print("Shape of the first convolutional layer's weights:", filters.shape)
print("Number of filters in the first convolutional layer:", filters.shape[0])
print(filters[0, :, :, :])  # Print the weights of the first filter of first conv layer