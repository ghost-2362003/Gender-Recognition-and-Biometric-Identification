import torch
from torch import nn
import torchvision
from torchvision.models.alexnet import AlexNet_Weights

model = torchvision.models.alexnet(weights = AlexNet_Weights.DEFAULT)
conv_1 = model.features[0]

# get the weights of the 1st conv layer
weights = conv_1.weight
num_filters = weights.shape[0]
num_color_channels = weights.shape[1]

# change the shape of the conv_1 layer
before_luma_weights = torch.zeros(64, 3, 121)
for i in range(num_filters):
    temp = weights[i].reshape(weights[i].size(0), -1)
    before_luma_weights[i] = temp

# compute the luma weights
luma_weights = torch.zeros((num_filters, 121, 1))       ## initalize the luma_weights
luma_components = torch.tensor([[0.2989, 0.578, 0.114]])  ## luma components for RGB to grayscale conversion

for i in range(num_filters):
    temp = before_luma_weights[i].T @ luma_components.T
    luma_weights[i] = temp
    
model.features[0].weight = torch.nn.Parameter(luma_weights.reshape(64, 1, 11, 11))      # set the new luma weights to the conv2d_1 layer

class SecondStream(nn.Module):
    def __init__(self):
        super(SecondStream, self).__init__()
        
        modified_alexnet = model
        
        self.features = modified_alexnet.features # conv layers
        
        # Use AlexNet fc6 and fc7
        self.fc6 = modified_alexnet.classifier[0]  # Linear(9216, 4096)
        self.relu6 = modified_alexnet.classifier[1]
        self.dropout6 = modified_alexnet.classifier[2]

        self.fc7 = modified_alexnet.classifier[3]  # Linear(4096, 4096)
        self.relu7 = modified_alexnet.classifier[4]
        self.dropout7 = modified_alexnet.classifier[5]
        
        self.fc8 = nn.Linear(4096, 2048)
        self.relu8 = nn.ReLU()
        self.dropout8 = nn.Dropout(p=0.5)
        
        self.fc9 = nn.Linear(2048, 2048)
        self.relu9 = nn.ReLU()
        self.dropout9 = nn.Dropout(p=0.5)
        
        self.fc10 = nn.Linear(2048, 531)
        
    def forward(self, x):
        x = self.features(x)
        
        x = torch.flatten(x, 1)  # Flatten to (B, 9216)
        
        x = self.fc6(x)
        x = self.relu6(x)
        x = self.dropout6(x)
        
        x = self.fc7(x)
        x = self.relu7(x)
        x = self.dropout7(x)
        
        x = self.fc8(x)
        x = self.relu8(x)
        x = self.dropout8(x)
        
        x = self.fc9(x)
        x = self.relu9(x)
        x = self.dropout9(x)
        
        x = self.fc10(x)
        
        return x