import torch
from torch import nn
import torchvision
from torchvision.models.alexnet import AlexNet_Weights

class FirstStream(nn.Module):
    def __init__(self):
        super(FirstStream, self).__init__()
        
        # Load pretrained AlexNet
        alexnet = torchvision.models.alexnet(pretrained=AlexNet_Weights.DEFAULT)
        
        # Use AlexNet features (conv1 to conv5)
        self.features = alexnet.features  # Conv layers

        # Use AlexNet fc6 and fc7
        self.fc6 = alexnet.classifier[0]  # Linear(9216, 4096)
        self.relu6 = alexnet.classifier[1]
        self.dropout6 = alexnet.classifier[2]

        self.fc7 = alexnet.classifier[3]  # Linear(4096, 4096)
        self.relu7 = alexnet.classifier[4]
        self.dropout7 = alexnet.classifier[5]

        # Custom fc8 and fc9 layers
        self.fc8 = nn.Linear(4096, 2048)
        self.relu8 = nn.ReLU()
        self.dropout8 = nn.Dropout(p=0.5)

        self.fc9 = nn.Linear(2048, 531)
       # self.relu9 = nn.ReLU()
       # self.dropout9 = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.features(x)              # conv1–conv5
        x = torch.flatten(x, 1)           # Flatten to (B, 9216)
        
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
       # x = self.relu9(x)
       # x = self.dropout9(x)

        return x