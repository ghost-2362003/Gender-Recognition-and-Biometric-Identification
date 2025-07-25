import torch
from torch import nn

class modifiedTwoStreamNet(nn.Module):
    def __init__(self, FirstStream, SecondStream): 
        
        super(modifiedTwoStreamNet, self).__init__()
        self.stream1 = FirstStream
        self.stream2 = SecondStream
        
        self.sequential = nn.Sequential(
            nn.Linear(in_features=1062, out_features=1062),
            nn.Unflatten(1, (1, 1062)),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Flatten(), 
            nn.Linear(in_features=531, out_features=2),
            nn.Softmax(dim=1)
        )

        self.fusion = nn.Linear(in_features=1062, out_features=1062)

    def forward(self, blurred_img, detailed_img):
        f1 = self.stream1(blurred_img)
        f2 = self.stream2(detailed_img)

        x = torch.concat((f1, f2), dim=1)
        fusion_output = self.fusion(x)

        return f1, f2, fusion_output