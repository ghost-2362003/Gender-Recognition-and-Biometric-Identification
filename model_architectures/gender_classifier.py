from model_architectures.stream1 import FirstStream
from model_architectures.stream2 import SecondStream
from model_architectures.TwoStreamNet import TwoStreamNet
from pre_processing.img_converter import createHighFrequencyComponent, createLowFrequencyComponent
import torch
import numpy as np
import pandas as pd
import cv2
from torchvision.datasets import ImageFolder
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path

path = Path()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# create the stream1 and stream2 model objects and load model file
stream_1 = FirstStream()
stream_2 = SecondStream()
stream_1.load_state_dict(torch.load('G:/11k_hands/model_files/stream1_model.pth',
                                    map_location=device
                                    ), 
                         strict=False)
stream_2.load_state_dict(torch.load('G:/11k_hands/model_files/stream2_model.pth', 
                                    map_location=device
                                    ), 
                         strict=False)

classifier = TwoStreamNet(stream_1, stream_2).to(device)
classifier.load_state_dict(torch.load('G:/11k_hands/model_files/joint_model.pth', 
                                      map_location=device
                                      ), 
                           strict=False)

# create the dataloader
BATCH_SIZE = 1
data_root = 'G:/11k_hands/dataset/train'
test_root = Path("G:/11k_hands/dataset/test")
vectors = []
labels = []

# override the ImageFolder to include the custom function
class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root=root, transform=None)  # disable transform for now
        self.base_transform = transform  # keep the transform pipeline without the custom fn

    def __getitem__(self, index):
        path, target = self.samples[index]

        blurred_img = createLowFrequencyComponent(path)
        blurred_img = (blurred_img - blurred_img.min())/(blurred_img.max() - blurred_img.min())
        blurred_img = torch.from_numpy(blurred_img).permute(2, 0, 1).float()

        detailed_img = createHighFrequencyComponent(path)
        detailed_img = cv2.resize(detailed_img, (224, 224))
        detailed_img = np.expand_dims(detailed_img, axis=0)  # shape: (1, 224, 224)
        detailed_img = torch.from_numpy(detailed_img).float()
        
        if self.base_transform is not None:
            blurred_img = self.base_transform(blurred_img)
            detailed_img = self.base_transform(detailed_img)

        return blurred_img, detailed_img, target

base_transform = transforms.Compose([
    transforms.Resize((224, 224))
])

dataset = CustomImageFolder(root=data_root, transform=base_transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# creating the feature file for storing the outputs 
classifier.eval()
loop = tqdm(dataloader, total=len(dataloader))

for (blurred_img, detailed_img, label) in loop:
    blurred_img, detailed_img, label = blurred_img.to(device), detailed_img.to(device), label.to(device)

    f1, f2, x = classifier(blurred_img, detailed_img)
    concatenated_feature = torch.concat([f1, f2], dim=1)
    concatenated_feature = torch.concat([concatenated_feature, x], dim=1)
    
    vectors.append(concatenated_feature.cpu().detach().numpy())
    labels.append(label.cpu().detach().numpy())

# save the features and labels to a file
df = pd.DataFrame(vectors)
df['Label'] = labels

df.to_csv('G:/11k_hands/features.csv', index=False)
