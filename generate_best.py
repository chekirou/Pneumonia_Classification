# script to generate csv prediction file on the test set

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pydicom import dcmread
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import pdb



import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils import data
from models.ViT import ViT

label_data = pd.read_csv('/tempory/rsna_data/stage_2_train_images/stage_2_train_labels.csv')
columns = ['patientId', 'Target']

label_data = label_data.filter(columns).drop_duplicates()


train_labels, val_labels = train_test_split(label_data.values, test_size=0.1, random_state=0)

train_f = '/tempory/rsna_data/stage_2_train_images'
test_f = '/tempory/rsna_data/stage_2_test_images'

val_paths = [os.path.join(train_f, image[0]) for image in val_labels]

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
#, normalize
transform = transforms.Compose([
    transforms.Resize(512),
    transforms.ToTensor()])

class Dataset(data.Dataset):
    
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform
    
    def __getitem__(self, index):
        
        image = dcmread(f'{self.paths[index]}.dcm')
        image = image.pixel_array
        image = image / 255.0

        image = (255*image).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(image)

        label = self.labels[index][1]
        patientId = self.labels[index][0]
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label, patientId
    
    def __len__(self):
        
        return len(self.paths)
    

val_dataset = Dataset(val_paths, val_labels, transform=transform)
val_loader = data.DataLoader(dataset=val_dataset, batch_size=8, shuffle=False)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load vision transformer
# change this step for loading other models

model = ViT(
    image_size = 512,
    patch_size = 32,
    num_classes = 2,
    dim = 1024,
    depth = 6,
    heads = 8,
    channels = 1,
    mlp_dim = 1024,
    dropout = 0.1,
    emb_dropout = 0.1,
    pool="mean"
)
model.load_state_dict(torch.load("/tempory/rsna_checkpoints/ViT_best.pth"))
model.eval()

model.to(device)

best = 0

print(model)


results = pd.DataFrame(data= val_labels, columns= ['patientId', 'Target'] )
results.set_index('patientId', inplace=True)
results["predicted"] = 0


correct = 0
total = 0  
for images, labels, patientId in tqdm(val_loader):
    images = images.to(device)
    labels = labels.to(device)
    predictions = model(images)
    _, predicted = torch.max(predictions, 1)
    total += labels.size(0)
    correct += (labels == predicted).sum().item()
    for i in range(labels.size(0)):
        results.at[patientId[i], "predicted"] = predicted[i].item()
    del images
    del labels
    del patientId
    torch.cuda.empty_cache()
    
    
print(f'Val_Acc: {100*correct/total}')
results.to_csv('best_predictions/ViT_best.csv')
