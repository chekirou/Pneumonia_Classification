import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pydicom import dcmread
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import pdb
from sklearn.metrics import roc_auc_score


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

train_paths = [os.path.join(train_f, image[0]) for image in train_labels]
val_paths = [os.path.join(train_f, image[0]) for image in val_labels]



transform = transforms.Compose([
    transforms.Resize(512),
    transforms.ToTensor()])

transform_val = transforms.Compose([
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
    

train_dataset = Dataset(train_paths, train_labels, transform=transform)
val_dataset = Dataset(val_paths, val_labels, transform=transform_val)


train_loader = data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
val_loader = data.DataLoader(dataset=val_dataset, batch_size=16, shuffle=False)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


model.to(device)
class_weights = torch.FloatTensor([1.0, (label_data.Target == 0).sum()/(label_data.Target == 1).sum()]).cuda()

criterion = nn.CrossEntropyLoss(weight = class_weights)

# Observe that all parameters are being optimized
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

best = 0
print(model)
losses, accuracies = [], []
num_epochs = 20
# Train the model

results = pd.DataFrame(data= val_labels, columns= ['patientId', 'Target'] )
results.set_index('patientId', inplace=True)

total_step = len(train_loader)
for epoch in range(num_epochs):
    # Training step
    for i, (images, labels, patientId) in tqdm(enumerate(train_loader)):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if (i+1) % 100 == 0:
            
            print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}"
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
        
        del images
        del labels
        del patientId
        torch.cuda.empty_cache()

    # Validation step
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
        del predictions
        torch.cuda.empty_cache()
    AUC = roc_auc_score(results.Target.values.astype(int), results.predicted.values.astype(int))
    print(f'Epoch: {epoch+1}/{num_epochs}, Val_Acc: {100*correct/total}, Val_auc: {AUC}')
    accuracies.append(100*correct/total)
    if AUC > best :
        best = AUC
        torch.save(model.state_dict(), f"/tempory/rsna_checkpoints/ViT_best.pth")
    torch.save(model.state_dict(), f"/tempory/rsna_checkpoints/ViT_current.pth")
    
    
accuracies = np.array(accuracies)
losses = np.array(losses)
with open('accuracies/ViT_mean.npy', 'wb') as f:
    np.save(f, accuracies)
with open('losses/ViT_mean.npy', 'wb') as f:
    np.save(f, losses)

model.eval()

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
results.to_csv('predictions/ViT_mean.csv')
