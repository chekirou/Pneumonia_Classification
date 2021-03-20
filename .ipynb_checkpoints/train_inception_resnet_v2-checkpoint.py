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

label_data = pd.read_csv('/tempory/rsna_data/stage_2_train_images/stage_2_train_labels.csv')
columns = ['patientId', 'Target']

label_data = label_data.filter(columns).drop_duplicates()


train_labels, val_labels = train_test_split(label_data.values, test_size=0.1, random_state=0)
print("shapes")
print(train_labels.shape)
print(val_labels.shape)

train_f = '/tempory/rsna_data/stage_2_train_images'
test_f = '/tempory/rsna_data/stage_2_test_images'

train_paths = [os.path.join(train_f, image[0]) for image in train_labels]
val_paths = [os.path.join(train_f, image[0]) for image in val_labels]


"""def imshow(num_to_show=9):
    
    plt.figure(figsize=(10,10))
    
    for i in range(num_to_show):
        plt.subplot(3, 3, i+1)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        
        img_dcm = dcmread(f'{train_paths[i+20]}.dcm')
        img_np = img_dcm.pixel_array
        plt.imshow(img_np, cmap=plt.cm.binary)
        plt.xlabel(train_labels[i+20][1])

imshow()"""

transform = transforms.Compose([
    transforms.Resize(299),
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
        image = Image.fromarray(image).convert('RGB')

        label = self.labels[index][1]
        patientId = self.labels[index][0]
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label, patientId
    
    def __len__(self):
        
        return len(self.paths)
    

train_dataset = Dataset(train_paths, train_labels, transform=transform)



train_dataset = Dataset(train_paths, train_labels, transform=transform)
val_dataset = Dataset(val_paths, val_labels, transform=transform)
train_loader = data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
val_loader = data.DataLoader(dataset=val_dataset, batch_size=16, shuffle=False)

"""batch = iter(train_loader)
images, labels = next(batch)"""

"""image_grid = torchvision.utils.make_grid(images[:4])
image_np = image_grid.numpy()
img = np.transpose(image_np, (1, 2, 0))
plt.imshow(img)"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torchvision.models.inception_v3(pretrained=True)

# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model.AuxLogits.fc = nn.Linear(768, 2)
model.fc = nn.Linear(2048, 2)

model.to(device)
class_weights = torch.FloatTensor([1.0, (label_data.Target == 0).sum()/(label_data.Target == 1).sum()]).cuda()

criterion = nn.CrossEntropyLoss(weight = class_weights)

# Observe that all parameters are being optimized
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
best = 0

print(model)
losses, accuracies = [], []
num_epochs = 20
# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    # Training step
    for i, (images, labels, patientId) in tqdm(enumerate(train_loader)):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs, aux_outputs = model(images)
        loss1 = criterion(outputs, labels)
        loss2 = criterion(aux_outputs, labels)
        loss = loss1 + 0.4*loss2
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if (i+1) % 2000 == 0:
            
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
        predictions, aux_outputs = model(images)
        _, predicted = torch.max(predictions, 1)
        total += labels.size(0)
        correct += (labels == predicted).sum().item()
        del images
        del labels
        del patientId
        del aux_outputs
        torch.cuda.empty_cache()
    print(f'Epoch: {epoch+1}/{num_epochs}, Val_Acc: {100*correct/total}')
    accuracies.append(100*correct/total)
    if 100*correct/total > best :
        best = 100*correct/total
        torch.save(model, f"/tempory/rsna_checkpoints/inception_v3_horizontal_flip.pth")
    
    
accuracies = np.array(accuracies)
losses = np.array(losses)
with open('accuracies/inception_v3_horizontal_flip.npy', 'wb') as f:
    np.save(f, accuracies)
with open('losses/inception_v3_horizontal_flip.npy', 'wb') as f:
    np.save(f, losses)

model.eval()

results = pd.DataFrame(data= label_data, columns= ['patientId', 'Target'] )
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
results.to_csv('predictions/inception_v3_horizontal_flip.csv')
