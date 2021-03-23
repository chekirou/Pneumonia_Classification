import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pydicom import dcmread
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import pdb
import cv2



import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils import data
from models.ViT import ViT
from models.ViT import ATTN_weights

def visualize(im,logits, att_mat):
    

    att_mat = torch.stack(att_mat).squeeze(1).detach().cpu()

    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=1)

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1))
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

    # Attention from the output token to the input space.
    v = joint_attentions[-1]
    grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
    pdb.set_trace()
    mask = cv2.resize(mask / mask.max(), (im.shape[1], im.shape[2]))[..., np.newaxis]
    result = (mask * im.numpy().transpose(1,2,0)).astype("float64")
    pdb.set_trace()
    

    

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
val_loader = data.DataLoader(dataset=val_dataset, batch_size=1, shuffle=True)


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
model.load_state_dict(torch.load("/tempory/rsna_checkpoints/ViT_best.pth"))
model.eval()

model.to(device)


print(model)



correct = 0
total = 0  
for images, labels, patientId in tqdm(val_loader):
    images = images.to(device)
    labels = labels.to(device)
    predictions = model(images)
    _, predicted = torch.max(predictions, 1)
    print("Target")
    print(labels)
    print("Predicted")
    print(predicted)
    visualize(images.detach().cpu()[0],predictions.detach().cpu(),ATTN_weights[-1])
    pdb.set_trace()
    del images
    del labels
    del patientId
    torch.cuda.empty_cache()
    
    
print(f'Val_Acc: {100*correct/total}')
results.to_csv('best_predictions/ViT_best.csv')
