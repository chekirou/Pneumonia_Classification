# Pneumonia_prediction 
Classifying Pneumonia CXR.
## Models Tested : 
- `Resnet 18` : Best single model 79% ROC AUC
- `Resnet 50` : 76.84% ROC AUC
- `Inception V3` : 78.64% ROC AUC
- `Vision Transformer` : 72.51% ROC AUC
- Weighted ensemble of the 5 models : 80.34% ROC AUC

## Trainning

Execute scripts `train_[Model Name].py` and change the data repository path .

## Visualizing Attention weights: 

function `visualize` used with the `Visualize_attentions.py`
## Data 
RSNA Pneumonia detection challenge : https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data
