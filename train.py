import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from read_data import read_and_split_data, cal_weights, cal_sample_weights
from load_data import train_val_transforms, load_dataset
from model_setting import EarlyStopping, model_set, train_model

#  ------------------------------------------------
#Suppose you connect to the "train" folder
#  ------------------------------------------------

labels_path = "train.csv"
train_df, val_df = read_and_split_data(labels_path, test_size=0.2)

train_class_weights = cal_weights(train_df)

sampler = cal_sample_weights(train_df)
#  ------------------------------------------------

train_transform, val_transform = train_val_transforms()

train_loader, val_loader = load_dataset(train_df, val_df, train_transform, val_transform, sampler)
#  ------------------------------------------------

model = model_set()
#  ------------------------------------------------

# train_loader 和 val_loader
dataloaders = {
    'train': train_loader,
    'val': val_loader
}

# Load model and data to device (GPU or CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss function
criterion = nn.CrossEntropyLoss(weight=torch.tensor(list(train_class_weights.values()), dtype=torch.float).to(device)) # 前傳導，權重越大loss越少
# Optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)
# Adjustment learning rate
scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

early_stopping = EarlyStopping(patience=5, verbose=True)
# Training
model = train_model(device, model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, early_stopping=early_stopping)
