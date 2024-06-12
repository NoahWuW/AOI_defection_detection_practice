import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets, models
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

from load_data import test_transforms_submit, load_dataset_submit
from model_setting import model_set

# Load test data
test_csv_path = "/content/drive/MyDrive/AOI/test/test.csv"
test_df = pd.read_csv(test_csv_path)

test_transform = test_transforms_submit()
test_loader = load_dataset_submit(test_transform)

# Define model
model = model_set()
# Load checkpoint
model_path = "/content/drive/MyDrive/AOI/checkpoint.pt"
model.load_state_dict(torch.load(model_path))
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
model.eval()

all_predictions = []

with torch.no_grad():
    for images, _ in test_loader:
        images = images.to('cuda' if torch.cuda.is_available() else 'cpu')
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_predictions.extend(preds.cpu().numpy())

# Convert the result to DataFrame
test_df['Label'] = all_predictions

output_csv_path = "/content/drive/MyDrive/AOI/test/test.csv"
test_df.to_csv(output_csv_path, index=False)