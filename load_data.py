import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd


# Because we are training our own data set, we need to customize the data set class
class CustomDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.dataframe.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        label = self.dataframe.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label


# Data enhancement and preprocessing ## for train
def train_val_transforms() :
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform

# Data Loader ## for train
def load_dataset(train_df, val_df, train_transform, val_transform, sampler):
    # Load imgs
    image_dir = '/content/drive/MyDrive/AOI/train/train_images'

    # Build dataset
    train_dataset = CustomDataset(dataframe=train_df, image_dir=image_dir, transform=train_transform)
    val_dataset = CustomDataset(dataframe=val_df, image_dir=image_dir, transform=val_transform)

    # Create data loader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=4, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    return train_loader, val_loader

# Data enhancement and preprocessing ## for submit
def test_transforms_submit() :
    test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return test_transform


# Data Loader ## for submit
def load_dataset_submit(test_df, test_transform):
    # Load imgs
    data_path = "/content/drive/MyDrive/AOI/test/test_images"

    # Build dataset
    test_dataset = CustomDataset(dataframe=test_df, image_dir=data_path, transform=test_transform)

    # Create data loader
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return test_loader