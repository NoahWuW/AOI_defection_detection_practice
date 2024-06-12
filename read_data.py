import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

# Read and split data
def read_and_split_data(labels_path, test_size=0.2):
    df = pd.read_csv(labels_path)
    train_df, val_df = train_test_split(df, test_size=test_size, stratify=df['Label'])

    return train_df, val_df

# Calculate the class weights of the training set ## for loss func
def cal_weights(train_df):
    train_class_weights = compute_class_weight('balanced', classes=np.unique(train_df['Label']), y=train_df['Label'])
    train_class_weights = {i: weight for i, weight in enumerate(train_class_weights)}

    return train_class_weights

# Adjust sampling weight ## for sample each epoch
def cal_sample_weights(train_df):
    class_sample_count = train_df['Label'].value_counts().sort_index()
    weights = 1. / torch.tensor(class_sample_count, dtype=torch.float)
    samples_weights = weights[train_df['Label'].values] # Efficient indexing, tensor->series
    sampler = WeightedRandomSampler(samples_weights, len(samples_weights)) 
    # The samples of each batch can be averaged, and the weight is proportional to the probability

    return sampler

