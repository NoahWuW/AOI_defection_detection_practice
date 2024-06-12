import torch
import torch.nn as nn
import numpy as np
import time
import copy
from torchvision.models import mobilenet_v2

# Load the model(mobilenet_v2) and modify the classification layer
def model_set():
    model = mobilenet_v2(pretrained=True)
    # Modify 
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 6)

    return model


# early stop
class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

# Train the model and evaluate it on the validation set, save the best model weights
def train_model(device, model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, early_stopping=None):

    since = time.time() # Record training start time

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Perform training and evaluation respectively
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval() # Evaluation mode, there will be no calculation or modification behavior

            running_loss = 0.0 # Initialize batch loss
            running_corrects = 0 # Initialize accurate batch counting

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad() # Clear gradient

                # Whether to calculate the gradient 
                # According to the training or evaluation phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Perform backpropagation and parameter updates 
                    # During training
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics of batch loss and accuracy
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # Update learning rate
            if phase == 'train':
                scheduler.step()

            # Calculate the average loss and accuracy of each epoch
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Save the best model weights on the validation set
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            # Check if early stopping is required
            if phase == 'val' and early_stopping:
                early_stopping(epoch_loss, model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    # Load the best model weights
                    model.load_state_dict(best_model_wts)
                    return model

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    #Load the best model weights
    model.load_state_dict(best_model_wts)
    return model