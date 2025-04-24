#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# # ## Introduction
# 
# In this notebook, I implement the training process for a binary classifier that detects signs of pneumonia in chest X-ray images. The dataset used comes from the RSNA Pneumonia Detection Challenge, and the model is built using PyTorch Lightning for efficient experimentation.

# ## Imports
#  
#  - `torch` and `torchvision` for model definition and data loading
#  - `torchvision.transforms` for preprocessing and data augmentation
#  - `torchmetrics` to calculate performance metrics
#  - `pytorch_lightning` for training and experiment management
#  - `ModelCheckpoint` and `TensorBoardLogger` to save models and track metrics

# In[ ]:





# In[ ]:





# In[3]:


import torch
import torchvision
from torchvision import transforms
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm.notebook import tqdm
import numpy as np
import matplotlib.pyplot as plt


# We first create the dataset. `torchvision.datasets.DatasetFolder` simplifies this by automatically associating directory names with class labels. All we need to provide is a loader function for our `.npy` images.

# In[ ]:





# In[4]:


def load_file(path):
    return np.load(path).astype(np.float32)


# In[ ]:





# In[6]:


train_transforms = transforms.Compose([
                                    transforms.ToTensor(),  # Convert numpy array to tensor
                                    transforms.Normalize(0.49, 0.248),  # Use mean and std from preprocessing notebook
                                    transforms.RandomAffine( # Data Augmentation
                                        degrees=(-5, 5), translate=(0, 0.05), scale=(0.9, 1.1)),
                                    transforms.RandomResizedCrop((224, 224), scale=(0.35, 1))])


# In[ ]:





# In[7]:


val_transforms = transforms.Compose([
                                    transforms.ToTensor(),  # Convert numpy array to tensor
                                    transforms.Normalize([0.49], [0.248]),  # Use mean and std from preprocessing notebook
])


# We now instantiate both the training and validation datasets and their corresponding data loaders. Adjust `batch_size` and `num_workers` depending on your system's capabilities.

# In[ ]:





# In[8]:


train_dataset = torchvision.datasets.DatasetFolder(
    "Processed/train/",
    loader=load_file, extensions="npy", transform=train_transforms)


# In[9]:


val_dataset = torchvision.datasets.DatasetFolder(
    "Processed/val/",
    loader=load_file, extensions="npy", transform=val_transforms)


# Let's take a look at some augmented training samples to verify the transforms.

# In[ ]:





# In[10]:


fig, axis = plt.subplots(2, 2, figsize=(9, 9))
for i in range(2):
    for j in range(2):
        random_index = np.random.randint(0, 20000)
        x_ray, label = train_dataset[random_index]
        axis[i][j].imshow(x_ray[0], cmap="bone")
        axis[i][j].set_title(f"Label:{label}")


# In[ ]:





# In[11]:


import os


# In[12]:


import os
import sys


# In[13]:


def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        return shell == "ZMQInteractiveShell"
    except NameError:
        return False


# In[14]:


# Use more workers if safe (e.g. script, not notebook)
if os.name == 'nt' and not is_notebook():
    num_workers = min(8, os.cpu_count())  # Use up to 8 workers safely
    persistent_workers = True
else:
    num_workers = 0
    persistent_workers = False


# In[15]:


print(f"Using num_workers = {num_workers}")


# In[16]:


batch_size = 64  # Puedes ajustar según tu GPU o RAM


# In[17]:


train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=True,
    pin_memory=True,
    persistent_workers=persistent_workers
)


# In[18]:


val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=False,
    pin_memory=True,
    persistent_workers=persistent_workers
)


# In[19]:


print(f"Using num_workers = {num_workers}")
print(f"There are {len(train_dataset)} train images and {len(val_dataset)} val images")


# This dataset is imbalanced—there are more healthy samples than pneumonia cases. Some ways to address this include:
# - Using a weighted loss function
# - Oversampling the minority class
# - Proceeding without adjustment
#  
# For simplicity, I continue without rebalancing. This often provides competitive results, but other strategies can be explored.

# In[ ]:





# In[20]:


np.unique(train_dataset.targets, return_counts=True), np.unique(val_dataset.targets, return_counts=True)


# ## Model Definition with PyTorch Lightning
# 
# A PyTorch Lightning model requires:
# - An `__init__` method to define layers
# - A `forward()` method for predictions
# - A `training_step()` to compute loss
# - `configure_optimizers()` to define the optimizer
#  
# Optionally, we can also implement validation/test logic.

# Now I define the model architecture. I use ResNet18 from `torchvision.models`, modifying the first convolutional layer to accept a single-channel input since X-ray images are grayscale.

# ### Optimizer and Loss Function
# 
# I use the Adam optimizer with a learning rate of 0.0001, and `BCEWithLogitsLoss` as the loss function. This loss function internally applies the sigmoid activation, making it ideal for binary classification.

# In[ ]:





# In[37]:


class PneumoniaModel(pl.LightningModule):
    def __init__(self, weight=1):
        super().__init__()
        
        self.model = torchvision.models.resnet18()
        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = torch.nn.Linear(in_features=512, out_features=1)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight], device="cuda"))

        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

        # Métricas por epoch para guardar
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def forward(self, data):
        return self.model(data)
    
    def training_step(self, batch, batch_idx):
        x_ray, label = batch
        label = label.float()
        pred = self(x_ray)[:, 0]
        loss = self.loss_fn(pred, label)
        acc = self.train_acc(torch.sigmoid(pred), label.int())

        self.log("train_loss", loss)
        self.log("step_train_acc", acc)
        return {"loss": loss, "acc": acc}
    
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['acc'] for x in outputs]).mean()

        self.log("train_acc", avg_acc, prog_bar=True)
        self.log("train_loss", avg_loss, prog_bar=True)

        self.train_losses.append(avg_loss.item())
        self.train_accuracies.append(avg_acc.item())

    def validation_step(self, batch, batch_idx):
        x_ray, label = batch
        label = label.float()
        pred = self(x_ray)[:, 0]
        loss = self.loss_fn(pred, label)
        acc = self.val_acc(torch.sigmoid(pred), label.int())

        self.log("val_loss", loss)
        self.log("step_val_acc", acc)
        return {"val_loss": loss, "val_acc": acc}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        self.log("val_acc", avg_acc, prog_bar=True)
        self.log("val_loss", avg_loss, prog_bar=True)

        self.val_losses.append(avg_loss.item())
        self.val_accuracies.append(avg_acc.item())

    def configure_optimizers(self):
        return [self.optimizer]

    def save_metrics(self, filename="training_metrics.csv"):
    	import pandas as pd
    	length = min(len(self.train_losses), len(self.train_accuracies),
                 len(self.val_losses), len(self.val_accuracies))

    	df = pd.DataFrame({
        	"epoch": list(range(length)),
        	"train_loss": self.train_losses[:length],
        	"train_acc": self.train_accuracies[:length],
        	"val_loss": self.val_losses[:length],
        	"val_acc": self.val_accuracies[:length]
    		})
    	df.to_csv(filename, index=False)


# In[ ]:





# In[38]:


model = PneumoniaModel().to("cuda")  # Instanciate the model


# Here, I define a checkpoint callback to save the top 10 models during training based on validation accuracy.

# In[39]:


# Create the checkpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor='val_acc',  # <--- corregido
    save_top_k=10,
    mode='max',
    dirpath='weights/',
    filename='-{epoch:02d}-{val_acc:.2f}'
)


# You can refer to the PyTorch Lightning Trainer documentation for more configuration options: https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html

# Create the trainer
# Change the gpus parameter to the number of available gpus on your system. Use 0 for CPU training

# In[40]:


trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        logger=TensorBoardLogger(save_dir="./logs"),
        log_every_n_steps=1,
        callbacks=[checkpoint_callback],
        max_epochs=35
)


# In[ ]:

if __name__ == '__main__':
	trainer.fit(model, train_loader, val_loader)
	model.save_metrics("metrics.csv")
	torch.save(model.model.state_dict(), "resnet18_pneumonia.pth")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




