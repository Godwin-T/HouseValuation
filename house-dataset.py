# Importing libraries

from PIL import Image
import shutil
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import glob
from pathlib import Path
import os
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import copy
from io import BytesIO
from urllib import request

from sklearn.preprocessing import MinMaxScaler
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch
import torchvision

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler
np.random.seed(42)


import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

class HouseDataset(Dataset):
    def __init__(self, image_paths, targets, transforms=None):
        super().__init__()
        self.image_paths = image_paths
        self.targets = targets
        self.transforms = transforms
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        price = self.targets[index]
        
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        height, width = image.shape[:-1]
        if self.transforms:
            image = self.transforms(image=image)['image']
        else:
            image = torch.tensor(image, dtype=torch.float32)
        price = torch.tensor(price, dtype=torch.float32)
        return image, price
    def __len__(self):
        return len(self.image_paths)


def visualize(dataset, idx=0):
    dataset = copy.deepcopy(dataset)
    dataset.transforms = A.Compose([t for t in dataset.transforms if not isinstance(t, (ToTensorV2, A.Normalize))])
    figure, ax = plt.subplots(nrows=2, ncols=3, figsize=(10, 10))
    axes = ax.ravel()
    for i in range(6):
        image, target = dataset[i]
        axes[i].imshow(image)
        axes[i].set_title(f"price: {target}")
        axes[i].set_axis_off()
    plt.tight_layout()
    figure.savefig('sample.jpg')

def data_prep(imagepath, target):
    zipped = list(zip(imagepath, targets))
    n = int(0.7 * len(zipped))
    path_img, path_targets = zip(*np.random.permutation(zipped))

    train_img, val_img = np.array(list(path_img[:n])), np.array(list(path_img[n:]))
    train_targets, val_targets = np.array(list(path_targets[:n])), np.array(list(path_targets[n:]))

    scaler = MinMaxScaler()
    train_targets = scaler.fit_transform(train_targets.reshape(-1,1)).reshape(-1)
    val_targets = scaler.transform(val_targets.reshape(-1,1)).reshape(-1)

    # train_img, val_img = list(path_img[:n]), list(path_img[n:])
    # train_targets, val_targets = list(path_targets[:n]), list(path_targets[n:])
    
    train_dataset = HouseDataset(train_img, train_targets, train_transform)
    val_dataset = HouseDataset(val_img, val_targets, val_transform)
    len(train_dataset), len(val_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        pin_memory=True,
    )
    return (train_loader, val_loader, train_dataset,
            val_dataset, train_targets, val_targets, scaler)

def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler,scaler, num_epochs=25):
    model = model.to(device)
    for epoch in range(num_epochs):
        model.train()
        closs = 0
        for batch_idx, (image, price) in enumerate(train_dataloader):
            optimizer.zero_grad()
            image, price = image.to(device), price.to(device)
            output = model(image).view(-1)
            loss = criterion(output, price)
            #print("loss: ", loss.item())
            # print("output: ", output)
            # print("price: ", price)
            closs += loss.item()
            loss.backward()
            optimizer.step()
            #wandb.log({'batch train': batch_idx, 'batch loss: ':loss.item()})
        print(f"End of epoch {epoch}. Loss {closs/len(train_loader)}")
        #wandb.log({'epoch': epoch,'Loss epoch: ': closs/len(train_loader)})
        
        scheduler.step()
        cmae = 0
        model.eval()
        for batch_idx, (image, price) in enumerate(val_dataloader):
            image, price = image.to(device), price
            output = model(image)
            output = torch.tensor(scaler.inverse_transform(output.cpu().detach().numpy()), dtype=torch.float32)
            price = torch.tensor(scaler.inverse_transform(price.numpy().reshape(-1, 1)), dtype=torch.float32)
            
            # print("output ", output)
            # print("price ", price)
            mae = metric(output, price)
            #print(mae.item)
            cmae += mae.item()
            #wandb.log({'batch eval': batch_idx, 'batch MAE metric: ': mae.item()})
        print(f"MAE {cmae/len(val_dataloader)}")
       # wandb.log({'epoch': epoch, 'MAE epoch: ' : cmae/len(val_dataloader)})

def check(model, dataset, scaler):
    model.eval()
    dataset_tomodel = copy.deepcopy(dataset)
    dataset = copy.deepcopy(dataset)
    dataset.transforms = A.Compose([t for t in dataset.transforms if not isinstance(t, (ToTensorV2, A.Normalize))])
    print(dataset.transforms)
    figure, ax = plt.subplots(nrows=2, ncols=3, figsize=(10, 10))
    axes = ax.ravel()
    for i, n in zip(range(6),np.random.randint(len(dataset), size=6)):
        image, target = dataset_tomodel[n]
        print(image.shape)
        image = image.to(device)
        output = model(image.unsqueeze(0)).view(-1)
        output = scaler.inverse_transform(output.cpu().detach().numpy().reshape(1, -1))
        image, target = dataset[n]
        target = scaler.inverse_transform(target.cpu().detach().numpy().reshape(1, -1))
        axes[i].imshow(image)
        axes[i].set_title(f"predicted price: {output[0][0]}\n true price:{target[0][0]}")
        axes[i].set_axis_off()
    plt.tight_layout()

    # figure.savefig('sample.jpg')

# Image Transformation
train_transform = A.Compose(
    [
        A.Resize(224, 224),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, p=0.5),
        A.RGBShift(r_shift_limit=0.05, g_shift_limit=0.05, b_shift_limit=0.05, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

val_transform = A.Compose(
    [
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

# Loading data
house_info_string = glob.glob(r"../input/house-price/Houses Dataset/*.txt")[0]
with open(house_info_string, 'r') as f:
    info = f.readlines()

# Turn metadata to Dataframe
new_info = [i.split(' ') for i in info]
bedroom = [int(i[0]) for i in new_info]
bathroom = [float(i[1]) for i in new_info]
area = [float(i[2]) for i in new_info]
zipcode = [int(i[3]) for i in new_info]
price = [int(i[4].strip()) for i in new_info]

dict = {'bedroom': bedroom, 'bathroom':bathroom, 
        'area':area, 'zipcode':zipcode, 'price':price}
metadata = pd.DataFrame(dict)


houses_frontal = glob.glob(r"../input/house-price/Houses Dataset/*frontal.jpg")
houses_bedroom = glob.glob(r"../input/house-price/Houses Dataset/*bedroom.jpg")
houses_bathroom = glob.glob(r"../input/house-price/Houses Dataset/*bathroom.jpg")
houses_kitchen = glob.glob(r"../input/house-price/Houses Dataset/*kitchen.jpg")

houses_frontal.sort(key=natural_keys)
houses_bedroom.sort(key=natural_keys)
houses_bathroom.sort(key=natural_keys)
houses_kitchen.sort(key=natural_keys)

targets = metadata['price'].to_numpy()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Load model
model = torchvision.models.resnet50(pretrained=True)
in_f = model.fc.in_features
model.fc = torch.nn.Sequential(torch.nn.Linear(in_f, 1), torch.nn.Sigmoid())

# hyperparameter
criterion = torch.nn.MSELoss()
metric = torch.nn.L1Loss()

# Observe that all parameters are being optimized
optimizer_ft = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# load data
(train_loader, val_loader, train_dataset,
        val_dataset, train_target, val_target, scaler) = data_prep(houses_bedroom, targets)

for x in train_dataset:
    print(x[0].shape)
    break

# train model
train_model(model, train_loader, val_loader, criterion, 
            optimizer_ft, exp_lr_scheduler,scaler, num_epochs=25)

# test model
check(model, val_dataset, scaler)