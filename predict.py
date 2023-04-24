import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import pickle
from io import BytesIO
from urllib import request
from PIL import Image
import numpy as np

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2



def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img

def datapreparation(image, transforms = None):

    image = np.array(image).astype('float32')
    image /= 255.0
    if transforms:
        image = transforms(image=image)['image']
        #image = torch.tensor(image, dtype = torch.float32)
        image = image.clone().detach()
    else:
        image = torch.tensor(image, dtype=torch.float32)
    return image


def prediction(model, image_url, scaler):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    image = download_image(image_url)
    image = prepare_image(image)
    image = datapreparation(image,transforms = val_transform).to(device)
    image = image.to(device)
    output = model(image.unsqueeze(0)).view(-1)
    output = scaler.inverse_transform(output.cpu().detach().numpy().reshape(1, -1))[0][0]
    return output

def load_utils(model_path, scaler_path):
    
    model = torch.load(model_path, map_location=torch.device('cpu'))
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

# image transformation
val_transform = A.Compose(
                            [
                                A.Resize(224, 224),
                                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                ToTensorV2(),
                            ]
                        )

# model_path = '/kaggle/working/model.pth'
# scaler_path = '/kaggle/working/scalel.pkl'

model_path = './model.pth'
scaler_path = './scaler.pkl'

model, scaler = load_utils(model_path, scaler_path) # loading model and scaler

part1 = 'https://plus.unsplash.com/premium_photo-1674574586052-6d0cee95c473?ixlib=rb-4.'
part2 = '0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1129&q=80'

image_url = part1 + part2
prediction(model, image_url, scaler)