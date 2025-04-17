import os
import torch
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
import torchvision.transforms.functional as tf
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from wc import *

import matplotlib.pyplot as plt

from model import *
from Dataloader import *
from unet_loss import *

from tqdm import tqdm

transform = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])

dataset= SegmentationDataset(
    data_dir=r'PATH', #train data path
    transform=transform
)

transform_val = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])

val_dataset = SegmentationDataset(data_dir=r'PATH', #val data [ath
    transform=transform_val)

BATCH_SIZE = 6
Data = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
Data_val = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

EPOCH = 30
LR = 1e-3

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

min_loss = float('inf')

model = Unet()
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCH):
    running_loss = 0.0
    for data in tqdm(Data, leave=False):
        input, label = data['input'], data['label']
        input = input.float()

        wc = compute_inverse_frequencies(label)

        y_pred = model(input)
        label = label.reshape(BATCH_SIZE,512,512)
        label_crop = tf.center_crop(label, [388,388])

        optimizer.zero_grad()
        criterion = weighted_cross_loss(y_pred, label_crop.long(), wc = wc)
        criterion.backward()
        optimizer.step()
        running_loss += criterion.item()

    val_loss = 0.0
    with torch.no_grad():
        for val in tqdm(Data_val, leave=False):
            input_val, label_val = data['input'], data['label']
            input_val = input_val.float()

            wc_val = compute_inverse_frequencies(label_val)

            y_pred_val = model(input_val)
            label_val = label_val.reshape(BATCH_SIZE, 512, 512)
            label_crop_val = tf.center_crop(label_val, [388, 388])
            criterion_val = weighted_cross_loss(y_pred_val, label_crop_val.long(), wc=wc_val)
            val_loss += criterion_val.item()

    if val_loss < min_loss:
        min_loss = val_loss
        torch.save(model.state_dict(), r'PATH') #model save path

    predicted_labels = torch.argmax(y_pred_val[0], dim=0)
    segmentation_map = predicted_labels.squeeze(0).cpu().numpy()

    plt.figure(figsize=(8, 8))
    plt.imshow(segmentation_map, cmap='gray')
    plt.title("Segmentation Map")
    plt.axis('off')
    plt.show()

    print(f"{epoch+1}/{EPOCH} Train Loss: {running_loss/len(Data):.4f}, val Loss: {val_loss/len(Data_val):.4f}")

#epoch : 23
#epoch : 29
#epoch : 27