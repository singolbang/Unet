import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms
from Dataloader import *
from model import Unet
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_dir = r'PATH' #test data path
model_path = r'PATH' #model save path

transform = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])


test_dataset = SegmentationDataset(
    data_dir=test_dir,
    transform=transform
)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = Unet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

with torch.no_grad():
    batch = next(iter(test_loader))
    inp = batch['input'].to(device)
    lbl = batch['label']
    out = model(inp.float())
    if out.shape[1] == 1:
        pred = (torch.sigmoid(out) > 0.5).cpu().squeeze().numpy()
    else:
        pred = torch.argmax(out, dim=1).cpu().squeeze().numpy()
    inp_img = inp.cpu().squeeze().numpy()
    lbl_img = lbl.squeeze().numpy()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
if inp_img.ndim == 3:
    axes[0].imshow(inp_img.transpose(1, 2, 0))
else:
    axes[0].imshow(inp_img, cmap='gray')
axes[0].axis('off')
axes[1].imshow(lbl_img, cmap='gray')
axes[1].axis('off')
axes[2].imshow(pred, cmap='gray')
axes[2].axis('off')
plt.tight_layout()
plt.show()
