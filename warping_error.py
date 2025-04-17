import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from Dataloader import *
from model import Unet
import random

from tqdm.auto import tqdm

def crop_center(img, crop_size):
    # img: (C, H, W)
    _, h, w = img.shape
    ch, cw = crop_size
    start_h = (h - ch) // 2
    start_w = (w - cw) // 2
    return img[:, start_h:start_h + ch, start_w:start_w + cw]


def compute_warping_error(pred, target):
    return torch.abs(pred.float() - target.float()).mean().item()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])

test_dir = r'PATH' #test data path
test_dataset = SegmentationDataset(
    data_dir=test_dir,
    transform=transform
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = Unet().to(device)
model.load_state_dict(torch.load(r'PATH', map_location=device)) #model save path
model.eval()

errors = []
results = []
with torch.no_grad():
    for batch in tqdm(test_loader):
        inp = batch['input'].to(device)  # (1, C, 572, 572)
        lbl = batch['label']  # (1, 1, 572, 572)
        out = model(inp.float())  # 출력 shape: (1, 2, 388, 388) 혹은 (1, 1, 388, 388)
        if out.shape[1] == 1:
            pred = (torch.sigmoid(out) > 0.5).long()  # (1, 1, 388, 388)
        else:
            pred = torch.argmax(out, dim=1, keepdim=True)  # (1, 1, 388, 388)
        # ground truth가 572×572라면, 중앙 388×388 영역 crop
        lbl_crop = crop_center(lbl[0], (388, 388))  # (1, 388, 388)
        pred_crop = pred[0].cpu()  # (1, 388, 388)
        error = compute_warping_error(pred_crop, lbl_crop)
        errors.append(error)
        results.append((inp[0].cpu(), lbl_crop.cpu(), pred_crop))

avg_error = np.mean(errors)
print("Average Warping Error:", avg_error)

inp_img, lbl_img, pred_img = random.choice(results)
inp_img = inp_img.numpy()
if inp_img.shape[0] == 1:
    inp_img = inp_img.squeeze(0)
else:
    inp_img = inp_img.transpose(1, 2, 0)
lbl_img = lbl_img.squeeze(0).numpy()
pred_img = pred_img.squeeze(0).numpy()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(inp_img, cmap='gray' if inp_img.ndim == 2 else None)
axes[0].set_title("Input (572x572)")
axes[0].axis('off')
axes[1].imshow(lbl_img, cmap='gray')
axes[1].set_title("Ground Truth (cropped 388x388)")
axes[1].axis('off')
axes[2].imshow(pred_img, cmap='gray')
axes[2].set_title("Prediction (388x388)")
axes[2].axis('off')
plt.tight_layout()
plt.show()
