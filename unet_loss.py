import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.io import imshow
from skimage.measure import label
from scipy.ndimage.morphology import distance_transform_edt
import numpy as np


def unet_weight_map(y, wc=None, w0 = 10, sigma = 5):

    labels = label(y)
    no_labels = labels == 0
    label_ids = sorted(np.unique(labels))[1:]

    if len(label_ids) > 1:
        distances = np.zeros((y.shape[0], y.shape[1], len(label_ids)))

        for i, label_id in enumerate(label_ids):
            distances[:,:,i] = distance_transform_edt(labels != label_id)

        distances = np.sort(distances, axis=2)
        d1 = distances[:,:,0]
        d2 = distances[:,:,1]
        w = w0 * np.exp(-1/2*((d1 + d2) / sigma)**2) * no_labels
    else:
        w = np.zeros_like(y)
    if wc:
        class_weights = np.zeros_like(y)
        for k, v in wc.items():
            class_weights[y == k] = v
        w = w + class_weights
    return w

def weighted_cross_loss(y_pred, y_true, wc=None, w0=10, sigma=5):
    losses = []
    cross_fn = nn.CrossEntropyLoss(reduction='none')
    batch_size = y_true.shape[0]

    for i in range(batch_size):
        if y_true.dim() == 4 and y_true.size(1) == 1:
            y_true_instance = y_true[i, 0]
        else:
            y_true_instance = y_true[i]

        y_true_np = y_true_instance.detach().cpu().numpy()
        weight_map_np = unet_weight_map(y_true_np, wc=wc, w0=w0, sigma=sigma)
        weight_map = torch.from_numpy(weight_map_np).to(y_pred.device).float()

        if y_pred.dim() == 4 and y_pred.size(1) == 1:
            y_pred_instance = y_pred[i, 0]
        else:
            y_pred_instance = y_pred[i]

        y_pred_instance = y_pred_instance.unsqueeze(0)
        y_true_instance = y_true_instance.unsqueeze(0)
        #print(y_true_instance.shape, y_pred_instance.shape)
        cross = cross_fn(y_pred_instance, y_true_instance)
        loss_instance = torch.mean(weight_map * cross)
        losses.append(loss_instance)

    return torch.stack(losses).mean()
