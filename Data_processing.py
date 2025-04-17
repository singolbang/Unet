import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import torch
import torch.nn.functional as F

folder_path = r"path" #data path

for filename in os.listdir(folder_path):
    if filename.endswith(".npy") and "input" in filename:
        file_path = os.path.join(folder_path, filename)
        data = np.load(file_path)

        tensor_data = torch.from_numpy(data)

        if tensor_data.ndim == 2:
            tensor_data = tensor_data.unsqueeze(0).unsqueeze(0)
        elif tensor_data.ndim == 3:
            tensor_data = tensor_data.unsqueeze(0)

        padded_tensor = F.pad(tensor_data, (30, 30, 30, 30), mode='reflect')

        if data.ndim == 2:
            padded_data = padded_tensor.squeeze(0).squeeze(0).numpy()
        elif data.ndim == 3:
            padded_data = padded_tensor.squeeze(0).numpy()
        else:
            padded_data = padded_tensor.numpy()

        np.save(file_path, padded_data)

