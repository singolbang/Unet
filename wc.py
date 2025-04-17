import os
import numpy as np
import torch


def compute_inverse_frequencies(image: torch.Tensor) -> dict:

    total_pixels = image.numel()
    bg_count = torch.sum(image == 255).item()  # 배경의 픽셀 수
    cell_count = torch.sum(image == 0).item()  # 세포의 픽셀 수

    bg_inv = total_pixels / bg_count if bg_count > 0 else 0.0
    cell_inv = total_pixels / cell_count if cell_count > 0 else 0.0

    return {0 : bg_inv, 1 : cell_inv}


#0이 세포, 255가 배경
