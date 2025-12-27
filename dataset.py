# src/dataset.py
import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

class ForgeryDataset(Dataset):
    def __init__(self, samples, masks_path, img_size, is_train=True):
        self.samples = samples
        self.masks_path = masks_path
        self.img_size = img_size
        self.is_train = is_train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, is_forged = self.samples[idx]

        # 1. 加载图像并调整大小
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))

        # 2. 加载或创建掩码
        mask_path = os.path.join(self.masks_path, f"{os.path.basename(img_path).split('.')[0]}.npy")

        if is_forged and os.path.exists(mask_path):
            try:
                mask = np.load(mask_path)
                if mask.ndim == 3:
                    mask = mask.max(axis=0) if mask.shape[0] <= 10 else mask.max(axis=-1)
                mask = cv2.resize(mask.astype(np.uint8), (self.img_size, self.img_size))
                mask = (mask > 0).astype(np.float32)
            except Exception as e:
                mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        else:
            mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)

        # 3. 归一化和转换
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)  # HWC -> CWH
        mask = torch.from_numpy(mask).unsqueeze(0)  # HW -> 1HW

        return img, mask