import os
import cv2
import numpy as np
import torch
from pathlib import Path

def seed_everything(seed=42):
    """固定随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def rle_encode(mask):
    """游程编码（竞赛官方格式）"""
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_decode(mask_rle, shape):
    """游程解码（竞赛官方格式）"""
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    mask = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask.reshape(shape)

def load_image(path, size=None):
    """加载图像并可选调整尺寸"""
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if size is not None:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img

def load_mask(path, size=None):
    """加载掩码（伪造区域）并可选调整尺寸"""
    if path.endswith('.npy'):
        mask = np.load(path)
    else:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if size is not None:
        mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0).astype(np.uint8)  # 二值化
    return mask