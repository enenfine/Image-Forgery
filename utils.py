# src/utils.py
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
VISUALIZE_SAVE_DIR = "test_mask_visualization"
MASK_SAVE_DIR = "test_mask_npy"
os.makedirs(VISUALIZE_SAVE_DIR, exist_ok=True)
os.makedirs(MASK_SAVE_DIR, exist_ok=True)


class CFG:
    BASE_PATH = "/media/dongli911/Software2/llj/task/imagine/database"

    TRAIN_AUTH_PATH = f"{BASE_PATH}/train_images/authentic"
    TRAIN_FORGED_PATH = f"{BASE_PATH}/train_images/forged"
    TRAIN_MASKS_PATH = f"{BASE_PATH}/train_masks"
    TEST_PATH = f"{BASE_PATH}/test_images"
    SAMPLE_SUB_PATH = f"{BASE_PATH}/submission.csv"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
       # 👇 必须添加下面这两行！
    MASK_SAVE_DIR = "test_mask_npy"
    VISUALIZE_SAVE_DIR = "test_mask_visualization"

    IMG_SIZE = 256
    BATCH_SIZE = 16
    NUM_EPOCHS = 10
    LR = 1e-3
    VAL_SPLIT = 0.2
    SEED = 42

    NUM_WORKERS = 2
    PIN_MEMORY = True

    INFERENCE_THRESHOLD = 0.5
    MIN_MASK_SIZE = 100


def visualize_test_mask(img_path, mask_binary, case_id, save_dir):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(img)
    ax1.set_title(f"Original Image: {case_id}")
    ax1.axis('off')

    ax2.imshow(img)
    ax2.imshow(mask_binary, cmap='Reds', alpha=0.5)
    ax2.set_title(f"Predicted Forgery Mask\nArea: {mask_binary.sum()} pixels")
    ax2.axis('off')

    save_path = os.path.join(save_dir, f"test_mask_{case_id}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 可视化结果已保存：{save_path}")


def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history['loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history['val_f1'], label='Val F1', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('F1 Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    plt.show()


def evaluate_test_set_with_f1(model, test_folder, mask_folder, device, threshold=0.5):
    model.eval()
    all_f1_scores = []

    test_files = sorted(os.listdir(test_folder))
    with torch.no_grad():
        for file in test_files:
            case_id = file.split('.')[0]
            img_path = os.path.join(test_folder, file)
            mask_path = os.path.join(mask_folder, f"{case_id}.npy")

            if not os.path.exists(mask_path):
                continue

            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            original_size = img.shape[:2]
            img_resized = cv2.resize(img, (CFG.IMG_SIZE, CFG.IMG_SIZE))
            img_tensor = (img_resized.astype(np.float32) / 255.0)
            img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1).unsqueeze(0).to(device)

            mask_pred = model(img_tensor)[0, 0].cpu().numpy()
            mask_pred = cv2.resize(mask_pred, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
            pred_mask_binary = (mask_pred > threshold).astype(np.uint8)

            try:
                gt_mask = np.load(mask_path)
                if gt_mask.ndim == 3:
                    gt_mask = gt_mask.max(axis=0) if gt_mask.shape[0] <= 10 else gt_mask.max(axis=-1)
                gt_mask_binary = (gt_mask > 0).astype(np.uint8)

                pred_masks = [pred_mask_binary]
                gt_masks = [gt_mask_binary]
                f1_score_val = oF1_score(pred_masks, gt_masks)
                all_f1_scores.append(f1_score_val)
            except Exception as e:
                print(f"Error loading mask for {case_id}: {e}")

    if all_f1_scores:
        avg_f1 = np.mean(all_f1_scores)
        print(f"\nTest Set Evaluation:")
        print(f"  Avg F1: {avg_f1:.4f}, Min: {np.min(all_f1_scores):.4f}, Max: {np.max(all_f1_scores):.4f}")
        return avg_f1, all_f1_scores
    else:
        print("No ground truth masks found.")
        return None, None