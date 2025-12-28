import os
import time
import torch
import torchvision
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from .dataset import ForgeryDataset
from .model import FastUNet, HybridLoss
from .utils import CFG, visualize_test_mask, plot_training_history, evaluate_test_set_with_f1
from .metrics import rle_encode, oF1_score

# 设置matplotlib
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = [12, 8]

# 设置随机种子
torch.manual_seed(CFG.SEED)
np.random.seed(CFG.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CFG.SEED)

# ===================== 修改后的三栏可视化函数 =====================
def create_triplet_visualization(original_img, mask_binary, prob_map, case_id, save_dir="triplet_visualizations"):
    """
    创建三栏拼接可视化图
    左栏: 覆盖图 (红色掩码叠加在原始图像上)
    中栏: 预测掩码 (白色显示)
    右栏: 原始图像
    """
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 确保mask_binary是正确的类型
    mask_binary = mask_binary.astype(np.uint8)
    
    # 调整图像大小，使其具有相同的高度
    target_height = 400
    aspect_ratio = original_img.shape[1] / original_img.shape[0]
    target_width = int(target_height * aspect_ratio)
    
    # 调整图像大小
    original_resized = cv2.resize(original_img, (target_width, target_height))
    
    # 调整概率图大小
    if prob_map is not None:
        prob_map_resized = cv2.resize(prob_map, (target_width, target_height))
    
    # 调整二值掩码大小
    mask_binary_resized = cv2.resize(mask_binary, (target_width, target_height))
    
    # 创建三栏拼接图
    triplet_img = np.zeros((target_height, target_width * 3, 3), dtype=np.uint8)
    
    # 1. 左栏：覆盖图（红色掩码叠加在原始图像上）
    overlay = original_resized.copy()
    if mask_binary_resized.sum() > 0:
        # 创建红色掩码层
        red_mask = np.zeros_like(overlay)
        red_mask[:, :, 0] = 255  # 红色通道
        
        # 扩展掩码为3通道
        mask_expanded = np.stack([mask_binary_resized] * 3, axis=2)
        
        # 在掩码区域应用红色叠加
        overlay = np.where(mask_expanded > 0, 
                          cv2.addWeighted(original_resized, 0.6, red_mask, 0.4, 0), 
                          original_resized)
    triplet_img[:, :target_width] = overlay
    
    # 2. 中栏：预测掩码（白色显示）
    mask_vis = np.zeros_like(original_resized)
    # 白色显示掩码区域
    mask_vis[mask_binary_resized > 0] = [255, 255, 255]
    triplet_img[:, target_width:target_width*2] = mask_vis
    
    # 3. 右栏：原始图像
    triplet_img[:, target_width*2:] = original_resized
    
    # 添加分隔线和标签
    cv2.line(triplet_img, (target_width, 0), (target_width, target_height), (255, 255, 255), 2)
    cv2.line(triplet_img, (target_width*2, 0), (target_width*2, target_height), (255, 255, 255), 2)
    
    # 添加标题
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(triplet_img, 'Overlay (Red Mask)', (10, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(triplet_img, 'Predicted Mask (White)', (target_width + 10, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(triplet_img, 'Original Image', (target_width*2 + 10, 30), font, 0.7, (255, 255, 255), 2)
    
    # 添加Case ID
    cv2.putText(triplet_img, f'Case ID: {case_id}', (10, target_height - 20), font, 0.6, (255, 255, 255), 2)
    
    # 添加掩码面积信息
    mask_area = mask_binary.sum()
    if mask_area > 0:
        img_height, img_width = original_img.shape[:2]
        mask_percentage = (mask_area / (img_height * img_width)) * 100
        area_text = f'Mask Area: {mask_area:,}px ({mask_percentage:.2f}%)'
        cv2.putText(triplet_img, area_text, (target_width + 10, target_height - 20), font, 0.6, (255, 255, 255), 2)
    else:
        cv2.putText(triplet_img, 'Authentic (No Mask)', (target_width + 10, target_height - 20), font, 0.6, (0, 255, 0), 2)
    
    # 保存图像
    save_path = os.path.join(save_dir, f"triplet_{case_id}.png")
    cv2.imwrite(save_path, cv2.cvtColor(triplet_img, cv2.COLOR_RGB2BGR))
    
    return save_path

def visualize_triplet_predictions(model, data_loader, save_dir="triplet_visualizations", num_samples=5):
    """
    可视化多个样本的三栏拼接图
    """
    model.eval()
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    samples_collected = 0
    all_triplet_paths = []
    
    with torch.no_grad():
        for imgs, masks in data_loader:
            imgs = imgs.to(CFG.DEVICE)
            outputs = model(imgs)
            
            for i in range(imgs.size(0)):
                if samples_collected >= num_samples:
                    break
                
                # 获取原始图像
                img_np = imgs[i].cpu().numpy().transpose(1, 2, 0)
                img_np = (img_np * 255).astype(np.uint8)
                
                # 获取概率图
                prob_map = outputs[i, 0].cpu().numpy()
                
                # 创建二值掩码
                mask_binary = (prob_map > CFG.INFERENCE_THRESHOLD).astype(np.uint8)
                
                # 小区域过滤
                if mask_binary.sum() < CFG.MIN_MASK_SIZE:
                    mask_binary[:] = 0
                
                # 创建三栏可视化
                save_path = create_triplet_visualization(
                    img_np, mask_binary, prob_map,
                    f"sample_{samples_collected+1}",
                    save_dir
                )
                all_triplet_paths.append(save_path)
                
                samples_collected += 1
            
            if samples_collected >= num_samples:
                break
    
    print(f"Generated {len(all_triplet_paths)} triplet visualizations in '{save_dir}'")
    return all_triplet_paths

def plot_training_triplet_comparison(train_history, val_f1_scores, save_path="training_comparison.png"):
    """
    绘制训练历史的三图对比
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    epochs = range(1, len(train_history['loss']) + 1)
    
    # 1. 损失曲线
    axes[0].plot(epochs, train_history['loss'], 'b-', linewidth=2, label='Train Loss')
    axes[0].plot(epochs, train_history['val_loss'], 'r-', linewidth=2, label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # 2. F1分数曲线
    axes[1].plot(epochs, train_history['val_f1'], 'g-', linewidth=2, label='Validation F1')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('F1 Score')
    axes[1].set_title('Validation F1 Score')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # 3. 训练vs验证对比
    if len(val_f1_scores) > 0:
        train_epochs = range(1, len(val_f1_scores) + 1)
        axes[2].plot(train_epochs, val_f1_scores, 'orange', linewidth=2, marker='o', label='Best Val F1 per Checkpoint')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('F1 Score')
        axes[2].set_title('Best Validation F1 Scores')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training comparison plot saved: {save_path}")
    plt.close()

# ===================== 修改后的visualize_test_mask函数 =====================
def visualize_test_mask(img_path, mask_binary, case_id, save_dir, prob_map=None):
    """
    可视化测试集预测mask - 三栏版本
    左栏: 覆盖图 (红色掩码叠加在原始图像上)
    中栏: 预测掩码 (白色显示)
    右栏: 原始图像
    """
    # 读取原始图像
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not load image {img_path}")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 确保mask_binary是正确的类型
    mask_binary = mask_binary.astype(np.uint8)
    
    # 创建三栏子图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. 左侧：覆盖图（红色掩码叠加在原始图像上）
    overlay = img.copy()
    if mask_binary.sum() > 0:
        # 创建红色掩码层
        red_mask = np.zeros_like(img)
        red_mask[:, :, 0] = 255  # 红色通道
        
        # 扩展掩码为3通道
        mask_expanded = np.stack([mask_binary] * 3, axis=2)
        
        # 在掩码区域应用红色叠加
        overlay = np.where(mask_expanded > 0, 
                          cv2.addWeighted(img, 0.6, red_mask, 0.4, 0), 
                          img)
    
    axes[0].imshow(overlay)
    axes[0].set_title("Overlay (Red Mask)", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # 2. 中间：预测掩码（白色显示）
    mask_vis = np.zeros_like(img)
    # 白色显示掩码区域
    mask_vis[mask_binary > 0] = [255, 255, 255]
    axes[1].imshow(mask_vis)
    axes[1].set_title("Predicted Mask (White)", fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # 3. 右侧：原始图像
    axes[2].imshow(img)
    axes[2].set_title(f"Original Image\n{case_id}", fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    # 添加整体标题
    mask_area = mask_binary.sum()
    img_height, img_width = img.shape[:2]
    total_pixels = img_height * img_width
    
    if mask_area > 0:
        forgery_type = "FORGED"
        mask_percentage = (mask_area / total_pixels) * 100
        title = f"Case {case_id}: {forgery_type} (Mask Area: {mask_area:,}px, {mask_percentage:.2f}%)"
    else:
        forgery_type = "AUTHENTIC"
        title = f"Case {case_id}: {forgery_type}"
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    # 添加阈值信息
    thresh_info = f"Threshold: {CFG.INFERENCE_THRESHOLD}"
    if prob_map is not None:
        thresh_info += f" | Max Probability: {prob_map.max():.3f}"
    
    plt.figtext(0.5, 0.02, thresh_info, ha='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    save_path = os.path.join(save_dir, f"test_mask_{case_id}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path

print("Starting data preparation...")

# 收集样本
all_samples = []
for file in os.listdir(CFG.TRAIN_AUTH_PATH):
    all_samples.append((os.path.join(CFG.TRAIN_AUTH_PATH, file), 0))

for file in os.listdir(CFG.TRAIN_FORGED_PATH):
    all_samples.append((os.path.join(CFG.TRAIN_FORGED_PATH, file), 1))

print(f"Total samples collected: {len(all_samples)}")

# 分割数据
train_samples, val_samples = train_test_split(
    all_samples,
    test_size=CFG.VAL_SPLIT,
    random_state=CFG.SEED,
    stratify=[s[1] for s in all_samples]
)

# 创建数据集和 dataloader
train_dataset = ForgeryDataset(train_samples, CFG.TRAIN_MASKS_PATH, CFG.IMG_SIZE, is_train=True)
val_dataset = ForgeryDataset(val_samples, CFG.TRAIN_MASKS_PATH, CFG.IMG_SIZE, is_train=False)

train_loader = DataLoader(
    train_dataset,
    batch_size=CFG.BATCH_SIZE,
    shuffle=True,
    num_workers=CFG.NUM_WORKERS,
    pin_memory=CFG.PIN_MEMORY
)

val_loader = DataLoader(
    val_dataset,
    batch_size=CFG.BATCH_SIZE,
    shuffle=False,
    num_workers=CFG.NUM_WORKERS,
    pin_memory=CFG.PIN_MEMORY
)

print("\nStarting model training...")
start_time = time.time()

# 初始化模型、损失、优化器
model = FastUNet().to(CFG.DEVICE)
criterion = HybridLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=CFG.LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)

best_val_f1 = 0.0
best_model_path = 'best_model.pth'
train_history = {'loss': [], 'val_loss': [], 'val_f1': []}
checkpoint_val_f1s = []

# 训练循环
for epoch in range(CFG.NUM_EPOCHS):
    print(f"\n--- Epoch {epoch + 1}/{CFG.NUM_EPOCHS} ---")

    # 训练
    model.train()
    total_loss = 0
    for imgs, masks in tqdm(train_loader, desc="Training"):
        imgs = imgs.to(CFG.DEVICE, non_blocking=True)
        masks = masks.to(CFG.DEVICE, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_loss = total_loss / len(train_loader)

    # 验证
    model.eval()
    val_loss, val_f1 = 0, 0
    with torch.no_grad():
        for imgs, masks in tqdm(val_loader, desc="Validating"):
            imgs = imgs.to(CFG.DEVICE, non_blocking=True)
            masks = masks.to(CFG.DEVICE, non_blocking=True)
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

            outputs_np = outputs.cpu().numpy()
            masks_np = masks.cpu().numpy()
            for i in range(outputs_np.shape[0]):
                prob_map = outputs_np[i, 0]
                gt_mask = masks_np[i, 0].astype(np.uint8)

                mean_prob = prob_map.mean()
                th = 0.35 if mean_prob < 0.15 else 0.4 if mean_prob < 0.25 else CFG.INFERENCE_THRESHOLD
                pred_mask = (prob_map > th).astype(np.uint8)

                if pred_mask.sum() < CFG.MIN_MASK_SIZE:
                    pred_mask[:] = 0

                if gt_mask.sum() == 0 and pred_mask.sum() == 0:
                    image_score = 1.0
                elif gt_mask.sum() == 0 or pred_mask.sum() == 0:
                    image_score = 0.0
                else:
                    image_score = oF1_score([pred_mask], [gt_mask])

                val_f1 += image_score

    val_loss /= len(val_loader)
    val_f1 /= len(val_loader.dataset)

    train_history['loss'].append(train_loss)
    train_history['val_loss'].append(val_loss)
    train_history['val_f1'].append(val_f1)

    print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val F1 = {val_f1:.4f}")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_f1': val_f1,
            'val_loss': val_loss,
        }, best_model_path)
        print(f"F1 improved to {val_f1:.4f}. Model saved.")

    if (epoch + 1) % 5 == 0:
        checkpoint_path = f'checkpoint_epoch_{epoch + 1}.pth'
        torch.save(model.state_dict(), checkpoint_path)
        checkpoint_val_f1s.append(val_f1)
        print(f"Checkpoint saved: {checkpoint_path}")
        
        # 可视化验证集的三栏图
        if (epoch + 1) % 10 == 0:
            print(f"Generating triplet visualizations for epoch {epoch + 1}...")
            triplet_paths = visualize_triplet_predictions(
                model, val_loader, 
                save_dir=f"triplet_epoch_{epoch+1}",
                num_samples=3
            )

end_time = time.time()
print(f"\nTraining finished in {(end_time - start_time) / 60:.2f} minutes.")

# 绘制训练历史
plot_training_history(train_history)
plot_training_triplet_comparison(train_history, checkpoint_val_f1s)

# 可视化训练集和验证集的样本
print("\nGenerating training set triplet visualizations...")
train_triplet_paths = visualize_triplet_predictions(
    model, train_loader, 
    save_dir="triplet_train_samples",
    num_samples=5
)

print("\nGenerating validation set triplet visualizations...")
val_triplet_paths = visualize_triplet_predictions(
    model, val_loader, 
    save_dir="triplet_val_samples",
    num_samples=5
)

# 推理阶段
print("\n" + "="*50)
print("Starting inference...")
print("="*50)

model.load_state_dict(torch.load(best_model_path)['model_state_dict'])
model.eval()

predictions = {}
test_files = sorted(os.listdir(CFG.TEST_PATH))

# 创建测试集可视化目录
test_triplet_dir = "triplet_test_results"
if not os.path.exists(test_triplet_dir):
    os.makedirs(test_triplet_dir)

with torch.no_grad():
    for file in tqdm(test_files, desc="Generating Predictions"):
        case_id = file.split('.')[0]
        img_path = os.path.join(CFG.TEST_PATH, file)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_size = img.shape[:2]
        img_resized = cv2.resize(img, (CFG.IMG_SIZE, CFG.IMG_SIZE))
        img_tensor = (img_resized.astype(np.float32) / 255.0)
        img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1).unsqueeze(0).to(CFG.DEVICE)

        mask_pred = model(img_tensor)[0, 0].cpu().numpy()
        mask_pred_resized = cv2.resize(mask_pred, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
        mask_binary = (mask_pred_resized > CFG.INFERENCE_THRESHOLD).astype(np.uint8)

        if mask_binary.sum() < CFG.MIN_MASK_SIZE:
            predictions[case_id] = "authentic"
        else:
            predictions[case_id] = rle_encode(mask_binary)

        # 保存 mask 和可视化
        np.save(os.path.join(CFG.MASK_SAVE_DIR, f"test_mask_{case_id}.npy"), mask_binary)
        
        # 使用新的三栏可视化函数
        visualize_test_mask(img_path, mask_binary, case_id, CFG.VISUALIZE_SAVE_DIR, mask_pred_resized)
        
        # 创建测试集的三栏可视化
        create_triplet_visualization(
            img, mask_binary, mask_pred_resized, 
            case_id, test_triplet_dir
        )

# 创建提交文件
sample_df = pd.read_csv(CFG.SAMPLE_SUB_PATH)
submission_data = [{'case_id': case_id, 'annotation': predictions.get(case_id, "authentic")} for case_id in sample_df['case_id']]
submission_df = pd.DataFrame(submission_data)
submission_df.to_csv('submission.csv', index=False)

print(f"\nGenerated {len(test_files)} triplet visualizations for test set in '{test_triplet_dir}'")
print("\n🎉 Submission created successfully!")
print(f"\n📊 Training completed with best validation F1: {best_val_f1:.4f}")
print("📈 Training curves and triplet visualizations have been generated.")