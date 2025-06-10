import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
from sklearn.model_selection import train_test_split

# Albumentations 导入
import albumentations as A
from albumentations.pytorch import ToTensorV2 # 用于将 numpy array 转换为 PyTorch tensor

from idc_classification_project import configs 

def load_image_paths(data_dir):
    """
    从指定目录递归加载所有 PNG 图像的路径。
    """
    pattern = os.path.join(data_dir, '**', '*.png')
    image_paths = glob.glob(pattern, recursive=True)
    image_paths = [path for path in image_paths if os.path.isfile(path)]
    return image_paths

def extract_label_from_path(image_path):
    """
    从图像路径中提取标签。
    文件名格式为 ..._class0.png 或 ..._class1.png
    """
    filename = os.path.basename(image_path)
    try:
        label_str = filename.split('_class')[-1].split('.')[0]
        return int(label_str)
    except Exception as e:
        print(f"警告: 无法从文件名 {filename} 提取标签: {e}")
        return None

class IDCDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths (list): 图像文件路径列表。
            labels (list): 对应的标签列表。
            transform (callable, optional): 应用于样本的可选转换。
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform 

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.image_paths[idx]
        try:
            pil_image = Image.open(img_path).convert('RGB')
            image_np = np.array(pil_image)
        except FileNotFoundError:
            print(f"错误: 文件未找到 {img_path}")
            return None, None 
        except Exception as e:
            print(f"错误: 加载图像 {img_path} 时出错: {e}")
            return None, None
            
        label = self.labels[idx]

        if self.transform:
            try:
                augmented = self.transform(image=image_np)
                image = augmented['image'] 
            except Exception as e:
                print(f"错误: 应用 Albumentations 转换到图像 {img_path} 时出错: {e}")
                if 'image_np' in locals():
                    image = ToTensorV2()(image=image_np)['image'] 
                else: 
                    return None, None 
        else: 
             image = ToTensorV2()(image=image_np)['image']

        return image, torch.tensor(label, dtype=torch.long)

def get_data_transforms(patch_size=configs.PATCH_SIZE, use_augmentation=True):
    """
    使用 Albumentations 获取训练和验证/测试的数据转换。
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    if use_augmentation:
        # 训练数据增强 (使用 Albumentations)
        data_transform = A.Compose([
            A.Resize(height=patch_size, width=patch_size, interpolation=cv2.INTER_AREA), # 确保图像大小一致
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            # A.ElasticTransform(alpha=10, sigma=50, alpha_affine=30, p=0.3, border_mode=cv2.BORDER_CONSTANT, value=(0,0,0)), # 暂时注释掉以避免参数警告
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.Normalize(mean=imagenet_mean, std=imagenet_std),
            ToTensorV2() 
        ])
    else:
        # 验证/测试数据转换
        data_transform = A.Compose([
            A.Resize(height=patch_size, width=patch_size, interpolation=cv2.INTER_AREA),
            A.Normalize(mean=imagenet_mean, std=imagenet_std),
            ToTensorV2()
        ])
    
    return data_transform

def prepare_dataloaders(data_dir, batch_size, test_size=0.2, val_size=0.1, random_state=42, patch_size=configs.PATCH_SIZE, num_workers=0):

    all_image_paths = load_image_paths(data_dir)
    if not all_image_paths:
        print(f"错误: 在 {data_dir} 中未找到图像。请检查路径和文件。")
        return None, None, None

    labels = [extract_label_from_path(p) for p in all_image_paths]
    valid_indices = [i for i, label in enumerate(labels) if label is not None]
    image_paths = [all_image_paths[i] for i in valid_indices]
    labels = [labels[i] for i in valid_indices]

    if not image_paths:
        print("错误: 没有有效的图像和标签对可供处理。")
        return None, None, None

    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, labels, test_size=test_size, stratify=labels, random_state=random_state
    )
    actual_val_size = val_size / (1 - test_size) if (1-test_size) > 0 else 0
    if actual_val_size >= 1.0 or actual_val_size <=0:
        print(f"警告: 计算得到的验证集比例 actual_val_size ({actual_val_size:.2f}) 无效。请检查 test_size 和 val_size。暂时不划分验证集。")
        val_paths, val_labels = [], [] 
    else:
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_paths, train_labels, test_size=actual_val_size, stratify=train_labels, random_state=random_state
        )

    train_transforms = get_data_transforms(patch_size=patch_size, use_augmentation=True)
    eval_transforms = get_data_transforms(patch_size=patch_size, use_augmentation=False)

    train_dataset = IDCDataset(train_paths, train_labels, transform=train_transforms)
    val_dataset = IDCDataset(val_paths, val_labels, transform=eval_transforms) if val_paths else None # 如果 val_paths 为空则不创建
    test_dataset = IDCDataset(test_paths, test_labels, transform=eval_transforms)

    if len(train_dataset) == 0 or len(test_dataset) == 0:
        print("警告: 训练集或测试集为空。检查数据划分和原始数据量。")
    if val_dataset and len(val_dataset) == 0:
        print("警告: 验证集为空。")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader

# 辅助函数，用于可视化数据增强效果
def visualize_augmentations(dataset, num_samples=5):
    """
    可视化应用了数据增强的样本。
    (需要适配 albumentations 和 PIL->NumPy->Tensor 流程后的反归一化)
    """
    print("提示: visualize_augmentations 可能需要更新以完全兼容 albumentations 输出。")
    if not isinstance(dataset, IDCDataset) or not dataset.transform:
        print("错误: 需要一个应用了转换的 IDCDataset 实例。")
        return

    if len(dataset) == 0:
        print("错误: 数据集为空，无法可视化。")
        return

    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    if num_samples == 1:
        axes = [axes] 
        
    for i in range(num_samples):
        idx = random.randint(0, len(dataset) - 1)
        image, label = dataset[idx]
        if image is None: 
            print(f"警告: 索引 {idx} 处的样本加载失败，跳过可视化。")
            if hasattr(axes[i], 'set_title'):
                 axes[i].set_title(f"加载失败\n索引 {idx}")
                 axes[i].axis('off')
            continue

        # 反归一化以进行可视化 
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        img_display = image.cpu().clone() # 克隆以避免修改原始数据
        img_display.mul_(imagenet_std).add_(imagenet_mean) 
        img_display = img_display.permute(1, 2, 0).numpy() 
        img_display = np.clip(img_display, 0, 1)
        
        if hasattr(axes[i], 'imshow'): 
            axes[i].imshow(img_display)
            axes[i].set_title(f"标签: {label.item()}\n(增强后)")
            axes[i].axis('off')
        else:
            print(f"警告: axes[{i}] 不是有效的 Matplotlib Axes 对象。")

    plt.tight_layout()
    # 保存图像代替显示
    output_dir = getattr(configs, 'OUTPUT_DIR', 'output')
    vis_save_path = os.path.join(output_dir, 'visualizations', 'augmented_samples.png')
    os.makedirs(os.path.dirname(vis_save_path), exist_ok=True)
    plt.savefig(vis_save_path)
    print(f"增强样本可视化图已保存到: {vis_save_path}")
    plt.close(fig) 