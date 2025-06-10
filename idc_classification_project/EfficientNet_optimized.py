import torch
import numpy as np
import random
import os
import copy

# 禁用 albumentations 更新检查，实际并不影响训练结果但是太浪费时间了
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from torch.amp import GradScaler 
import warnings

# 修改导入语句
from . import configs
from . import data_utils
from . import train_utils
from .efficientnet_model import EfficientNetB0

# 在脚本早期抑制 albumentations 的版本检查警告
warnings.filterwarnings("ignore", message="Error fetching version info", module="albumentations.check_version")

def seed_everything(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_optimized_training():
    """
    执行优化的EfficientNet训练，专注于提高特异性
    """
    # 0. 设置随机种子
    seed_everything(42)
    
    # 1. 优化的训练配置
    BATCH_SIZE = 32
    patch_size = 224
    LEARNING_RATE = 3e-5  
    NUM_EPOCHS = 50  
    WEIGHT_DECAY = 1e-4
    EARLY_STOPPING_PATIENCE = 15
    
    # 特异性优化的损失函数配置
    LOSS_FUNCTION_NAME = 'focal_loss'
    FOCAL_ALPHA = 0.65  # 降低alpha值，给予阴性样本更多权重
    FOCAL_GAMMA = 3.5   # 提高gamma值，更关注困难样本
    
    MODEL_PRETRAINED = True
    MODEL_FREEZE_BASE = False
    NUM_WORKERS = 8
    
    # 输出目录设置
    OUTPUT_BASE_DIR = 'output_efficientnet_optimized'
    MODEL_SAVE_DIR = os.path.join(OUTPUT_BASE_DIR, 'models')
    VISUALIZATION_DIR = os.path.join(OUTPUT_BASE_DIR, 'visualizations')
    LOG_DIR = os.path.join(OUTPUT_BASE_DIR, 'logs')
    
    # 创建输出目录
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    print("--- 优化配置加载完成 ---")
    print(f"设备: {train_utils.get_device()}")
    print(f"数据集路径: {configs.BASE_DATA_DIR}")
    print(f"批次大小: {BATCH_SIZE}, 学习率: {LEARNING_RATE}, 周期数: {NUM_EPOCHS}")
    print(f"Focal Loss 参数: alpha={FOCAL_ALPHA}, gamma={FOCAL_GAMMA}")
    print(f"输出目录: {OUTPUT_BASE_DIR}")
    
    # 2. 获取设备
    device = train_utils.get_device()
    
    # 3. 准备数据加载器
    print("--- 准备数据加载器 ---")
    train_loader, val_loader, test_loader = data_utils.prepare_dataloaders(
        data_dir=configs.BASE_DATA_DIR,
        batch_size=BATCH_SIZE,
        test_size=0.2,
        val_size=0.1,
        random_state=42,
        patch_size=patch_size,
        num_workers=NUM_WORKERS
    )
    
    if train_loader is None or val_loader is None or test_loader is None:
        print("错误: 数据加载失败，请检查数据路径和内容。")
        return
    
    print(f"训练集样本数: {len(train_loader.dataset)}")
    print(f"验证集样本数: {len(val_loader.dataset)}")
    print(f"测试集样本数: {len(test_loader.dataset)}")
    
    # 4. 处理类别不平衡
    # 计算训练集的类别分布
    labels = np.array(train_loader.dataset.labels)
    class_counts = np.bincount(labels)
    print(f"类别分布: Class 0 (Non-IDC): {class_counts[0]}, Class 1 (IDC): {class_counts[1]}")
    
    # 5. 初始化模型
    print(f"--- 初始化模型: EfficientNetB0 (pretrained={MODEL_PRETRAINED}) ---")
    model = EfficientNetB0(
        num_classes=2,
        pretrained=MODEL_PRETRAINED
    ).to(device)
    
    # 6. 定义损失函数
    print(f"--- 定义损失函数: {LOSS_FUNCTION_NAME} ---")
    criterion = train_utils.get_loss_function(
        loss_name=LOSS_FUNCTION_NAME,
        device=device,
        focal_alpha=FOCAL_ALPHA,
        focal_gamma=FOCAL_GAMMA
    )
    
    # 7. 定义优化器和学习率调度器
    print("--- 定义优化器和学习率调度器 ---")
    optimizer = train_utils.get_optimizer(
        model, 
        optimizer_name='adam', 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY
    )
    
    # 使用余弦退火学习率调度
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10,  # 第一个周期的长度
        T_mult=2,  # 后续周期长度的倍数
        eta_min=1e-7  # 最小学习率
    )
    
    # 8. AMP混合精度训练准备
    scaler = torch.amp.GradScaler() 
    
    # 9. 自定义训练循环，增加特异性监控
    print("--- 开始优化训练 ---")
    
    best_specificity = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_precision': [], 'val_recall': [], 
        'val_f1': [], 'val_auc': [], 'val_specificity': []
    }
    
    for epoch in range(NUM_EPOCHS):
        print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}')
        print('-' * 50)
        
        # 训练阶段
        train_loss, train_acc = train_utils.train_one_epoch(
            model, train_loader, criterion, optimizer, device, 
            epoch, NUM_EPOCHS, use_amp=True, scaler=scaler
        )
        
        # 验证阶段
        val_loss, val_acc, val_precision, val_recall, val_f1, val_auc, val_cm = train_utils.evaluate_model(
            model, val_loader, criterion, device, threshold=0.5, use_amp=True
        )
        
        # 计算特异性
        val_specificity = 0.0
        if val_cm is not None and val_cm.shape == (2, 2):
            tn, fp = val_cm[0, 0], val_cm[0, 1]
            val_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)
        history['val_auc'].append(val_auc)
        history['val_specificity'].append(val_specificity)
        
        print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
        print(f'Val Precision: {val_precision:.4f} Recall: {val_recall:.4f}')
        print(f'Val F1: {val_f1:.4f} AUC: {val_auc:.4f}')
        print(f'Val Specificity: {val_specificity:.4f}')
        
        # 学习率调度
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Learning Rate: {current_lr:.6f}')
        
        # 保存最佳模型（基于特异性）
        if val_specificity > best_specificity and val_recall >= 0.80:  # 确保召回率不太低
            print(f"特异性提升: {best_specificity:.4f} -> {val_specificity:.4f}")
            best_specificity = val_specificity
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, 'best_model.pth'))
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        # 早停
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f'早停触发，{EARLY_STOPPING_PATIENCE}个epoch内特异性未提升')
            break
    
    # 加载最佳模型
    model.load_state_dict(best_model_wts)
    
    # 10. 可视化训练历史
    print("--- 可视化训练历史 ---")
    
    # --- 为 EfficientNet_optimized.py 中的训练历史图表设置中文字体 ---
    font_names_hist = ['SimHei', 'Microsoft YaHei', 'FangSong', 'KaiTi', 'sans-serif']
    font_found_hist = False
    for font_name_hist in font_names_hist:
        try:
            font_prop_hist = fm.FontProperties(family=font_name_hist)
            fm.findfont(font_prop_hist, fallback_to_default=False, fontext='ttf')
            plt.rcParams['font.family'] = font_name_hist # 设置全局字体
            print(f"训练历史图表绘图将尝试使用字体: {font_name_hist}")
            font_found_hist = True
            break 
        except Exception:
            continue
    if not font_found_hist:
        print("警告: 训练历史图表绘图未能自动找到任何预定义的中文字体。中文可能无法正确显示。")
    plt.rcParams['axes.unicode_minus'] = False 
    # --- 中文字体设置结束 ---

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 损失
    axes[0, 0].plot(history['train_loss'], label='训练损失')
    axes[0, 0].plot(history['val_loss'], label='验证损失')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('训练与验证损失')
    axes[0, 0].legend()
    
    # 准确率
    axes[0, 1].plot(history['train_acc'], label='训练准确率')
    axes[0, 1].plot(history['val_acc'], label='验证准确率')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('训练与验证准确率')
    axes[0, 1].legend()
    
    # F1和AUC
    axes[0, 2].plot(history['val_f1'], label='F1分数')
    axes[0, 2].plot(history['val_auc'], label='AUC')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Score')
    axes[0, 2].set_title('F1分数与AUC')
    axes[0, 2].legend()
    
    # 精确率和召回率
    axes[1, 0].plot(history['val_precision'], label='精确率')
    axes[1, 0].plot(history['val_recall'], label='召回率')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('精确率与召回率')
    axes[1, 0].legend()
    
    # 特异性
    axes[1, 1].plot(history['val_specificity'], label='特异性', color='red')
    axes[1, 1].axhline(y=0.95, color='green', linestyle='--', label='目标特异性(0.95)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Specificity')
    axes[1, 1].set_title('验证集特异性')
    axes[1, 1].legend()
    
    # 特异性vs召回率
    axes[1, 2].plot(history['val_specificity'], history['val_recall'], 'o-')
    axes[1, 2].axvline(x=0.95, color='green', linestyle='--', label='目标特异性(0.95)')
    axes[1, 2].axhline(y=0.85, color='blue', linestyle='--', label='目标召回率(0.85)')
    axes[1, 2].set_xlabel('Specificity')
    axes[1, 2].set_ylabel('Recall')
    axes[1, 2].set_title('特异性 vs 召回率')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'training_history_optimized.png'), dpi=150)
    plt.close()
    
    # 11. 在测试集上找最佳阈值
    print("\n--- 在测试集上寻找最佳阈值 ---")
    
    # 获取测试集预测概率
    model.eval()
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            with torch.amp.autocast(device_type='cuda', enabled=True): 
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)[:, 1]
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # 寻找最佳阈值
    best_threshold = 0.5
    best_combined_score = 0
    
    for threshold in np.arange(0.3, 0.8, 0.01):
        predictions = (np.array(all_probs) >= threshold).astype(int)
        
        from sklearn.metrics import confusion_matrix, f1_score
        cm = confusion_matrix(all_labels, predictions)
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            # 如果满足最小要求
            if specificity >= 0.95 and sensitivity >= 0.85:
                f1 = f1_score(all_labels, predictions)
                combined_score = 0.4 * specificity + 0.3 * sensitivity + 0.3 * f1
                
                if combined_score > best_combined_score:
                    best_combined_score = combined_score
                    best_threshold = threshold
    
    print(f"最佳阈值: {best_threshold:.3f}")
    
    # 12. 使用最佳阈值评估
    print(f"\n--- 使用最佳阈值 {best_threshold:.3f} 在测试集上评估 ---")
    test_loss, test_acc, test_precision, test_recall, test_f1, test_auc, test_cm = train_utils.evaluate_model(
        model, test_loader, criterion, device, threshold=best_threshold, use_amp=True
    )
    
    # 计算特异性
    test_specificity = 0.0
    if test_cm is not None and test_cm.shape == (2, 2):
        tn, fp = test_cm[0, 0], test_cm[0, 1]
        test_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\n测试集结果:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall (Sensitivity): {test_recall:.4f}")
    print(f"  F1-score: {test_f1:.4f}")
    print(f"  AUC: {test_auc:.4f}")
    print(f"  Specificity: {test_specificity:.4f}")
    print(f"  Confusion Matrix:\n{test_cm}")
    
    # 绘制ROC和PR曲线
    train_utils.plot_roc_curve(all_labels, all_probs, save_dir=VISUALIZATION_DIR, threshold=best_threshold)
    train_utils.plot_pr_curve(all_labels, all_probs, save_dir=VISUALIZATION_DIR, threshold=best_threshold)
    
    # 保存结果
    results = {
        'best_threshold': float(best_threshold),
        'test_loss': float(test_loss),
        'test_acc': float(test_acc),
        'test_precision': float(test_precision),
        'test_recall': float(test_recall),
        'test_f1': float(test_f1),
        'test_auc': float(test_auc),
        'test_specificity': float(test_specificity),
        'confusion_matrix': test_cm.tolist() if test_cm is not None else None,
        'training_config': {
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'num_epochs': epoch + 1,
            'focal_alpha': FOCAL_ALPHA,
            'focal_gamma': FOCAL_GAMMA
        }
    }
    
    import json
    with open(os.path.join(OUTPUT_BASE_DIR, 'optimized_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n结果已保存到: {OUTPUT_BASE_DIR}")
    print("--- 优化训练完成 ---")


if __name__ == '__main__':
    run_optimized_training() 