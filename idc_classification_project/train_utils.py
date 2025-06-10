import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import numpy as np
import time
import os
import copy
from torch.cuda.amp import GradScaler

from idc_classification_project import configs
from idc_classification_project.ResNet_model import SimpleResNetClassifier 

import random
import os
import numpy as np
import torch


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



# --- Matplotlib 中文字体设置函数定义 ---

def set_chinese_font_for_matplotlib_local(): 
    import matplotlib.font_manager as fm 
    import matplotlib.pyplot as plt    
    font_names = ['SimHei', 'Microsoft YaHei', 'FangSong', 'KaiTi', 'sans-serif'] 
    font_path_found = None
    for font_name in font_names:
        try:
            font_prop = fm.FontProperties(family=font_name)
            fm.findfont(font_prop, fallback_to_default=False, fontext='ttf') 
            plt.rcParams['font.family'] = font_name
            font_path_found = font_name
            print(f"Matplotlib 将使用字体: {font_name} 来显示中文。")
            break  
        except Exception:
            if font_name == font_names[-1]:
                 print(f"警告: 未能自动找到并设置合适的中文字体 (如 SimHei, Microsoft YaHei)。中文可能显示为方块。")

    plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

# --- 设备配置 ---
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")



# --- Focal Loss 实现 ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha # 通常 alpha 对于数量较少的类别赋予较高权重，对于数量较多的类别赋予较低权重。alpha for class 1
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):

        BCE_loss = F.binary_cross_entropy_with_logits(inputs[:, 1], targets.float(), reduction='none')

        pt = torch.exp(-BCE_loss) 


        alpha_factor = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        

        F_loss = alpha_factor * (1 - pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

# --- 损失函数 --- 
def get_loss_function(loss_name='cross_entropy', class_weights=None, device=None, focal_alpha=0.25, focal_gamma=2.0):
    """
    获取损失函数。

    """
    if loss_name == 'weighted_cross_entropy':
        if class_weights is None:
            raise ValueError("使用 weighted_cross_entropy 时必须提供 class_weights")
        if device:
            class_weights = class_weights.to(device)
        return nn.CrossEntropyLoss(weight=class_weights)
    elif loss_name == 'focal_loss':
        print(f"使用 Focal Loss (alpha={focal_alpha}, gamma={focal_gamma})")
        return FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    elif loss_name == 'cross_entropy':
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"不支持的损失函数: {loss_name}")

# --- 优化器 ---
def get_optimizer(model, optimizer_name='adam', lr=1e-3, weight_decay=0):
    """
    获取优化器。
    """
    if optimizer_name.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"不支持的优化器: {optimizer_name}")

# --- 学习率调度器 ---
def get_lr_scheduler(optimizer, scheduler_name='reduce_on_plateau', patience=5, factor=0.1, min_lr=1e-6, step_size=10, gamma=0.1):
    """
    获取学习率调度器。
    """
    if scheduler_name.lower() == 'reduce_on_plateau':
        return ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, min_lr=min_lr)
    elif scheduler_name.lower() == 'step_lr':
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        return None 

# --- 训练一个 Epoch ---
def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch_num=0, total_epochs=0, use_amp=False, scaler=None):
    model.train() 
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    start_time = time.time()
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        if inputs is None or labels is None: 
            print(f"警告: 在 epoch {epoch_num+1}, batch {batch_idx+1} 中检测到无效数据，跳过此批次。")
            continue
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad() 
        if use_amp and scaler is not None:
            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == labels.data)
        total_samples += labels.size(0)
        if (batch_idx + 1) % 50 == 0: # 每50个batch打印一次进度
            batch_time = time.time() - start_time
            print(f'Epoch [{epoch_num+1}/{total_epochs}] Batch [{batch_idx+1}/{len(dataloader)}] Loss: {loss.item():.4f} Time: {batch_time:.2f}s')
            start_time = time.time() 
    epoch_loss = (running_loss / total_samples) if total_samples > 0 else 0
    epoch_acc = (correct_predictions.double() / total_samples).item() if total_samples > 0 else 0 # 使用 .item() 转换为 python 标量
    return epoch_loss, epoch_acc

# --- 评估模型 ---
def evaluate_model(model, dataloader, criterion, device, threshold=0.5, use_amp=False):
    model.eval() 
    running_loss = 0.0
    all_labels = []
    all_probs = [] 
    with torch.no_grad(): 
        for inputs, labels in dataloader:
            if inputs is None or labels is None: 
                print(f"警告: 在评估过程中检测到无效数据，跳过此批次。")
                continue
            inputs, labels = inputs.to(device), labels.to(device)
            if use_amp:
                with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            probs_for_class_1 = torch.softmax(outputs, dim=1)[:, 1] 
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs_for_class_1.cpu().numpy())
    num_samples = len(all_labels)
    if num_samples == 0:
        print("警告: 评估数据集中没有有效样本。")
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, None 
    all_preds_thresholded = (np.array(all_probs) >= threshold).astype(int)
    val_loss = running_loss / num_samples
    val_acc = accuracy_score(all_labels, all_preds_thresholded)
    val_precision = precision_score(all_labels, all_preds_thresholded, average='binary', pos_label=1, zero_division=0)
    val_recall = recall_score(all_labels, all_preds_thresholded, average='binary', pos_label=1, zero_division=0)
    val_f1 = f1_score(all_labels, all_preds_thresholded, average='binary', pos_label=1, zero_division=0)
    if len(np.unique(all_labels)) < 2:
        val_auc = 0.0 
        print("警告: 评估集中的标签只有单一类别，AUC无法计算，设为0.0")
    else:
        val_auc = roc_auc_score(all_labels, all_probs) 
    conf_matrix = confusion_matrix(all_labels, all_preds_thresholded, labels=[0, 1]) # 确保标签顺序
    return val_loss, val_acc, val_precision, val_recall, val_f1, val_auc, conf_matrix

# --- 主训练循环 ---
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, 
                num_epochs=configs.NUM_EPOCHS if hasattr(configs, 'NUM_EPOCHS') else 25, 
                early_stopping_patience=configs.EARLY_STOPPING_PATIENCE if hasattr(configs, 'EARLY_STOPPING_PATIENCE') else 5,
                model_save_path=os.path.join(configs.OUTPUT_DIR if hasattr(configs, 'OUTPUT_DIR') else 'output', 'models'),
                use_amp=False, scaler=None):
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path, exist_ok=True)
        print(f"创建模型保存目录: {model_save_path}")
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_metric = 0.0 
    epochs_no_improve = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 
               'val_precision': [], 'val_recall': [], 'val_f1': [], 'val_auc': []}
    start_total_time = time.time()
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, num_epochs, use_amp=use_amp, scaler=scaler)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        print(f'Epoch {epoch+1} Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
        val_loss, val_acc, val_precision, val_recall, val_f1, val_auc, conf_matrix = evaluate_model(model, val_loader, criterion, device, use_amp=use_amp)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)
        history['val_auc'].append(val_auc)
        print(f'Epoch {epoch+1} Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} Precision: {val_precision:.4f} Recall: {val_recall:.4f} F1: {val_f1:.4f} AUC: {val_auc:.4f}')
        if conf_matrix is not None:
            print(f'Validation Confusion Matrix:\n{conf_matrix}')
        if scheduler and isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        elif scheduler and isinstance(scheduler, StepLR):
            scheduler.step() 
        current_val_metric = val_f1 
        if current_val_metric > best_val_metric:
            print(f"Validation F1 improved from {best_val_metric:.4f} to {current_val_metric:.4f}. Saving model...")
            best_val_metric = current_val_metric
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), os.path.join(model_save_path, 'best_model.pth'))
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"Validation F1 did not improve. ({current_val_metric:.4f} vs best {best_val_metric:.4f}). Patience: {epochs_no_improve}/{early_stopping_patience}")
            if epochs_no_improve >= early_stopping_patience:
                print(f'Early stopping triggered after {epoch+1} epochs.')
                break
        epoch_time_taken = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} completed in {epoch_time_taken:.2f}s")
    total_time_taken = time.time() - start_total_time
    print(f'\nTraining complete in {total_time_taken // 60:.0f}m {total_time_taken % 60:.0f}s')
    print(f'Best validation F1-score: {best_val_metric:4f}')
    model.load_state_dict(best_model_wts)
    return model, history

# --- 训练过程可视化 ---
def plot_training_history(history, save_dir, model_name_suffix="model"):
    import os
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    # 设置中文字体
    font_names = ['SimHei', 'Microsoft YaHei', 'FangSong', 'KaiTi', 'sans-serif']
    for font_name in font_names:
        try:
            font_prop = fm.FontProperties(family=font_name)
            fm.findfont(font_prop, fallback_to_default=False, fontext='ttf')
            plt.rcParams['font.family'] = font_name
            break
        except Exception:
            continue
    plt.rcParams['axes.unicode_minus'] = False
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 10))
    # 损失
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history['train_loss'], 'o-', color='#1f77b4', label='训练损失', linewidth=2, markersize=5)
    plt.plot(epochs, history['val_loss'], 'o-', color='#d62728', label='验证损失', linewidth=2, markersize=5)
    plt.title('训练与验证损失', fontsize=14)
    plt.xlabel('轮次', fontsize=12)
    plt.ylabel('损失', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    # 准确率
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history['train_acc'], 'o-', color='#1f77b4', label='训练准确率', linewidth=2, markersize=5)
    plt.plot(epochs, history['val_acc'], 'o-', color='#d62728', label='验证准确率', linewidth=2, markersize=5)
    plt.title('训练与验证准确率', fontsize=14)
    plt.xlabel('轮次', fontsize=12)
    plt.ylabel('准确率', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    # F1/AUC
    plt.subplot(2, 2, 3)
    plt.plot(epochs, history['val_f1'], 'o-', color='#2ca02c', label='验证F1分数', linewidth=2, markersize=5)
    plt.plot(epochs, history['val_auc'], 'o-', color='#9467bd', label='验证AUC', linewidth=2, markersize=5)
    plt.title('验证F1分数与AUC', fontsize=14)
    plt.xlabel('轮次', fontsize=12)
    plt.ylabel('分数', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    # 精确率/召回率
    plt.subplot(2, 2, 4)
    plt.plot(epochs, history['val_precision'], 'o-', color='#17becf', label='验证精确率', linewidth=2, markersize=5)
    plt.plot(epochs, history['val_recall'], 'o-', color='#bcbd22', label='验证召回率', linewidth=2, markersize=5)
    plt.title('验证精确率与召回率', fontsize=14)
    plt.xlabel('轮次', fontsize=12)
    plt.ylabel('分数', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    # 使用 model_name_suffix 来定制文件名
    base_filename = 'training_history'
    if model_name_suffix:
        base_filename += f"_{model_name_suffix}"

    save_path = os.path.join(save_dir, f'{base_filename}.png') 

    plt.savefig(save_path, dpi=150)
    print(f"训练历史图已保存到 {save_path}")
    plt.close()

def plot_roc_curve(labels, probs, save_dir, threshold=0.57):
    import os
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    from sklearn.metrics import roc_curve, auc
    font_names = ['SimHei', 'Microsoft YaHei', 'FangSong', 'KaiTi', 'sans-serif']
    for font_name in font_names:
        try:
            font_prop = fm.FontProperties(family=font_name)
            fm.findfont(font_prop, fallback_to_default=False, fontext='ttf')
            plt.rcParams['font.family'] = font_name
            break
        except Exception:
            continue
    plt.rcParams['axes.unicode_minus'] = False
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7,5))
    plt.plot(fpr, tpr, color='#ff7f0e', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('假阳性率', fontsize=12)
    plt.ylabel('真阳性率', fontsize=12)
    plt.title('ROC曲线', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'roc_curve_thresh{threshold}.png')
    plt.savefig(save_path, dpi=150)
    print(f"ROC曲线已保存到 {save_path}")
    plt.close()

def plot_pr_curve(labels, probs, save_dir, threshold=0.57):
    import os
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    from sklearn.metrics import precision_recall_curve, average_precision_score
    font_names = ['SimHei', 'Microsoft YaHei', 'FangSong', 'KaiTi', 'sans-serif']
    for font_name in font_names:
        try:
            font_prop = fm.FontProperties(family=font_name)
            fm.findfont(font_prop, fallback_to_default=False, fontext='ttf')
            plt.rcParams['font.family'] = font_name
            break
        except Exception:
            continue
    plt.rcParams['axes.unicode_minus'] = False
    precision, recall, _ = precision_recall_curve(labels, probs)
    ap = average_precision_score(labels, probs)
    plt.figure(figsize=(7,5))
    plt.plot(recall, precision, color='#1f77b4', lw=2, label=f'精确率-召回率曲线 (AP = {ap:.4f})')
    plt.xlabel('召回率', fontsize=12)
    plt.ylabel('精确率', fontsize=12)
    plt.title('精确率-召回率曲线', fontsize=14)
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'pr_curve_thresh{threshold}.png')
    plt.savefig(save_path, dpi=150)
    print(f"PR曲线已保存到 {save_path}")
    plt.close()
