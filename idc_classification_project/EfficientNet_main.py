import torch
import numpy as np
import random
import os
import matplotlib.pyplot as plt  
from torch.cuda.amp import autocast, GradScaler

from . import configs
from . import data_utils
from . import train_utils
from .efficientnet_model import EfficientNetB0

# 置随机种子保证可复现性
def seed_everything(seed=configs.RANDOM_SEED if hasattr(configs, 'RANDOM_SEED') else 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_training_pipeline():
    """执行完整的训练和评估流程。"""
    # 0. 设置随机种子
    seed_everything()

    # 1. 加载/定义配置
    BATCH_SIZE = 32  
    patch_size = 224 # EfficientNet输入为224
    LEARNING_RATE = getattr(configs, 'LEARNING_RATE', 1e-4)
    NUM_EPOCHS = 1  # 只训练一轮用于测试，真正训练需要运行optimized文件
    WEIGHT_DECAY = getattr(configs, 'WEIGHT_DECAY', 1e-4)
    EARLY_STOPPING_PATIENCE = getattr(configs, 'EARLY_STOPPING_PATIENCE', 15)
    LOSS_FUNCTION_NAME = getattr(configs, 'LOSS_FUNCTION_NAME', 'focal_loss')
    FOCAL_ALPHA = getattr(configs, 'FOCAL_LOSS_ALPHA', 0.75)
    FOCAL_GAMMA = getattr(configs, 'FOCAL_LOSS_GAMMA', 2.0)
    MODEL_PRETRAINED = getattr(configs, 'MODEL_PRETRAINED', True)
    MODEL_FREEZE_BASE = getattr(configs, 'MODEL_FREEZE_BASE', False)
    NUM_WORKERS = getattr(configs, 'NUM_DATALOADER_WORKERS', 8)

    # 输出目录设置
    OUTPUT_BASE_DIR = 'output_efficientnet'  
    MODEL_SAVE_DIR = os.path.join(OUTPUT_BASE_DIR, 'models')
    VISUALIZATION_DIR = os.path.join(OUTPUT_BASE_DIR, 'visualizations')
    LOG_DIR = os.path.join(OUTPUT_BASE_DIR, 'logs')

    # 创建输出目录
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    print("--- 配置加载完成 ---")
    print(f"设备: {train_utils.get_device()}")
    print(f"数据集路径: {configs.BASE_DATA_DIR}")
    print(f"批次大小: {BATCH_SIZE}, 学习率: {LEARNING_RATE}, 周期数: {NUM_EPOCHS}")
    print(f"EfficientNet patch_size: {patch_size}")
    print(f"输出目录: {OUTPUT_BASE_DIR}")

    # 2. 获取设备
    device = train_utils.get_device()

    # 3. 准备数据加载器
    print("--- 准备数据加载器 ---")
    train_loader, val_loader, test_loader = data_utils.prepare_dataloaders(
        data_dir=configs.BASE_DATA_DIR,
        batch_size=BATCH_SIZE,
        test_size=getattr(configs, 'TEST_SPLIT_RATIO', 0.2),
        val_size=getattr(configs, 'VAL_SPLIT_RATIO', 0.1),
        random_state=getattr(configs, 'RANDOM_SEED', 42),
        patch_size=patch_size,   
        num_workers=NUM_WORKERS
    )
    
    if train_loader is None or val_loader is None or test_loader is None:
        print("错误: 数据加载失败，请检查数据路径和内容。程序将退出。")
        return
    
    print(f"训练集样本数: {len(train_loader.dataset)}, 验证集样本数: {len(val_loader.dataset)}, 测试集样本数: {len(test_loader.dataset)}")

    # 4. 处理类别不平衡
    class_weights = None
    if LOSS_FUNCTION_NAME == 'weighted_cross_entropy':
        print("--- 计算类别权重 ---")
        try:
            labels_for_weights = np.array(train_loader.dataset.labels)
            class_counts = np.bincount(labels_for_weights)
            if len(class_counts) < 2:
                print("警告: 训练数据中类别少于2个，将使用普通交叉熵。")
                LOSS_FUNCTION_NAME = 'cross_entropy'
            else:
                weights = 1. / class_counts
                weights = weights / np.sum(weights)
                class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
                print(f"类别计数: {class_counts}, 计算得到的权重: {class_weights.cpu().numpy()}")
        except Exception as e:
            print(f"错误: 计算类别权重失败: {e}。将使用普通交叉熵。")
            LOSS_FUNCTION_NAME = 'cross_entropy'

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
        class_weights=class_weights,
        device=device,
        focal_alpha=FOCAL_ALPHA,
        focal_gamma=FOCAL_GAMMA
    )

    # 7. 定义优化器和学习率调度器
    print("--- 定义优化器和学习率调度器 ---")
    optimizer = train_utils.get_optimizer(model, optimizer_name='adam', lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    USE_LR_SCHEDULER = getattr(configs, 'USE_LR_SCHEDULER', True)
    scheduler = None
    if USE_LR_SCHEDULER:
        LR_SCHEDULER_PATIENCE = getattr(configs, 'LR_SCHEDULER_PATIENCE', 5)
        LR_SCHEDULER_FACTOR = getattr(configs, 'LR_SCHEDULER_FACTOR', 0.5)
        LR_SCHEDULER_MIN_LR = getattr(configs, 'LR_SCHEDULER_MIN_LR', 1e-6)
        scheduler = train_utils.get_lr_scheduler(
            optimizer,
            scheduler_name='reduce_on_plateau',
            patience=LR_SCHEDULER_PATIENCE,
            factor=LR_SCHEDULER_FACTOR,
            min_lr=LR_SCHEDULER_MIN_LR
        )
        print(f"学习率调度器已启用: ReduceLROnPlateau (patience={LR_SCHEDULER_PATIENCE}, factor={LR_SCHEDULER_FACTOR})")
    else:
        print("学习率调度器未启用。")

    # 8. AMP混合精度训练准备
    scaler = GradScaler()

    # 9. 开始训练
    print("--- 开始训练 (AMP混合精度) ---")
    trained_model, history = train_utils.train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, device,
        num_epochs=NUM_EPOCHS,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        model_save_path=MODEL_SAVE_DIR,
        use_amp=True,
        scaler=scaler
    )
    print("--- 训练完成 ---")

    # 10. 可视化训练历史
    if history:
        print("--- 可视化训练历史 ---")
        train_utils.plot_training_history(history, save_dir=VISUALIZATION_DIR)
    else:
        print("警告: 未生成训练历史记录，跳过可视化。")

    # 11. 在测试集上评估最终模型
    print("--- 在测试集上评估最终模型 ---")
    OPTIMIZED_THRESHOLD = 0.57  
    print(f"应用分类阈值: {OPTIMIZED_THRESHOLD}")
    test_loss, test_acc, test_precision, test_recall, test_f1, test_auc, test_cm = train_utils.evaluate_model(
        trained_model, test_loader, criterion, device, threshold=OPTIMIZED_THRESHOLD, use_amp=True
    )
    # 获取所有标签和概率用于绘图
    all_labels = []
    all_probs = []
    trained_model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            with autocast():
                outputs = trained_model(inputs)
                probs = torch.softmax(outputs, dim=1)[:, 1]
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    # 绘制并保存ROC和PR曲线
    train_utils.plot_roc_curve(all_labels, all_probs, save_dir=VISUALIZATION_DIR, threshold=OPTIMIZED_THRESHOLD)
    train_utils.plot_pr_curve(all_labels, all_probs, save_dir=VISUALIZATION_DIR, threshold=OPTIMIZED_THRESHOLD)
    
    print(f"\n测试集结果 (来自 {OUTPUT_BASE_DIR}, 阈值: {OPTIMIZED_THRESHOLD}):")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Precision (IDC): {test_precision:.4f}")
    print(f"  Recall (Sensitivity for IDC): {test_recall:.4f}")
    print(f"  F1-score (IDC): {test_f1:.4f}")
    print(f"  AUC: {test_auc:.4f}")
    
    if test_cm is not None:
        print(f"  Confusion Matrix:\n{test_cm}")
        tn = test_cm[0,0]
        fp = test_cm[0,1]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        print(f"  Specificity (Non-IDC): {specificity:.4f}")

    results = {
        'test_loss': test_loss,
        'test_acc': test_acc,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'test_auc': test_auc,
        'test_specificity': specificity,
        'threshold': OPTIMIZED_THRESHOLD
    }
    
    import json
    with open(os.path.join(OUTPUT_BASE_DIR, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    print("--- 训练与评估流程结束 ---")

if __name__ == '__main__':
    # 设置默认配置
    if not hasattr(configs, 'RANDOM_SEED'): configs.RANDOM_SEED = 42
    if not hasattr(configs, 'BATCH_SIZE'): configs.BATCH_SIZE = 32
    if not hasattr(configs, 'LEARNING_RATE'): configs.LEARNING_RATE = 1e-4
    if not hasattr(configs, 'NUM_EPOCHS'): configs.NUM_EPOCHS = 50
    if not hasattr(configs, 'WEIGHT_DECAY'): configs.WEIGHT_DECAY = 1e-4
    if not hasattr(configs, 'EARLY_STOPPING_PATIENCE'): configs.EARLY_STOPPING_PATIENCE = 15
    
    if not hasattr(configs, 'LOSS_FUNCTION_NAME'): configs.LOSS_FUNCTION_NAME = 'focal_loss'
    if not hasattr(configs, 'FOCAL_LOSS_ALPHA'): configs.FOCAL_LOSS_ALPHA = 0.75
    if not hasattr(configs, 'FOCAL_LOSS_GAMMA'): configs.FOCAL_LOSS_GAMMA = 2.0
    
    if not hasattr(configs, 'MODEL_PRETRAINED'): configs.MODEL_PRETRAINED = True
    if not hasattr(configs, 'MODEL_FREEZE_BASE'): configs.MODEL_FREEZE_BASE = False
    if not hasattr(configs, 'TEST_SPLIT_RATIO'): configs.TEST_SPLIT_RATIO = 0.2
    if not hasattr(configs, 'VAL_SPLIT_RATIO'): configs.VAL_SPLIT_RATIO = 0.1
    if not hasattr(configs, 'OUTPUT_DIR'): configs.OUTPUT_DIR = 'output_efficientnet'
    if not hasattr(configs, 'NUM_DATALOADER_WORKERS'): configs.NUM_DATALOADER_WORKERS = 8
    
    run_training_pipeline() 
