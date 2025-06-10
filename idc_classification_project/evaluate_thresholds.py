import torch
import numpy as np
import os
import argparse
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, accuracy_score, roc_auc_score, roc_curve, auc as sklearn_auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 项目模块导入
from . import configs
from . import data_utils
from . import ResNet_model
from . import efficientnet_model
from . import train_utils

def get_predictions_and_labels(model, dataloader, device):
    """获取模型在给定数据加载器上的预测概率和真实标签"""
    model.eval()
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            # 获取类别1的概率 
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    return np.array(all_labels), np.array(all_probs)

def plot_roc_curve(true_labels, pred_probs, roc_auc, save_path):
    # 设置中文字体
    font_names = ['SimHei', 'Microsoft YaHei', 'FangSong', 'KaiTi', 'sans-serif']
    font_found = False
    for font_name in font_names:
        try:
            font_prop = fm.FontProperties(family=font_name)
            fm.findfont(font_prop, fallback_to_default=False, fontext='ttf')
            plt.rcParams['font.family'] = font_name
            print(f"ROC曲线绘图将使用字体: {font_name}")
            font_found = True
            break
        except Exception:
            continue
    if not font_found:
        print("警告: ROC曲线绘图未能自动找到并设置任何预定义的中文字体 (SimHei, Microsoft YaHei等).")
        print("      生成的图表中可能无法正确显示中文。")
    plt.rcParams['axes.unicode_minus'] = False
    
    fpr, tpr, _ = roc_curve(true_labels, pred_probs)
    plt.figure(figsize=(7, 5))
    lw = 2
    plt.plot(fpr, tpr, color='#ff7f0e', lw=lw, label=f'ROC曲线 (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--', label='随机分类器')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率', fontsize=12)
    plt.ylabel('真阳性率', fontsize=12)
    plt.title('ROC曲线', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"ROC曲线已保存到 {save_path}")

def plot_pr_curve(true_labels, pred_probs, avg_precision, save_path):
    # 设置中文字体
    font_names = ['SimHei', 'Microsoft YaHei', 'FangSong', 'KaiTi', 'sans-serif']
    font_found = False
    for font_name in font_names:
        try:
            font_prop = fm.FontProperties(family=font_name)
            fm.findfont(font_prop, fallback_to_default=False, fontext='ttf')
            plt.rcParams['font.family'] = font_name
            print(f"PR曲线绘图将使用字体: {font_name}")
            font_found = True
            break
        except Exception:
            continue
    if not font_found:
        print("警告: PR曲线绘图未能自动找到并设置任何预定义的中文字体 (SimHei, Microsoft YaHei等).")
        print("      生成的图表中可能无法正确显示中文。")
    plt.rcParams['axes.unicode_minus'] = False
    
    precision_vals, recall_vals, _ = precision_recall_curve(true_labels, pred_probs)
    plt.figure(figsize=(7, 5))
    lw = 2
    plt.plot(recall_vals, precision_vals, color='#1f77b4', lw=lw, label=f'精确率-召回率曲线 (AP = {avg_precision:.4f})')
    plt.xlabel('召回率', fontsize=12)
    plt.ylabel('精确率', fontsize=12)
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('精确率-召回率曲线', fontsize=14)
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"精确率-召回率曲线已保存到 {save_path}")

def evaluate_single_threshold(true_labels, pred_probs, threshold, print_results=True, plot_curves=False, plots_save_dir='.'):
    """使用单个阈值评估指标，并可选绘制曲线"""
    binary_predictions = (pred_probs >= threshold).astype(int)
    
    cm = confusion_matrix(true_labels, binary_predictions, labels=[0, 1])
    tn, fp, fn, tp = 0,0,0,0
    if cm.shape == (2,2): tn, fp, fn, tp = cm.ravel()
    elif cm.shape == (1,1) and true_labels[0] == 0: tn = cm[0,0]
    elif cm.shape == (1,1) and true_labels[0] == 1: tp = cm[0,0]
    else:
        tn = np.sum((true_labels == 0) & (binary_predictions == 0))
        fp = np.sum((true_labels == 0) & (binary_predictions == 1))
        fn = np.sum((true_labels == 1) & (binary_predictions == 0))
        tp = np.sum((true_labels == 1) & (binary_predictions == 1))

    sensitivity = recall_score(true_labels, binary_predictions, pos_label=1, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = precision_score(true_labels, binary_predictions, pos_label=1, zero_division=0)
    f1 = f1_score(true_labels, binary_predictions, pos_label=1, zero_division=0)
    accuracy = accuracy_score(true_labels, binary_predictions)
    auc_val = 0.0
    avg_precision = 0.0
    if len(np.unique(true_labels)) >= 2:
        auc_val = roc_auc_score(true_labels, pred_probs)
        avg_precision = average_precision_score(true_labels, pred_probs)
    else:
        print("出错: 标签中只有一个类")

    if print_results:
        print(f"\n--- Metrics for Threshold: {threshold:.2f} ---")
        print(f"  Accuracy:    {accuracy:.4f}")
        print(f"  AUC:         {auc_val:.4f}")
        print(f"  Average Precision: {avg_precision:.4f}")
        print(f"  Sensitivity: {sensitivity:.4f} (Recall for IDC)")
        print(f"  Specificity: {specificity:.4f} (for Non-IDC)")
        print(f"  Precision:   {precision:.4f} (for IDC)")
        print(f"  F1-score:    {f1:.4f} (for IDC)")
        print(f"  Confusion Matrix (TN, FP, FN, TP): [{tn}, {fp}, {fn}, {tp}]")
        print(f"  TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
    
    if plot_curves and len(np.unique(true_labels)) >= 2:
        train_utils.set_chinese_font_for_matplotlib_local()
        roc_save_path = os.path.join(plots_save_dir, f'roc_curve_thresh{threshold:.2f}.png')
        pr_save_path = os.path.join(plots_save_dir, f'pr_curve_thresh{threshold:.2f}.png')
        plot_roc_curve(true_labels, pred_probs, auc_val, roc_save_path)
        plot_pr_curve(true_labels, pred_probs, avg_precision, pr_save_path)
            
    return sensitivity, specificity, precision, f1, accuracy, auc_val, (tn, fp, fn, tp)

def evaluate_multiple_thresholds(true_labels, pred_probs):
    """遍历多个阈值并打印表格，找出满足目标的最佳阈值"""
    print("\nThreshold | Sensitivity (Recall) | Specificity | Precision | F1-Score  | TP   | FP   | TN   | FN   |")
    print("----------------------------------------------------------------------------------------------------")
    best_threshold_for_goal = None
    highest_f1_at_goal = -1.0

    for thr_val in np.arange(0.05, 0.96, 0.01):
        thr = round(thr_val, 2)
        sens, spec, prec, f1, _, _, (tn, fp, fn, tp) = evaluate_single_threshold(true_labels, pred_probs, thr, print_results=False, plot_curves=False)
        print(f"{thr:<9.2f} | {sens:<20.4f} | {spec:<11.4f} | {prec:<9.4f} | {f1:<9.4f} | {tp:<4} | {fp:<4} | {tn:<4} | {fn:<4} |")
        if sens >= 0.85 and spec >= 0.95:
            if f1 > highest_f1_at_goal:
                highest_f1_at_goal = f1
                best_threshold_for_goal = thr
                print(f"  ^^^ 找到最优解: {best_threshold_for_goal} (F1: {highest_f1_at_goal:.4f}) ^^^")

    if best_threshold_for_goal is not None:
        print(f"\nOptimal threshold meeting Sensitivity >= 0.85 and Specificity >= 0.95 is {best_threshold_for_goal:.2f}")
        print(f"With F1-score: {highest_f1_at_goal:.4f}")
    else:
        print("\n没有阈值可实现 Sensitivity>= 0.85 且 Specificity >= 0.95.")
    print("----------------------------------------------------------------------------------------------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate model with different thresholds or a single threshold on a specific data split.\n"
            "Example for EfficientNet (default):\n"
            "  python -m idc_classification_project.evaluate_thresholds --model_dir output_efficientnet_optimized --split test\n"
            "Example for ResNet:\n"
            "  python -m idc_classification_project.evaluate_thresholds --model_dir output_resnet_focal --model_arch resnet --split test"
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--model_dir', type=str, required=True, help='Directory of the trained model output (e.g., output_efficientnet_optimized). This directory should contain a \'models/best_model.pth\' file.')
    parser.add_argument('--model_arch', type=str, default='efficientnet', choices=['resnet', 'efficientnet'], 
                        help='Model architecture to load. Default is "efficientnet".\n'
                             'To evaluate a ResNet model, set this to "resnet" and ensure --model_dir points to the correct ResNet output directory.')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'], help='Data split to evaluate on: "val" or "test". Default is "val".')
    parser.add_argument('--threshold', type=float, default=None, help='Specific threshold to evaluate. If None, a range of thresholds will be evaluated.')
    parser.add_argument('--plot_curves', action='store_true', help='Plot ROC and PR curves if a single threshold is specified.')

    args = parser.parse_args()

    MODEL_OUTPUT_DIR = args.model_dir
    VISUALIZATION_DIR = os.path.join(MODEL_OUTPUT_DIR, 'visualizations_threshold_eval')
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)

    model_filename = 'best_model.pth'
    model_full_path = os.path.join(MODEL_OUTPUT_DIR, 'models', model_filename)

    if not os.path.exists(model_full_path):
        print(f"Error: Model file not found at {model_full_path}")
        exit()

    device = train_utils.get_device()
    print(f"Using device: {device}")
    print(f"Loading model: {model_full_path} with architecture: {args.model_arch}")

    # 加载模型架构
    if args.model_arch == 'efficientnet':
        model = efficientnet_model.EfficientNetB0(
            num_classes=2,
            pretrained=False
        )
    elif args.model_arch == 'resnet':
        print("--- Loading ResNet architecture as per --model_arch resnet ---")
        model = ResNet_model.SimpleResNetClassifier(
            num_classes=2,
            pretrained=False
        )
    else:
        raise ValueError(f"Unsupported model architecture: {args.model_arch}")

    try:
        model.load_state_dict(torch.load(model_full_path, map_location=device, weights_only=True))
    except Exception as e:
        print(f"Warning: Failed to load model with weights_only=True ({e}). Trying with weights_only=False.")
        model.load_state_dict(torch.load(model_full_path, map_location=device, weights_only=False))
    model.to(device)
    model.eval()

    # 使用configs中的参数作为默认值
    batch_size = getattr(configs, 'BATCH_SIZE', 64)
    test_split_ratio = getattr(configs, 'TEST_SPLIT_RATIO', 0.2)
    val_split_ratio = getattr(configs, 'VAL_SPLIT_RATIO', 0.1)
    random_seed = getattr(configs, 'RANDOM_SEED', 42)
    patch_size_config = getattr(configs, 'PATCH_SIZE', 224)
    num_workers = getattr(configs, 'NUM_DATALOADER_WORKERS', 6)

    # 为不同模型确定合适的patch_size
    if 'efficientnet' in args.model_dir.lower():
        current_patch_size = 224
        print(f"Detected 'efficientnet' in model_dir, using patch_size: {current_patch_size}")
    elif 'resnet' in args.model_dir.lower() and '50' in args.model_dir.lower():
        current_patch_size = 50
        print(f"Detected 'resnet' and '50' in model_dir, using patch_size: {current_patch_size}")
    else:
        current_patch_size = patch_size_config
        print(f"Using default patch_size from configs: {current_patch_size}")

    if args.split == 'val':
        _, dataloader, _ = data_utils.prepare_dataloaders(
            data_dir=configs.BASE_DATA_DIR,
            batch_size=batch_size,
            test_size=test_split_ratio,
            val_size=val_split_ratio,
            random_state=random_seed,
            patch_size=current_patch_size,
            num_workers=num_workers
        )
    elif args.split == 'test':
        _, _, dataloader = data_utils.prepare_dataloaders(
            data_dir=configs.BASE_DATA_DIR,
            batch_size=batch_size,
            test_size=test_split_ratio,
            val_size=val_split_ratio,
            random_state=random_seed,
            patch_size=current_patch_size,
            num_workers=num_workers
        )
    else:
        raise ValueError(f"Invalid split: {args.split}")

    if dataloader is None:
        print(f"Error: {args.split} data loader could not be prepared. Check data paths and configurations.")
        exit()
    
    print(f"{args.split.capitalize()} dataloader prepared with {len(dataloader.dataset)} samples using patch size {current_patch_size}.")

    # --- 执行评估 ---
    true_labels, pred_probs = get_predictions_and_labels(model, dataloader, device)

    if len(np.unique(true_labels)) < 2 and args.threshold is None:
        print(f"Warning: {args.split.capitalize()} set contains only one class. Cannot reliably evaluate multiple thresholds.")
    elif args.threshold is not None:
        print(f"Evaluating on {args.split} set with specified threshold: {args.threshold:.2f}")
        evaluate_single_threshold(true_labels, pred_probs, args.threshold, plot_curves=args.plot_curves, plots_save_dir=VISUALIZATION_DIR)
    else:
        print(f"Evaluating on {args.split} set with multiple thresholds.")
        evaluate_multiple_thresholds(true_labels, pred_probs) 