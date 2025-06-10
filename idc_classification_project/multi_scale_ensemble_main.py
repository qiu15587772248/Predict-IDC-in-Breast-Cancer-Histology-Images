import torch
import numpy as np
import os
import json
import warnings

# 项目模块导入
from . import configs
from . import data_utils
from . import train_utils
from .ResNet_model import SimpleResNetClassifier
from .efficientnet_model import EfficientNetB0
from .multi_scale_ensemble import MultiScaleEnsembleModel
from .train_utils import seed_everything  

# 在脚本早期抑制 albumentations 的版本检查警告
warnings.filterwarnings("ignore", message="Error fetching version info", module="albumentations.check_version")


def evaluate_multi_scale_ensemble(ensemble_model, test_loader_50, test_loader_224, 
                                 device, threshold=0.5):
    """
    评估多尺度集成模型
    """
    ensemble_model.eval()
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for (inputs_50, labels_50), (inputs_224, labels_224) in zip(test_loader_50, test_loader_224):
            # 准备多尺度输入
            inputs = {
                '50x50': inputs_50.to(device),
                '224x224': inputs_224.to(device)
            }
            
            with torch.amp.autocast(device_type='cuda'):
                probs = ensemble_model.predict_proba(inputs)[:, 1]  
            
            all_labels.extend(labels_50.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # 计算各项指标
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                               f1_score, roc_auc_score, confusion_matrix)
    
    predictions = (np.array(all_probs) >= threshold).astype(int)
    
    accuracy = accuracy_score(all_labels, predictions)
    precision = precision_score(all_labels, predictions, zero_division=0)
    recall = recall_score(all_labels, predictions, zero_division=0)
    f1 = f1_score(all_labels, predictions, zero_division=0)
    
    if len(np.unique(all_labels)) >= 2:
        auc = roc_auc_score(all_labels, all_probs)
    else:
        auc = 0.0
    
    cm = confusion_matrix(all_labels, predictions, labels=[0, 1])
    
    # 计算特异性
    specificity = 0.0
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'specificity': specificity,
        'confusion_matrix': cm,
        'threshold': threshold,
        'all_labels': all_labels,
        'all_probs': all_probs
    }


def run_multi_scale_ensemble_pipeline():
    """
    执行多尺度集成模型的完整流程
    """
    # 设置随机种子
    train_utils.seed_everything()
    
    # 配置
    device = train_utils.get_device()
    print(f"使用设备: {device}")
    

    _script_path = os.path.abspath(__file__)
    # _script_directory 是脚本所在的目录 
    _script_directory = os.path.dirname(_script_path)
    # _project_root 是项目根目录 
    _project_root = os.path.dirname(_script_directory)
    
    # 模型路径。请根据实际情况修改这些路径
    RESNET_MODEL_PATH = os.path.normpath(os.path.join(
        _project_root,
        'output_resnet_focal_alpha0.7_gamma3_lrsched_albu',
        'models',
        'best_model.pth'
    ))
    EFFICIENTNET_MODEL_PATH = os.path.normpath(os.path.join(
        _project_root,
        'output_efficientnet_optimized',
        'models',
        'best_model.pth'
    ))
    
    # 输出目录
    OUTPUT_DIR = 'output_multi_scale_ensemble'
    VISUALIZATION_DIR = os.path.join(OUTPUT_DIR, 'visualizations')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)
    
    print("--- 加载已训练的模型 ---")
    
    # 检查模型文件是否存在
    if not os.path.exists(RESNET_MODEL_PATH):
        print(f"错误: ResNet模型文件不存在: {RESNET_MODEL_PATH}")
        print("请确保模型文件路径正确")
        return
    
    if not os.path.exists(EFFICIENTNET_MODEL_PATH):
        print(f"错误: EfficientNet模型文件不存在: {EFFICIENTNET_MODEL_PATH}")
        print("请确保模型文件路径正确")
        return
    
    # 加载ResNet模型
    resnet_model = SimpleResNetClassifier(num_classes=2, pretrained=False)
    try:
        resnet_model.load_state_dict(torch.load(RESNET_MODEL_PATH, map_location=device, weights_only=True))
    except:
        resnet_model.load_state_dict(torch.load(RESNET_MODEL_PATH, map_location=device, weights_only=False))
    resnet_model.to(device)
    resnet_model.eval()
    print("ResNet模型加载成功")
    
    # 加载EfficientNet模型
    efficientnet_model = EfficientNetB0(num_classes=2, pretrained=False)
    try:
        efficientnet_model.load_state_dict(torch.load(EFFICIENTNET_MODEL_PATH, map_location=device, weights_only=True))
    except:
        efficientnet_model.load_state_dict(torch.load(EFFICIENTNET_MODEL_PATH, map_location=device, weights_only=False))
    efficientnet_model.to(device)
    efficientnet_model.eval()
    print("EfficientNet模型加载成功")
    
    # 准备数据加载器
    print("\n--- 准备多尺度数据加载器 ---")
    
    # 为两个模型设置统一的批次大小，以避免集成时维度不匹配问题
    # 选择较小模型的批次大小（通常是针对较大输入尺寸的模型）
    UNIFIED_BATCH_SIZE_FOR_ENSEMBLE = 32 
    print(f"为集成模型的数据加载器设置统一批次大小: {UNIFIED_BATCH_SIZE_FOR_ENSEMBLE}")

    # ResNet数据加载器 (50x50)
    print("准备50x50输入的数据加载器...")
    train_loader_50, val_loader_50, test_loader_50 = data_utils.prepare_dataloaders(
        data_dir=configs.BASE_DATA_DIR,
        batch_size=UNIFIED_BATCH_SIZE_FOR_ENSEMBLE,  # 使用统一的批次大小
        test_size=getattr(configs, 'TEST_SPLIT_RATIO', 0.2),
        val_size=getattr(configs, 'VAL_SPLIT_RATIO', 0.1),
        random_state=getattr(configs, 'RANDOM_SEED', 42),
        patch_size=50,  # ResNet输入尺寸
        num_workers=getattr(configs, 'NUM_DATALOADER_WORKERS', 8)
    )
    
    # EfficientNet数据加载器 (224x224)
    print("准备224x224输入的数据加载器...")
    train_loader_224, val_loader_224, test_loader_224 = data_utils.prepare_dataloaders(
        data_dir=configs.BASE_DATA_DIR,
        batch_size=UNIFIED_BATCH_SIZE_FOR_ENSEMBLE,  # 使用统一的批次大小
        test_size=getattr(configs, 'TEST_SPLIT_RATIO', 0.2),
        val_size=getattr(configs, 'VAL_SPLIT_RATIO', 0.1),
        random_state=getattr(configs, 'RANDOM_SEED', 42),
        patch_size=224,  # EfficientNet输入尺寸
        num_workers=getattr(configs, 'NUM_DATALOADER_WORKERS', 8)
    )
    
    if val_loader_50 is None or test_loader_50 is None or val_loader_224 is None or test_loader_224 is None:
        print("错误：一个或多个数据加载器未能成功创建。请检查数据路径、配置和 prepare_dataloaders 函数的输出。")
        return

    print(f"验证集样本数 (50x50): {len(val_loader_50.dataset) if val_loader_50 and hasattr(val_loader_50, 'dataset') else 'N/A'}")
    print(f"测试集样本数 (50x50): {len(test_loader_50.dataset) if test_loader_50 and hasattr(test_loader_50, 'dataset') else 'N/A'}")
    print(f"验证集样本数 (224x224): {len(val_loader_224.dataset) if val_loader_224 and hasattr(val_loader_224, 'dataset') else 'N/A'}")
    print(f"测试集样本数 (224x224): {len(test_loader_224.dataset) if test_loader_224 and hasattr(test_loader_224, 'dataset') else 'N/A'}")
    
    # 创建多尺度集成模型
    print("\n--- 创建多尺度集成模型 ---")
    
    models_config = [
        {
            'model': resnet_model,
            'input_size': (50, 50),
            'name': 'ResNet18'
        },
        {
            'model': efficientnet_model,
            'input_size': (224, 224),
            'name': 'EfficientNet-B0'
        }
    ]
    
    # 使用特异性优化的集成方法
    ensemble_model = MultiScaleEnsembleModel(
        models_config=models_config,
        ensemble_method='specificity_focused',  # 使用特异性优化的集成方法
        initial_weights=None  # 让模型自动设置初始权重
    )
    ensemble_model.to(device)
    
    # 在验证集上优化权重和阈值
    print("\n--- 在验证集上优化集成权重和阈值 ---")
    print("正在搜索最佳权重组合和阈值，这可能需要几分钟...")
    
    best_weights, best_threshold, best_metrics = ensemble_model.optimize_for_metrics(
        val_loader_50=val_loader_50,
        val_loader_224=val_loader_224,
        device=device,
        min_specificity=0.95,  # 特异性要求
        min_sensitivity=0.85   # 敏感性要求
    )
    
    # 在测试集上评估
    print("\n--- 在测试集上评估多尺度集成模型 ---")
    test_results = evaluate_multi_scale_ensemble(
        ensemble_model, test_loader_50, test_loader_224, device, threshold=best_threshold
    )
    
    print(f"\n测试集结果 (阈值: {test_results['threshold']:.3f}):")
    print(f"  准确率: {test_results['accuracy']:.4f}")
    print(f"  精确率: {test_results['precision']:.4f}")
    print(f"  召回率 (敏感性): {test_results['recall']:.4f}")
    print(f"  F1分数: {test_results['f1']:.4f}")
    print(f"  AUC: {test_results['auc']:.4f}")
    print(f"  特异性: {test_results['specificity']:.4f}")
    print(f"  混淆矩阵:\n{test_results['confusion_matrix']}")
    
    # 检查是否达到项目要求
    print("\n--- 性能指标检查 ---")
    requirements_met = True
    
    if test_results['accuracy'] >= 0.90:
        print(f"✓ 准确率: {test_results['accuracy']:.4f} (要求 ≥ 0.90)")
    else:
        print(f"✗ 准确率: {test_results['accuracy']:.4f} (要求 ≥ 0.90)")
        requirements_met = False
    
    if test_results['auc'] >= 0.95:
        print(f"✓ AUC: {test_results['auc']:.4f} (要求 ≥ 0.95)")
    else:
        print(f"✗ AUC: {test_results['auc']:.4f} (要求 ≥ 0.95)")
        requirements_met = False
    
    if test_results['specificity'] >= 0.95:
        print(f"✓ 特异性: {test_results['specificity']:.4f} (要求 ≥ 0.95)")
    else:
        print(f"✗ 特异性: {test_results['specificity']:.4f} (要求 ≥ 0.95)")
        requirements_met = False
    
    if test_results['recall'] >= 0.85:
        print(f"✓ 敏感性: {test_results['recall']:.4f} (要求 ≥ 0.85)")
    else:
        print(f"✗ 敏感性: {test_results['recall']:.4f} (要求 ≥ 0.85)")
        requirements_met = False
    
    if requirements_met:
        print("\n🎉 恭喜！多尺度集成模型满足所有项目要求！")
    else:
        print("\n⚠️ 多尺度集成模型尚未满足所有要求。")
    
    # 保存结果
    results_dict = {
        'ensemble_method': 'specificity_focused',
        'model_weights': {
            'resnet': float(best_weights[0]),
            'efficientnet': float(best_weights[1])
        },
        'best_threshold': float(best_threshold),
        'validation_metrics': {
            'f1': float(best_metrics['f1']),
            'specificity': float(best_metrics['specificity']),
            'sensitivity': float(best_metrics['sensitivity']),
            'auc': float(best_metrics['auc']),
            'accuracy': float(best_metrics['accuracy'])
        },
        'test_results': {
            'accuracy': float(test_results['accuracy']),
            'precision': float(test_results['precision']),
            'recall': float(test_results['recall']),
            'f1': float(test_results['f1']),
            'auc': float(test_results['auc']),
            'specificity': float(test_results['specificity']),
            'confusion_matrix': test_results['confusion_matrix'].tolist()
        },
        'requirements_met': requirements_met
    }
    
    with open(os.path.join(OUTPUT_DIR, 'multi_scale_ensemble_results.json'), 'w') as f:
        json.dump(results_dict, f, indent=4, ensure_ascii=False)
    
    print(f"\n结果已保存到: {os.path.join(OUTPUT_DIR, 'multi_scale_ensemble_results.json')}")
    
    # 绘制ROC曲线和PR曲线
    print("\n--- 绘制性能曲线 ---")
    
    # 确保使用中文字体
    train_utils.set_chinese_font_for_matplotlib_local()
    
    train_utils.plot_roc_curve(
        test_results['all_labels'], 
        test_results['all_probs'], 
        save_dir=VISUALIZATION_DIR, 
        threshold=best_threshold
    )
    train_utils.plot_pr_curve(
        test_results['all_labels'], 
        test_results['all_probs'], 
        save_dir=VISUALIZATION_DIR, 
        threshold=best_threshold
    )
    
    # 绘制集成模型性能总结图
    visualize_ensemble_results(results_dict, VISUALIZATION_DIR)
    
    # 保存集成模型
    ensemble_state = {
        'models_config': [
            {'name': config['name'], 'input_size': config['input_size']} 
            for config in models_config
        ],
        'weights': best_weights,
        'threshold': best_threshold,
        'ensemble_method': 'specificity_focused'
    }
    
    torch.save(ensemble_state, os.path.join(OUTPUT_DIR, 'ensemble_model_state.pth'))
    print(f"\n集成模型状态已保存到: {os.path.join(OUTPUT_DIR, 'ensemble_model_state.pth')}")
    
    print("\n--- 多尺度集成模型评估完成 ---")


def visualize_ensemble_results(results_dict, save_dir):
    """
    可视化集成模型的结果，包括模型权重和性能对比
    """
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    
    # 设置中文字体
    font_names = ['SimHei', 'Microsoft YaHei', 'FangSong', 'KaiTi', 'sans-serif']
    for font_name in font_names:
        try:
            font_prop = fm.FontProperties(family=font_name)
            fm.findfont(font_prop, fallback_to_default=False, fontext='ttf')
            plt.rcParams['font.family'] = font_name
            print(f"集成模型可视化使用字体: {font_name}")
            break
        except Exception:
            continue
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 模型权重饼图
    ax1 = axes[0, 0]
    weights = [results_dict['model_weights']['resnet'], results_dict['model_weights']['efficientnet']]
    labels = ['ResNet18\n(50×50)', 'EfficientNet-B0\n(224×224)']
    colors = ['#ff9999', '#66b3ff']
    ax1.pie(weights, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('模型权重分配', fontsize=14, fontweight='bold')
    
    # 2. 性能指标对比柱状图
    ax2 = axes[0, 1]
    metrics = ['准确率', '精确率', '召回率', 'F1分数', 'AUC', '特异性']
    values = [
        results_dict['test_results']['accuracy'],
        results_dict['test_results']['precision'],
        results_dict['test_results']['recall'],
        results_dict['test_results']['f1'],
        results_dict['test_results']['auc'],
        results_dict['test_results']['specificity']
    ]
    bars = ax2.bar(metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel('分数', fontsize=12)
    ax2.set_title('集成模型性能指标', fontsize=14, fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3)
    
    # 在柱状图上添加数值
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 3. 混淆矩阵热力图
    ax3 = axes[1, 0]
    cm = results_dict['test_results']['confusion_matrix']
    im = ax3.imshow(cm, interpolation='nearest', cmap='Blues')
    ax3.set_xticks([0, 1])
    ax3.set_yticks([0, 1])
    ax3.set_xticklabels(['预测: 非IDC', '预测: IDC'])
    ax3.set_yticklabels(['实际: 非IDC', '实际: IDC'])
    ax3.set_xlabel('预测标签', fontsize=12)
    ax3.set_ylabel('实际标签', fontsize=12)
    ax3.set_title('混淆矩阵', fontsize=14, fontweight='bold')
    
    # 在混淆矩阵中添加数值
    for i in range(2):
        for j in range(2):
            text = ax3.text(j, i, str(cm[i][j]), ha="center", va="center",
                           color="white" if cm[i][j] > cm.max()/2 else "black", fontsize=12)
    
    # 4. 目标达成情况
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # 创建表格数据
    requirements = [
        ['指标', '要求', '实际值', '状态'],
        ['准确率', '≥ 90%', f"{results_dict['test_results']['accuracy']:.1%}", 
         '✓' if results_dict['test_results']['accuracy'] >= 0.90 else '✗'],
        ['AUC', '≥ 95%', f"{results_dict['test_results']['auc']:.1%}", 
         '✓' if results_dict['test_results']['auc'] >= 0.95 else '✗'],
        ['特异性', '≥ 95%', f"{results_dict['test_results']['specificity']:.1%}", 
         '✓' if results_dict['test_results']['specificity'] >= 0.95 else '✗'],
        ['敏感性', '≥ 85%', f"{results_dict['test_results']['recall']:.1%}", 
         '✓' if results_dict['test_results']['recall'] >= 0.85 else '✗']
    ]
    
    # 创建表格
    table = ax4.table(cellText=requirements[1:], colLabels=requirements[0],
                     cellLoc='center', loc='center', colWidths=[0.3, 0.2, 0.3, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # 设置表格样式
    for i in range(len(requirements)):
        for j in range(len(requirements[0])):
            cell = table[(i, j)]
            if i == 0:  # 表头
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            else:
                if j == 3:  # 状态列
                    if requirements[i][j] == '✓':
                        cell.set_facecolor('#c8e6c9')
                    else:
                        cell.set_facecolor('#ffcdd2')
    
    ax4.set_title('项目要求达成情况', fontsize=14, fontweight='bold', pad=20)
    
    # 调整子图间距
    plt.tight_layout()
    
    # 保存图形
    save_path = os.path.join(save_dir, 'ensemble_performance_summary.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"集成模型性能总结图已保存到: {save_path}")
    plt.close()


if __name__ == '__main__':
    run_multi_scale_ensemble_pipeline() 