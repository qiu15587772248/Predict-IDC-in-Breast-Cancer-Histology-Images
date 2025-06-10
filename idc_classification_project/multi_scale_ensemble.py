import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, accuracy_score, precision_score, recall_score


class MultiScaleEnsembleModel(nn.Module):
    """
    多尺度集成模型，支持不同输入尺寸的模型集成
    特别优化了特异性指标
    """
    
    def __init__(self, models_config: List[Dict], ensemble_method: str = 'adaptive_weighted', 
                 initial_weights: Optional[List[float]] = None):

        super(MultiScaleEnsembleModel, self).__init__()
        
        self.models_config = models_config
        self.num_models = len(models_config)
        self.ensemble_method = ensemble_method
        
        # 将模型添加到ModuleList
        self.models = nn.ModuleList([config['model'] for config in models_config])
        
        # 初始化权重
        if initial_weights is None:
            # 根据模型特点设置初始权重
            # ResNet有更好的特异性，给予更高的初始权重
            self.weights = []
            for config in models_config:
                if 'resnet' in config['name'].lower():
                    self.weights.append(0.6)  # ResNet初始权重更高
                else:
                    self.weights.append(0.4)  # EfficientNet初始权重较低
            # 归一化
            total = sum(self.weights)
            self.weights = [w/total for w in self.weights]
        else:
            self.weights = initial_weights
            
        # 温度参数，用于调节概率分布的平滑度
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
        # 特异性增强系数
        self.specificity_boost = nn.Parameter(torch.tensor(1.2))
        
    def forward(self, inputs: Dict[str, torch.Tensor]):
        """
        前向传播，支持多尺度输入

        """
        outputs = []
        confidences = []
        
        for i, config in enumerate(self.models_config):
            model = self.models[i]
            input_key = f"{config['input_size'][0]}x{config['input_size'][1]}"
            
            if input_key in inputs:
                model.eval()
                with torch.no_grad():
                    output = model(inputs[input_key])
                    # 计算模型置信度（基于预测概率的熵）
                    probs = F.softmax(output / self.temperature, dim=1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                    confidence = 1.0 / (1.0 + entropy.mean())
                    
                    outputs.append(output)
                    confidences.append(confidence)
            else:
                raise ValueError(f"缺少输入尺寸 {input_key}")
        
        # 根据集成方法处理输出
        if self.ensemble_method == 'adaptive_weighted':
            return self._adaptive_weighted_ensemble(outputs, confidences)
        elif self.ensemble_method == 'confidence_based':
            return self._confidence_based_ensemble(outputs, confidences)
        elif self.ensemble_method == 'specificity_focused':
            return self._specificity_focused_ensemble(outputs)
        else:
            raise ValueError(f"不支持的集成方法: {self.ensemble_method}")
    
    def _adaptive_weighted_ensemble(self, outputs: List[torch.Tensor], 
                                   confidences: List[torch.Tensor]) -> torch.Tensor:
        """自适应加权集成"""
        ensemble_output = torch.zeros_like(outputs[0])
        
        # 根据置信度调整权重
        adjusted_weights = []
        for i in range(len(outputs)):
            adjusted_weight = self.weights[i] * confidences[i].item()
            adjusted_weights.append(adjusted_weight)
        
        # 归一化权重
        total_weight = sum(adjusted_weights)
        adjusted_weights = [w/total_weight for w in adjusted_weights]
        
        # 加权平均
        for i, output in enumerate(outputs):
            ensemble_output += adjusted_weights[i] * output
            
        return ensemble_output
    
    def _confidence_based_ensemble(self, outputs: List[torch.Tensor], 
                                  confidences: List[torch.Tensor]) -> torch.Tensor:
        """基于置信度的集成"""
        # 使用置信度作为权重
        conf_weights = [c.item() for c in confidences]
        total_conf = sum(conf_weights)
        conf_weights = [w/total_conf for w in conf_weights]
        
        ensemble_output = torch.zeros_like(outputs[0])
        for i, output in enumerate(outputs):
            ensemble_output += conf_weights[i] * output
            
        return ensemble_output
    
    def _specificity_focused_ensemble(self, outputs: List[torch.Tensor]) -> torch.Tensor:
        """特异性优化的集成方法"""
        # 获取每个模型的预测概率
        probs_list = [F.softmax(output, dim=1) for output in outputs]
        
        # 对于阴性类别（class 0），使用更保守的策略
        # 提高阴性预测的阈值，减少假阳性
        ensemble_probs = torch.zeros_like(probs_list[0])
        
        for i, probs in enumerate(probs_list):
            if 'resnet' in self.models_config[i]['name'].lower():
                # ResNet在特异性方面表现更好，给予更高权重
                weight_for_negative = self.weights[i] * self.specificity_boost
                weight_for_positive = self.weights[i]
            else:
                # EfficientNet权重保持不变或略微降低
                weight_for_negative = self.weights[i]
                weight_for_positive = self.weights[i]
            
            # 分别处理阴性和阳性类别的概率
            ensemble_probs[:, 0] += weight_for_negative * probs[:, 0]
            ensemble_probs[:, 1] += weight_for_positive * probs[:, 1]
        
        # 归一化概率
        ensemble_probs = ensemble_probs / ensemble_probs.sum(dim=1, keepdim=True)
        
        # 转换回logits
        ensemble_output = torch.log(ensemble_probs + 1e-8)
        
        return ensemble_output
    
    def predict_proba(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """获取预测概率"""
        outputs = self.forward(inputs)
        return F.softmax(outputs, dim=1)
    
    def optimize_for_metrics(self, val_loader_50, val_loader_224, device, 
                             min_specificity=0.95, min_sensitivity=0.85, 
                             weight_step=0.05, threshold_step=0.01):
        """
        在验证集上优化集成权重和阈值，以满足特定的敏感性和特异性要求。
        会优先选择满足要求的、F1分数最高的组合。
        """
        self.eval()
        
        # 1. 从验证集加载所有数据到CPU，以备后续多次使用
        print("正在从验证集加载所有数据到CPU，这可能需要一些时间...")
        all_val_inputs_50_cpu = []
        all_val_inputs_224_cpu = []
        all_val_labels_cpu_list = []

        # 确保迭代次数以较短的数据加载器为准，或确保长度一致
        # 假设两个 val_loader 长度相同，因为它们基于相同的基础数据集和划分比例
        for (batch_inputs_50, batch_labels_50), (batch_inputs_224, _) in zip(val_loader_50, val_loader_224):
            all_val_inputs_50_cpu.append(batch_inputs_50.cpu())
            all_val_inputs_224_cpu.append(batch_inputs_224.cpu())
            all_val_labels_cpu_list.append(batch_labels_50.cpu())
        
        if not all_val_labels_cpu_list:
            print("错误: 未能从验证集加载任何数据。请检查数据加载器。")
            # 返回一个表示失败的状态或默认值
            return self.weights, 0.5, {'f1': 0, 'specificity': 0, 'sensitivity': 0, 'auc': 0, 'accuracy': 0}

        all_val_labels_cpu = torch.cat(all_val_labels_cpu_list).numpy()
        num_val_batches = len(all_val_inputs_50_cpu)
        print(f"验证集数据已加载到CPU: {num_val_batches}批, 总样本数: {len(all_val_labels_cpu)}")

        best_f1 = -1
        best_weights_combo = self.weights
        best_threshold_val = 0.5
        found_satisfactory_combo = False
        
        # 存储满足要求的组合及其F1分数
        satisfactory_combos = []

        original_weights = list(self.weights) # 保存原始权重以便恢复

        print(f"开始搜索最佳权重 (步长 {weight_step}) 和阈值 (步长 {threshold_step})...")
        
        # 遍历不同的ResNet权重 (w1)，EfficientNet的权重将是 (1-w1)
        for w1 in np.arange(0, 1 + weight_step, weight_step):
            w2 = 1 - w1
            current_weights = [w1, w2]
            self.weights = torch.tensor(current_weights, dtype=torch.float32).to(device)
            
            # 对于当前的权重组合，获取模型在整个验证集上的概率输出
            # 按批次处理以避免OOM
            current_all_probs_gpu = []
            with torch.no_grad():
                for i in range(num_val_batches):
                    batch_inputs = {
                        '50x50': all_val_inputs_50_cpu[i].to(device),
                        '224x224': all_val_inputs_224_cpu[i].to(device)
                    }
                    # 确保 predict_proba 返回的是概率，并且是IDC类别的概率
                    batch_probs = self.predict_proba(batch_inputs)[:, 1] # 获取类别1的概率
                    current_all_probs_gpu.append(batch_probs.cpu()) # 移动回CPU以累积
            
            if not current_all_probs_gpu:
                print(f"警告: 权重 {current_weights} 未产生任何预测概率。跳过此权重组合。")
                continue
                
            current_all_probs_np = torch.cat(current_all_probs_gpu).numpy()

            # 遍历不同的阈值
            for t in np.arange(0.05, 0.95 + threshold_step, threshold_step):
                predictions = (current_all_probs_np >= t).astype(int)
                
                accuracy = accuracy_score(all_val_labels_cpu, predictions)
                precision = precision_score(all_val_labels_cpu, predictions, zero_division=0)
                recall = recall_score(all_val_labels_cpu, predictions, zero_division=0) # Sensitivity
                f1 = f1_score(all_val_labels_cpu, predictions, zero_division=0)
                
                if len(np.unique(all_val_labels_cpu)) >= 2:
                    try:
                        auc = roc_auc_score(all_val_labels_cpu, current_all_probs_np)
                    except ValueError: # 如果预测概率全为0或1
                        auc = 0.0 
                else:
                    auc = 0.0 # AUC无法计算
                
                cm_val = confusion_matrix(all_val_labels_cpu, predictions, labels=[0,1])
                tn, fp, fn, tp = cm_val.ravel() if cm_val.shape == (2,2) else (0,0,0,0)
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                sensitivity = recall # 召回率即敏感性

                if specificity >= min_specificity and sensitivity >= min_sensitivity:
                    satisfactory_combos.append({
                        'weights': list(current_weights), # Store as list
                        'threshold': t,
                        'f1': f1,
                        'specificity': specificity,
                        'sensitivity': sensitivity,
                        'auc': auc,
                        'accuracy': accuracy
                    })
                    found_satisfactory_combo = True
            
            # 打印当前权重组合的搜索进度 (可选)
            # print(f"  搜索权重: w_resnet={w1:.2f}, w_efficientnet={w2:.2f} 完成.")
        
        self.weights = torch.tensor(original_weights, dtype=torch.float32).to(device) # 恢复模型权重

        if not found_satisfactory_combo:
            print(f"警告: 未找到同时满足特异性(≥{min_specificity})和敏感性(≥{min_sensitivity})的权重/阈值组合。")
            print("将选择在所有尝试中F1分数最高的组合（不保证满足要求）。")
            # 如果没有任何组合满足要求，我们需要一种回退机制，这里可以简单地返回初始值或最佳F1（即使不满足要求）
            # 为了简单起见，如果没有满意的，可以考虑遍历所有尝试过的组合找到最佳F1，但这会增加复杂性
            # 目前的逻辑是，如果没有满意的，会返回初始权重和阈值。
            # 这里应该改进为，如果 satisfactory_combos 为空，则选择所有评估过的组合中 F1 最高的那个。
            # 暂时返回初始设置。
            best_metrics_dict = {'f1': 0, 'specificity': 0, 'sensitivity': 0, 'auc': 0, 'accuracy': 0}
            # 实际上，如果一个都没找到，应该有一个逻辑去找到"不那么差"的
            # 这里为了简化，如果没找到，则返回的是默认的初始权重和0.5阈值，以及全0的指标
            # 更好的做法是记录所有组合的F1，然后选择最高的那个，即使它不满足要求
            # TODO: 实现上述回退逻辑
            print("优化过程结束，但未找到满足所有条件的组合。请检查模型性能或调整优化目标。")
            return self.weights, 0.5, best_metrics_dict

        else:
            # 从满足要求的组合中选择F1最高的
            best_combo = max(satisfactory_combos, key=lambda x: x['f1'])
            best_weights_combo = best_combo['weights']
            best_threshold_val = best_combo['threshold']
            best_f1 = best_combo['f1']
            
            best_metrics_dict = {
                'f1': best_combo['f1'],
                'specificity': best_combo['specificity'],
                'sensitivity': best_combo['sensitivity'],
                'auc': best_combo['auc'],
                'accuracy': best_combo['accuracy']
            }
            print(f"\n优化完成。找到满足要求的最佳组合:")
            print(f"  最佳权重 (ResNet, EfficientNet): ({best_weights_combo[0]:.2f}, {best_weights_combo[1]:.2f})")
            print(f"  最佳阈值: {best_threshold_val:.3f}")
            print(f"  F1分数 (验证集): {best_metrics_dict['f1']:.4f}")
            print(f"  特异性 (验证集): {best_metrics_dict['specificity']:.4f}")
            print(f"  敏感性 (验证集): {best_metrics_dict['sensitivity']:.4f}")
            print(f"  AUC (验证集): {best_metrics_dict['auc']:.4f}")
            print(f"  准确率 (验证集): {best_metrics_dict['accuracy']:.4f}")

        # 设置模型的权重为找到的最佳权重
        self.weights = torch.tensor(best_weights_combo, dtype=torch.float32).to(device)
        
        return best_weights_combo, best_threshold_val, best_metrics_dict 