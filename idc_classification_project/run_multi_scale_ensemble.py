"""
运行多尺度集成模型的脚本
"""

import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
import torch 

from idc_classification_project.multi_scale_ensemble_main import run_multi_scale_ensemble_pipeline

if __name__ == '__main__':
    print("="*60)
    print("开始运行多尺度集成模型")
    print("="*60)
    
    # 运行集成流程
    run_multi_scale_ensemble_pipeline()
    
    print("\n"+"="*60)
    print("多尺度集成模型运行完成")
    print("="*60) 