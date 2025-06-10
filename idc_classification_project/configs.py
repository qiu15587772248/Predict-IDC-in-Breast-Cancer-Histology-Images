import os

# 数据集路径
BASE_DATA_DIR = r"E:\datasets\IDC_data"

# 图像块大小
PATCH_SIZE = 50

# --- 核心训练参数 ---
RANDOM_SEED = 42
BATCH_SIZE = 64 
LEARNING_RATE = 5e-5 
NUM_EPOCHS = 40     
WEIGHT_DECAY = 1e-5 
EARLY_STOPPING_PATIENCE = 10 
NUM_DATALOADER_WORKERS = 8 # 设置 DataLoader的 num_workers

# --- 学习率调度器配置 (新增) ---
USE_LR_SCHEDULER = True
LR_SCHEDULER_PATIENCE = 5 # ReduceLROnPlateau 的耐心值
LR_SCHEDULER_FACTOR = 0.1 # 学习率降低的因子
LR_SCHEDULER_MIN_LR = 1e-7 # 学习率的下限

# --- 损失函数配置 ---
# 可选: 'cross_entropy', 'weighted_cross_entropy', 'focal_loss'
LOSS_FUNCTION_NAME = 'focal_loss' 
FOCAL_LOSS_ALPHA = 0.70 # 略微降低 alpha，以平衡特异性和召回率
FOCAL_LOSS_GAMMA = 3.0  # 进一步提升 gamma，以期提高特异性

# --- 模型配置 ---
MODEL_NAME = 'SimpleResNetClassifier' # 保留当前模型
MODEL_PRETRAINED = True
MODEL_FREEZE_BASE = False 

# --- 数据集划分 ---
TEST_SPLIT_RATIO = 0.2
VAL_SPLIT_RATIO = 0.1 # 占总数据集的10%，实际在 prepare_dataloaders 中会转换为相对于 (1-test_split) 的比例

# --- 输出目录 ---
OUTPUT_DIR = 'output_resnet_focal_alpha0.7_gamma3_lrsched_albu' 

# --- 数据增强配置  ---
USE_ALBUMENTATIONS = True 