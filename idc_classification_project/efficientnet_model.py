import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.efficientnet import EfficientNet_B0_Weights

class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(EfficientNetB0, self).__init__()
        # 加载预训练的EfficientNet-B0模型
        self.efficientnet = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # 获取输入特征数
        in_features = self.efficientnet.classifier[1].in_features
        
        self.efficientnet.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.efficientnet(x)

if __name__ == '__main__':
    # 测试模型
    model = EfficientNetB0(num_classes=2)  
    print("EfficientNet-B0 模型结构:")
    print(model)

    # 注意：虽然原始图像是50x50，但EfficientNet-B0期望224x224的输入，数据加载器会自动处理这个调整
    dummy_input = torch.randn(1, 3, 224, 224)  
    
    try:
        output = model(dummy_input)
        print(f"\n模型输出形状: {output.shape}")  
        if output.shape == (1, 2):
            print("模型前向传播测试成功！")
        else:
            print(f"模型输出形状不符合预期 (1, 2)，得到 {output.shape}")
    except Exception as e:
        print(f"模型前向传播测试失败: {e}")

    # 检查预训练权重是否加载
    model_pretrained = EfficientNetB0(num_classes=2, pretrained=True)
    model_scratch = EfficientNetB0(num_classes=2, pretrained=False)

    # 比较第一个卷积层的权重
    pretrained_conv1_weight = next(model_pretrained.efficientnet.features[0][0].parameters()).data
    scratch_conv1_weight = next(model_scratch.efficientnet.features[0][0].parameters()).data
    
    if not torch.allclose(pretrained_conv1_weight, scratch_conv1_weight):
        print("\n预训练权重已成功加载 (与从头训练的模型权重不同)。")
    else:
        print("\n预训练权重加载可能存在问题或模型参数一致。")

    # 验证不同类别数量
    model_custom_classes = EfficientNetB0(num_classes=100)
    dummy_input_custom = torch.randn(1, 3, 224, 224)
    output_custom = model_custom_classes(dummy_input_custom)
    print(f"\n自定义类别数量 (100) 的模型输出形状: {output_custom.shape}") # 预期: (1, 100)
    if output_custom.shape == (1, 100):
        print("自定义类别数量测试成功！")
    else:
        print(f"自定义类别数量测试形状不符合预期 (1, 100)，得到 {output_custom.shape}") 