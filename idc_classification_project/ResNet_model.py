import torch
import torch.nn as nn
import torchvision.models as models

class SimpleResNetClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, freeze_base=False):
        """
        基于 ResNet 的分类器。

        """
        super(SimpleResNetClassifier, self).__init__()
        
        # 加载预训练的 ResNet18
        self.resnet_base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        
        if freeze_base:
            for param in self.resnet_base.parameters():
                param.requires_grad = False
            # 解冻最后几层通常有助于微调
            # for param in self.resnet_base.layer4.parameters():
            #     param.requires_grad = True

        # 获取全连接层的输入特征数
        num_ftrs = self.resnet_base.fc.in_features
        
        # 替换原来的全连接层为一个新的，以匹配我们的类别数
        self.resnet_base.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet_base(x)

if __name__ == '__main__':
    # 测试模型定义
    dummy_input = torch.randn(4, 3, 50, 50) 
    
    # 测试 ResNet 分类器
    print("测试 ResNet 分类器:")
    resnet_model = SimpleResNetClassifier(num_classes=2, pretrained=True)
    output = resnet_model(dummy_input)
    print("ResNet 输出形状:", output.shape) 
    assert output.shape == (4, 2)


    print("\n模型架构定义基本测试完成。") 