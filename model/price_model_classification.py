
        
        
import torch.nn as nn
import torch
import torch.nn.functional as F
# from resnet import resnet18, resnet34, resnet50, resnet152

# input_feature = 1408 # effnet
# input_feature = 2048 # resnet50 or 152

        
        
import torch.nn as nn
import torch
import torch.nn.functional as F
# from resnet import resnet18, resnet34, resnet50, resnet152

# input_feature = 1408 # effnet
# input_feature = 2048 # resnet50 or 152
input_feature = 512 # resnet18 or 34
#input_feature = 3072 # nfnets

class Uniqlo_price_cls_model(nn.Module):
    def __init__(self, backbone):
        super(Uniqlo_price_cls_model, self).__init__()

        self.backbone = backbone
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(input_feature, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 10)

    def finetune_params(self):
        return self.backbone.parameters()
    
    def fresh_params(self):
        return list(self.fc1.parameters(), self.fc2.parameters())

    def forward(self, x):
        x = self.backbone(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
if __name__ == '__main__':
    # print(resnet50())
    model = Uniqlo_price_cls_model()
    x = torch.rand((1, 3, 512, 256))
    y= model(x)
    print(y.shape)
        

        