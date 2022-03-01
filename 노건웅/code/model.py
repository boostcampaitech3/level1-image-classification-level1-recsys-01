import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
import torchvision.models as models


class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


# Custom Model Template
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=18) 

        torch.nn.init.xavier_uniform_(self.model._fc.weight)
        stdv = 1. / math.sqrt(self.model._fc.weight.size(1))
        self.model._fc.bias.data.uniform_(-stdv, stdv)


    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.model(x)
        return x

class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=18) 

        torch.nn.init.xavier_uniform_(self.model._fc.weight)
        stdv = 1. / math.sqrt(self.model._fc.weight.size(1))
        self.model._fc.bias.data.uniform_(-stdv, stdv)


    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.model(x)
        return x

class MyDeitModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/deit:main', 'deit_base_distilled_patch16_224', pretrained=True)
        self.model.head = nn.Linear(768,18)
        self.model.head_dist = nn.Linear(768,18)

        torch.nn.init.xavier_uniform_(self.model.head.weight)
        stdv = 1. / math.sqrt(self.model.head.weight.size(1))
        self.model.head.bias.data.uniform_(-stdv, stdv)

        torch.nn.init.xavier_uniform_(self.model.head_dist.weight)
        stdv = 1. / math.sqrt(self.model.head_dist.weight.size(1))
        self.model.head_dist.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        return self.model(x)
