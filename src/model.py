import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Shared layers
        self.conv1 = nn.Conv2d(1, 32, 3)   # 28 → 26
        self.pool = nn.MaxPool2d(2, 2)     # /2
        self.conv2 = nn.Conv2d(32, 64, 3)  # 13 → 11
        
        self.fc1 = nn.Linear(64 * 5 * 5, 128)  # FIXED
        
        # Personalized layer
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 28 → 26 → 13
        x = self.pool(F.relu(self.conv2(x)))   # 13 → 11 → 5
        
        x = torch.flatten(x, 1)                # 64 * 5 * 5 = 1600
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_shared_params(model):
    return [
        p.detach().cpu().numpy()
        for name, p in model.named_parameters()
        if "fc2" not in name
    ]

# def set_shared_params(model, params):
#     i = 0
#     for name, p in model.named_parameters():
#         if "fc2" not in name:
#             p.data = torch.tensor(params[i])
#             i += 1

def set_shared_params(model, params):
    i = 0
    for name, p in model.named_parameters():
        if "fc" not in name:
            p.data = torch.tensor(params[i])
            i += 1

# def get_params(model, personalized):
#     if personalized:
#         return [
#             p.detach().cpu().numpy()
#             for name, p in model.named_parameters()
#             if "fc2" not in name
#         ]
#     else:
#         return [p.detach().cpu().numpy() for p in model.parameters()]

def get_params(model, personalized):
    if personalized:
        return [
            p.detach().cpu().numpy()
            for name, p in model.named_parameters()
            if "fc" not in name  # last layer local
        ]
    else:
        return [p.detach().cpu().numpy() for p in model.parameters()]


import torchvision.models as models
import torch.nn as nn

class ResNetPersonalized(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.base = models.resnet18(pretrained=False)

        num_ftrs = self.base.fc.in_features
        self.base.fc = nn.Linear(num_ftrs, 10)

    def forward(self, x):
        return self.base(x)