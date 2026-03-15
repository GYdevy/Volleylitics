import torch
import torch.nn as nn
from torchvision import models

def build_model():

    model = models.resnet18(pretrained=True)

    model.fc = nn.Linear(
        model.fc.in_features,
        2   # rally / non-rally
    )

    return model
