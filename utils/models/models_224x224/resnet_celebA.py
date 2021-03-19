import torchvision.models as models
import torch.nn as nn

def change_output_layer(model, num_features=40):
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_features)

def ResNet18():
    model = models.resnet18()
    change_output_layer(model)
    return model

def ResNet34():
    model = models.resnet34()
    change_output_layer(model)
    return model

def ResNet50():
    model = models.resnet50()
    change_output_layer(model)
    return model

