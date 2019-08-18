import torch
import torch.nn as nn
from torchvision import datasets, models, transforms

def set_parameter_requires_grad(model):
    i = 0
    for child in model.children():
        i = i + 1
        if i <=17:
            for param in child.parameters():
                param.requires_grad = False

class inception_model(nn.Module):
    def __init__(self):
        aux_logits = False
        super(inception_model, self).__init__()
        self.inception = models.inception_v3(pretrained=True)
        # print((self.inception).size())
        set_parameter_requires_grad(self.inception)
        # Handle the auxilary net
        self.num_ftrs = self.inception.AuxLogits.fc.in_features
        self.inception.AuxLogits.fc = nn.Linear(self.num_ftrs, 14)
        # Handle the primary net
        self.num_ftrs = self.inception.fc.in_features
        self.inception.fc = nn.Linear(self.num_ftrs,14)
        # print(self.inception.fc)
        self.input_size = 299

    def forward(self, x):
        print(x.size())
        # print(self.inception(x).size())
        return self.inception(x)
        
        # return self.inception(x).double()


class vgg_on_images(nn.Module):
    def __init__(self):
        super(vgg_on_images, self).__init__()
        self.vgg_model = models.vgg11_bn(pretrained=True)
        set_parameter_requires_grad(self.vgg_model)
        self.number_features = self.vgg_model.classifier[6].in_features
        self.vgg_model.classifier[6] = nn.Linear(self.number_features,14)
        self.input_size = 224

    def forward(self, x):
        return self.vgg_model(x).double()


class feature_extractor(nn.Module):
    def __init__(self):
        super(feature_extractor, self).__init__()
        self.resnet_model = models.resnet18(pretrained=True)
        set_parameter_requires_grad(self.resnet_model)
        self.number_features = self.resnet_model.fc.in_features
        self.resnet_model.fc = nn.Linear(self.number_features, 14)
        self.input_size = 224


    def forward(self, x):
        return self.resnet_model(x).double()
