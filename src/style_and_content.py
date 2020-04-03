import torch
from torchvision import models

def Net():
    # download the pretrained vgg19 model
    # as we only want to extract features, so no need of the classifier(FCN)
    vgg = models.vgg19(pretrained=True).features
    for params in vgg:
        params.requires_grad_(False)

    return vgg

def gramMatrix(tensor):
    _, d, h, w = tensor.size()
    
    # resizing the tensor into 2D featureMatrix
    # such that the all the feature values of a layer are in a single row of mod_tensor
    mod_tensor = tensor.view(d, h*w)

    return torch.mm(mod_tensor, mod_tensor.T)

def getFeatures(model, image):
    # defining a dict of all the layers that will
    # be used to extract features and style from out model according to the paper
    layers = {'0':'conv1_1',
                '5':'conv2_1',
                '10':'conv3_1',
                '19':'conv4_1',
                '21':'conv4_2',
                '28':'conv5_1',
    }

    style_features = {}
    x = image

    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            style_features[layers[name]] = x
    
    return style_features
