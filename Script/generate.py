import style_and_content as sac, utils
import torch
from torchvision import models
from torch import optim
import matplotlib.pyplot as plt
from matplotlib.image import imsave

def styleFeat(styleFeatures):
    style_grams = {}
    for layer in styleFeatures:
        style_grams[layer] = sac.gramMatrix(styleFeatures[layer])

    return style_grams

def train(style_grams, content_features, target, model, epochs, optimizer, alpha = 1, beta = 1e6):
    style_weight = {'conv1_1':1.,
                'conv2_1':.75,
                'conv3_1':.75,
                'conv4_1':0.50,
                'conv5_1':0.2,
    }

    for i in range(epochs):
        target_features = sac.getFeatures(model, target)
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)

        style_loss = 0
        for layer in style_weight:
            target_feature = target_features[layer]
            target_gram = sac.gramMatrix(target_feature)
            _, d, h, w = target_feature.shape
            style_gram = style_grams[layer]
            style_gram_loss = style_weight[layer]*torch.mean((style_gram - target_gram)**2)
            style_loss += style_gram_loss/(d*h*w)

        total_loss = style_loss*beta + content_loss*alpha

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print('Loss_is: {:.03f}'.format(total_loss))
    
    imsave('final.png', utils.DeNormalize(target))

def main(img1, img2, epochs, lr=0.01, alpha = 1, beta = 1e6):
    style, content = utils.LoadImage(img1, img2)
    target = content.clone().requires_grad_(True)
    model = sac.Net()
    style_features = sac.getFeatures(model, style)
    content_features = sac.getFeatures(model, content)
 
    style_gram = styleFeat(style_features)

    optimizer = optim.Adam([target], lr)

    train(style_gram, content_features, target, model, epochs, optimizer, alpha, beta)

if __name__ == "__main__":
    main('C:/Users/Ladre/Desktop/images.jpg', 'C:/Users/Ladre/Desktop/images1.jpg', 1000)