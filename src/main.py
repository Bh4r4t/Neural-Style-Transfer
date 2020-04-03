import argparse
import torch
from torchvision import models
from torch import optim
import matplotlib.pyplot as plt
from matplotlib.image import imsave
import style_and_content as sac, utils

def getArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--style_image", type=str, default=None)
    parser.add_argument("--content_image", type=str, default=None)
    parser.add_argument("--target_image", type=str, default='./result/final.jpg')
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--img_size", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=1,
                        help="style weight")
    parser.add_argument("--beta", type=float, default=1e6,
                        help="content weight")
    return parser.parse_args()


def styleFeat(styleFeatures):
    style_grams = {}
    for layer in styleFeatures:
        style_grams[layer] = sac.gramMatrix(styleFeatures[layer])

    return style_grams

def train(style_grams, content_features, target, model, optimizer, args):
    style_weight = {'conv1_1':1.,
                'conv2_1':.75,
                'conv3_1':.75,
                'conv4_1':0.50,
                'conv5_1':0.2,
    }

    for i in range(args.epochs):
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

        total_loss = style_loss*args.beta + content_loss*args.alpha

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print('Epoch: {}/{} | TotalLoss: {:.04f}'.format(i+1, args.epochs, total_loss))
        if (i+1)%200 == 0:
            plt.imshow(utils.DeNormalize(target))
            plt.show()
    
    imsave(args.target_image, utils.DeNormalize(target))
    print('Image Generated!')

def main(args):
    # check for cuda compatible accelerator
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # loading style and content image
    style, content = utils.LoadImage(args.style_image, args.content_image, device)
    # initializing target image
    target = content.clone().requires_grad_(True).to(device)
    model = sac.Net().to(device)

    # extracting style and content features from style and content images
    style_features = sac.getFeatures(model, style)
    content_features = sac.getFeatures(model, content)
    
    # gram matrix from style features
    style_gram = styleFeat(style_features)

    optimizer = optim.Adam([target], args.lr)
    train(style_gram, content_features, target, model, optimizer, args)


if __name__ == "__main__":
    args=getArguments()
    main(args)