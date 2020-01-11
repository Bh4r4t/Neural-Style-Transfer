from PIL import Image
import numpy as np
from torchvision import transforms

def LoadImage(img1, img2, device):
    #loading style and content images
    style = Image.open(img1)
    content = Image.open(img2)

    style_size = style.size
    #style_size = [width, height]
    content_size = content.size

    out_size = (min(style_size[1], content_size[1]), min(style_size[0], content_size[0]))
    #transforms.Resize takes (row, column) or (height, width)

    out_image = transforms.Compose([transforms.Resize(out_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    style_img = out_image(style).unsqueeze(0).to(device)
    content_img = out_image(content).unsqueeze(0).to(device)

    return style_img, content_img

def DeNormalize(img):
    img = img.to('cpu').clone()
    #with requires_grad tensors cannot be converted to numpy array, so use detach
    img = img.detach().numpy().squeeze()

    #img.shape = [no_of_channels, height, width]
    # but imshow expects the dimensions in the format [height, width, no_of_channels]
    img = np.transpose(img, (1,2,0))

    img = img*np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    img = img.clip(0,1)

    return img
