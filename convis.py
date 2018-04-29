import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from PIL import Image
from torch.autograd import Variable
from CaffeLoader import loadCaffemodel

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-input_image", help="Input target image", default='examples/inputs/tubingen.jpg')
parser.add_argument("-image_size", help="Maximum height / width of generated image", type=int, default=512)
parser.add_argument("-model_file", type=str, default='models/vgg19-d01eb7cb.pth')
parser.add_argument("-layer", help="layers for examination", default='relu2_2')
parser.add_argument("-pooling", help="max or avg pooling", type=str, default='max')
parser.add_argument("-output_image", default='out.png')
parser.add_argument("-output_dir", default='output')
params = parser.parse_args()


Image.MAX_IMAGE_PIXELS = 1000000000 # Support gigapixel images

def main(): 
    # Build the model definition and setup pooling layers:   
    cnn, layerList = loadCaffemodel(params.model_file, params.pooling, -1) 

    img = preprocess(params.input_image, params.image_size).float()    

    output_filename, file_extension = os.path.splitext(params.output_image)
    try:
        os.makedirs(params.output_dir)
    except OSError: 
        pass

    cnn = copy.deepcopy(cnn)
    net = nn.Sequential()  
    c, r, p = 0, 0, 0
    convName, reluName, poolName = None, None, None
    for layer in list(cnn):
        if isinstance(layer, nn.Conv2d):
            net.add_module(str(len(net)), layer)
            convName = layerList['C'][c]
            c+=1

        if isinstance(layer, nn.ReLU):
            net.add_module(str(len(net)), layer)
            reluName = layerList['R'][r]
            r+=1

        if isinstance(layer, nn.MaxPool2d) or isinstance(layer, nn.AvgPool2d):
            net.add_module(str(len(net)), layer) 
            poolName = layerList['P'][p]
            p+=1

        if convName == params.layer or reluName == params.layer or poolName == params.layer:
            break

    
    # Get the activations  
    y = net(img).squeeze(0)
    n = y.size(0)

    for i in range(n):
        y3 = torch.Tensor(3, y.size(1), y.size(2))
        y1 = y.clone().narrow(0,i,1)

        y3[0] = y1.data
        y3[1] = y1.data
        y3[2] = y1.data


        filename = str(params.output_dir) + "/" + str(output_filename) + "-" + str(params.layer) + "-" + str(i) + file_extension
        deprocess(y3, filename)
        print("Saving image: " + filename)

        if i == (n-1): 
            break


# Preprocess an image before passing it to a model.
# We need to rescale from [0, 1] to [0, 255], convert from RGB to BGR,
# and subtract the mean pixel.
def preprocess(image_name, image_size):
    image = Image.open(image_name).convert('RGB')
    image_size = tuple([int((float(image_size) / max(image.size))*x) for x in (image.height, image.width)]) 
    Loader = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])  # resize and convert to tensor
    rgb2bgr = transforms.Compose([transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]) ])
    Normalize = transforms.Compose([transforms.Normalize(mean=[103.939, 116.779, 123.68], std=[1,1,1]) ]) # Subtract BGR
    tensor = Variable(Normalize(rgb2bgr(Loader(image) * 256))).unsqueeze(0)
    return tensor
 
# Undo the above preprocessing and save the tensor as an image:
def deprocess(output_tensor, output_name):
    image = Image.open(params.input_image).convert('RGB')
    image_size = tuple([int((float(params.image_size) / max(image.size))*x) for x in (image.height, image.width)]) 
    Normalize = transforms.Compose([transforms.Normalize(mean=[-103.939, -116.779, -123.68], std=[1,1,1]) ]) # Add BGR
    bgr2rgb = transforms.Compose([transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]) ])
    ResizeImage = transforms.Compose([transforms.Resize(image_size)])
    output_tensor = bgr2rgb(Normalize(output_tensor.squeeze(0))) / 256
    output_tensor.clamp_(0, 1)
    Image2PIL = transforms.ToPILImage()
    image = Image2PIL(output_tensor.cpu())
    image = ResizeImage(image)
    image.save(str(output_name))
    
    
if __name__ == "__main__":
    main()
