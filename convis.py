# A PyTorch implementation of: https://gist.github.com/ProGamerGov/8f0560d8aea77c8c39c4d694b711e123
import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from PIL import Image
from torch.autograd import Variable
from CVGG import vgg19, vgg16, nin

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-input_image", help="Input target image", default='examples/inputs/tubingen.jpg')
parser.add_argument("-image_size", help="Maximum height / width of generated image", type=int, default=512)
parser.add_argument("-model_file", type=str, default='vgg19-d01eb7cb.pth')
parser.add_argument("-seed", type=int, default=-1)
parser.add_argument("-layer", help="layers for examination", default='relu2_2')
parser.add_argument("-pooling", help="max or avg pooling", type=str, default='max')
parser.add_argument("-output_image", default='out.png')
parser.add_argument("-output_dir", default='output')
params = parser.parse_args()


# Optionally set the seed value
if params.seed >= 0:
    torch.manual_seed(params.seed)

# Preprocess an image before passing it to a model: 
def ImageSetup(image_name, image_size):
    image = Image.open(image_name)
    image = image.convert('RGB')
    Loader = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])  # resize and convert to tensor
    rgb2bgr = transforms.Compose([transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]) ])
    Normalize = transforms.Compose([transforms.Normalize(mean=[103.939, 116.779, 123.68], std=[1,1,1]) ]) # Subtract BGR
    tensor = Variable(Normalize(rgb2bgr(Loader(image) * 256))).unsqueeze(0)
    return tensor
 
# Undo the above preprocessing and save the tensor as an image:
def SaveImage(output_tensor, output_name):
    Normalize = transforms.Compose([transforms.Normalize(mean=[-103.939, -116.779, -123.68], std=[1,1,1]) ]) # Add BGR
    bgr2rgb = transforms.Compose([transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]) ])
    ResizeImage = transforms.Compose([transforms.Resize(params.image_size)])
    output_tensor = bgr2rgb(Normalize(output_tensor.squeeze(0))) / 256
    output_tensor.clamp_(0, 1)
    Image2PIL = transforms.ToPILImage()
    image = Image2PIL(output_tensor.cpu())
    image = ResizeImage(image)
    image.save(str(output_name))
    
    
img = ImageSetup(params.input_image, params.image_size).float()
output_filename, file_extension = os.path.splitext(params.output_image)
try:
    os.makedirs(params.output_dir)
except OSError: 
    pass


# Get the model class, and configure pooling layer type
def buildCNN(model_file, pooling):
    cnn = None
    layerList = []
    if "vgg19" in str(model_file):
        print("VGG-19 Architecture Detected")
        cnn, layerList = vgg19(pooling)
    elif "vgg16" in str(model_file):
        print("VGG-16 Architecture Detected")
        cnn, layerList = vgg16(pooling)
    elif "nin" in str(model_file):
        print("NIN Architecture Detected")
        cnn, layerList = nin(pooling)
    return cnn, layerList

  
def modelSetup(cnn, layerList):
    cnn = copy.deepcopy(cnn)
    net = nn.Sequential()  
    c, r, p = 0, 0, 0
    convName, reluName, poolName = None, None, None

    for layer in list(cnn):
        if isinstance(layer, nn.Conv2d):
            net.add_module(str(len(net)), layer)
            layerType = layerList['C']
            convName = layerType[c]
            c+=1

        if isinstance(layer, nn.ReLU):
            net.add_module(str(len(net)), layer)
            layerType = layerList['R']
            reluName = layerType[r]
            r+=1

        if isinstance(layer, nn.MaxPool2d) or isinstance(layer, nn.AvgPool2d):
            net.add_module(str(len(net)), layer)
            layerType = layerList['P']
            poolName = layerType[p]
            p+=1

        if convName == params.layer or reluName == params.layer or poolName == params.layer:
            return net

    return net
    
    
    
cnn, layerList = buildCNN(params.model_file, params.pooling)   
cnn.load_state_dict(torch.load(params.model_file))
cnn = cnn.features 
net = modelSetup(cnn, layerList)

y = net(img).squeeze(0)
n = y.size(1)

for i in xrange(n):
    y3 = torch.Tensor(3, y.size(1), y.size(2))
    y1 = y.clone().narrow(0,i,1)

    y3[0] = y1.data
    y3[1] = y1.data
    y3[2] = y1.data


    filename = str(params.output_dir) + "/" +str(output_filename) + "_" + str(i) + "_" + str(params.layer) + file_extension
    SaveImage(y3, filename)
