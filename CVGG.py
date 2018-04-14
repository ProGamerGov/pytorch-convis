import torch
import torch.nn as nn


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )


class NIN(nn.Module):
    def __init__(self, pooling):
        super(NIN, self).__init__()
        pool2d = None
        if pooling == 'max':
            pool2d = nn.MaxPool2d((3, 3),(2, 2),(0, 0),ceil_mode=True)
        elif pooling == 'avg':
            pool2d = nn.AvgPool2d((3, 3),(2, 2),(0, 0),ceil_mode=True)

        self.features = nn.Sequential(
            nn.Conv2d(3,96,(11, 11),(4, 4)),
	    nn.ReLU(inplace=True),
	    nn.Conv2d(96,96,(1, 1)),
	    nn.ReLU(inplace=True),
	    nn.Conv2d(96,96,(1, 1)),
	    nn.ReLU(inplace=True),
            pool2d,
	    nn.Conv2d(96,256,(5, 5),(1, 1),(2, 2)),
	    nn.ReLU(inplace=True),
	    nn.Conv2d(256,256,(1, 1)),
	    nn.ReLU(inplace=True),
	    nn.Conv2d(256,256,(1, 1)),
	    nn.ReLU(inplace=True),
            pool2d,
	    nn.Conv2d(256,384,(3, 3),(1, 1),(1, 1)),
	    nn.ReLU(inplace=True),
	    nn.Conv2d(384,384,(1, 1)),
	    nn.ReLU(inplace=True),
	    nn.Conv2d(384,384,(1, 1)),
	    nn.ReLU(inplace=True),
            pool2d,
	    nn.Dropout(0.5),
	    nn.Conv2d(384,1024,(3, 3),(1, 1),(1, 1)),
	    nn.ReLU(inplace=True),
	    nn.Conv2d(1024,1024,(1, 1)),
	    nn.ReLU(inplace=True),
	    nn.Conv2d(1024,1000,(1, 1)),
	    nn.ReLU(inplace=True),
	    nn.AvgPool2d((6, 6),(1, 1),(0, 0),ceil_mode=True),
	    nn.Softmax(),
        )
 

		
def BuildSequential(channel_list, pooling):
    layers = []
    in_channels = 3
    if pooling == 'max':
       pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
    elif pooling == 'avg':
       pool2d = nn.AvgPool2d(kernel_size=2, stride=2)
    else: 
       print("Unrecognized pooling parameter")
       quit()
    for c in channel_list:
        if c == 'P':
            layers += [pool2d]
        else:
            conv2d = nn.Conv2d(in_channels, c, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = c
    return nn.Sequential(*layers)


channel_list = {
    'D': [64, 64, 'P', 128, 128, 'P', 256, 256, 256, 'P', 512, 512, 512, 'P', 512, 512, 512, 'P'],
    'E': [64, 64, 'P', 128, 128, 'P', 256, 256, 256, 256, 'P', 512, 512, 512, 512, 'P', 512, 512, 512, 512, 'P'],
}

nin_dict = {
'C': ['conv1', 'cccp1', 'cccp2', 'conv2', 'cccp3', 'cccp4', 'conv3', 'cccp5', 'cccp6', 'conv4-1024', 'cccp7-1024', 'cccp8-1024'], 
'R': ['relu0', 'relu1', 'relu2', 'relu3', 'relu5', 'relu6', 'relu7', 'relu8', 'relu9', 'relu10', 'relu11', 'relu12'],
'P': ['pool1', 'pool2', 'pool3', 'pool4'],
'D': ['drop'],
}
vgg16_dict = {
'C': ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3'],
'R': ['relu1_1', 'relu1_2', 'relu2_1', 'relu2_2', 'relu3_1', 'relu3_2', 'relu3_3', 'relu4_1', 'relu4_2', 'relu4_3', 'relu5_1', 'relu5_2', 'relu5_3'],
'P': ['pool1', 'pool2', 'pool3', 'pool4', 'pool5'],
}
vgg19_dict = {
'C': ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4', 'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4', 'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4'],
'R': ['relu1_1', 'relu1_2', 'relu2_1', 'relu2_2', 'relu3_1', 'relu3_2', 'relu3_3', 'relu3_4', 'relu4_1', 'relu4_2', 'relu4_3', 'relu4_4', 'relu5_1', 'relu5_2', 'relu5_3', 'relu5_4'],
'P': ['pool1', 'pool2', 'pool3', 'pool4', 'pool5'],
}


def vgg19(pooling, **kwargs):
    # VGG 19-layer model (configuration "E")
    model = VGG(BuildSequential(channel_list['E'], pooling), **kwargs)
    return model, vgg19_dict

def vgg16(pooling, **kwargs):
    # VGG 16-layer model (configuration "D")
    model = VGG(BuildSequential(channel_list['D'], pooling), **kwargs)
    return model, vgg16_dict

def nin(pooling, **kwargs):
    # Network In Network model 
    model = NIN(pooling)
    return model, nin_dict
