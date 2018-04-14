# Fixes from here: https://github.com/jcjohnson/pytorch-vgg/issues/3 
# Usage: python download_models.py or python3 download_models.py

import torch
import urllib
from collections import OrderedDict
from torch.utils.model_zoo import load_url


# Download the VGG-19 model and fix the layer names:
sd = load_url("https://s3-us-west-2.amazonaws.com/jcjohns-models/vgg19-d01eb7cb.pth")
map = {'classifier.1.weight':u'classifier.0.weight', 'classifier.1.bias':u'classifier.0.bias', 'classifier.4.weight':u'classifier.3.weight', 'classifier.4.bias':u'classifier.3.bias'}
sd = OrderedDict([(map[k] if k in map else k,v) for k,v in sd.iteritems()])
torch.save(sd, "models/vgg19-d01eb7cb.pth")

# Download the VGG-16 model and fix the layer names:
sd = load_url("https://s3-us-west-2.amazonaws.com/jcjohns-models/vgg16-00b39a1b.pth")
map = {'classifier.1.weight':u'classifier.0.weight', 'classifier.1.bias':u'classifier.0.bias', 'classifier.4.weight':u'classifier.3.weight', 'classifier.4.bias':u'classifier.3.bias'}
sd = OrderedDict([(map[k] if k in map else k,v) for k,v in sd.iteritems()])
torch.save(sd, "models/vgg16-00b39a1b.pth")

# Download the NIN model:
modelfile = urllib.URLopener()
modelfile.retrieve("https://raw.githubusercontent.com/ProGamerGov/pytorch-nin/master/nin_imagenet.pth", "models/nin_imagenet.pth")
