# convis
A tool to visualize convolutional layer activations on an input image.  

Creates images in which the input image is highlighted by each filter in the given conv layer.

<img src="https://raw.githubusercontent.com/htoyryla/convis/master/tubingen-conv3_2-17.png" width="480">

### Dependencies:

PyTorch 

### Usage:

```
python convis.py -image examples/inputs/tubingen.jpg -model models/vgg19-d01eb7cb.pth -layer conv2_2 -output_dir output -seed 876
```
 
You can also place `convis.py` in your neural-style-pt directory, in order to to more easily work with models and input images. 
 
### Parameters:

* `-input_image`: Path to the input image.
* `-image_size`: Maximum side length (in pixels) of of the generated image. Default is 512.
* `-layer`: The target layer. Default is `relu4_2`
* `-seed`: An optional value which makes all random values deterministic. 
* `-pooling`: The type of pooling layers to use; one of `max` or `avg`. Default is `max`.
* `-model_file`: Path to the `.pth` file for the VGG or NIN model.
* `-output_image`: Name of the output image. Default is `out.png`.
* `-output_dir`: Name of the output image directory. Default is `output`.

The output files will be named like `output/tubingen-conv3_2-69.png`
