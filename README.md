# ConditionalConvolution
Implements a Tensorflow Keras layer that extends a Keras convolution layer to condition the output for each image on an additional vector. The idea is to improve convolution expressiveness in cases where additional information can be supplied to reduce the domain of the transformation.

# Usage
The `CondConv` layer is defined in `layers.py` and wraps an existing Keras `Conv1D`, `Conv2D`, etc. layer to add a dense result to the output.

# Applications
Here are some potential applications for this layer:
* Conditional image-to-image mapping
* Conditional object detection/recognition
* Interactive image captioning
