#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 15:21:16 2021

@author: erinlee
"""


# referenced pages 160 - 65 (Chapter 5.4) of the book by Chollet


# Chapter 5.4.1: Visualizing intermediate activations

from keras.models import load_model
model = load_model('CovidVsPneumoniav7.h5')
model.summary()

# Listing 5.25. Preprocessing a single image

img_path = '/Users/erinlee/Downloads/lung-images/Curated X-Ray Dataset/TrainingExampleV2/test/covid/COVID-19 (1280).jpg'

from keras.preprocessing import image
import numpy as np

img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

print("Tensor shape:",img_tensor.shape)

# Listing 5.26. Displaying the test picture

import matplotlib.pyplot as plt

print("displaying an image from the test set (from path: test -> covid -> COVID-19 (1280).jpg")
plt.imshow(img_tensor[0])
plt.show()

# Listing 5.27. Instantiating a model from an input tensor and a list of output tensors
from keras import models

layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

# Listing 5.28. Running the model in predict mode
activations = activation_model.predict(img_tensor)


first_layer_activation = activations[0]
print("First-layer activation:",first_layer_activation.shape)


# Listing 5.31. Visualizing every channel in every intermediate activation

print("\n Extracting & visualizing activation layers in one big image tensor:\n") # every layer activation of image from test data

layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)  # names of the layers as part of the plot (i.e. "conv2d_1", "max_pooling2d_1", ...)

images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations): # displays the feature maps
    n_features = layer_activation.shape[-1] # number of features in the feature map

    size = layer_activation.shape[1]

    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))

    for col in range(n_cols):   # tiles each filter into a big horizontal grid
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                                             :, :,
                                             col * images_per_row + row]
            channel_image -= channel_image.mean() # post-processes the feature to make it visually palatable
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size, # displays the grid
                         row * size : (row + 1) * size] = channel_image

    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')

    
# Referencing Chapter 5.4.2: Visualizing convnet filters:
  
    
# Listing 5.32. Defining the loss tensor for filter visualization
    
from keras.applications import VGG16
from keras import backend as K

model = VGG16(weights='imagenet',
              include_top=False)

layer_name = 'block3_conv1'
filter_index = 0

layer_output = model.get_layer(layer_name).output
loss = K.mean(layer_output[:, :, :, filter_index])

# Listing 5.33. Obtaining the gradient of the loss with regard to the input
grads = K.gradients(loss, model.input)[0]

# Listing 5.34. Gradient-normalization trick
grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

# Listing 5.35. Fetching Numpy output values given Numpy input values
iterate = K.function([model.input], [loss, grads])

import numpy as np
loss_value, grads_value = iterate([np.zeros((1, 150, 150, 3))])

# Listing 5.36. Loss maximization via stochastic gradient descent
input_img_data = np.random.random((1, 150, 150, 3)) * 20 + 128.

step = 1.
for i in range(40):
    loss_value, grads_value = iterate([input_img_data])

    input_img_data += grads_value * step

# Listing 5.37. Utility function to convert a tensor into a valid image
def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    x += 0.5
    x = np.clip(x, 0, 1)

    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# Listing 5.38. Function to generate filter visualizations
def generate_pattern(layer_name, filter_index, size=150):
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])

    grads = K.gradients(loss, model.input)[0]

    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    iterate = K.function([model.input], [loss, grads])

    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.

    step = 1.
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

    img = input_img_data[0]
    return deprocess_image(img)

# Figure 5.29. Pattern that the zeroth channel in layer block3_conv1 responds to maximally
#plt.imshow(generate_pattern('block3_conv1', 0))
