# ResNet Implementation in Java

This project implements a Residual Neural Network (ResNet) in Java. ResNet is a deep learning architecture that addresses the vanishing gradient problem in very deep neural networks by introducing skip connections.

## Table of Contents

1. [Overview](#overview)
2. [Key Components](#key-components)
3. [Implementation Details](#implementation-details)
4. [Usage](#usage)
5. [Code Breakdown](#code-breakdown)

## Overview

The ResNet implementation includes the following main components:
- 2D Convolution
- Batch Normalization
- ReLU Activation
- Residual Blocks
- Global Average Pooling
- Fully Connected Layer
- Softmax Activation

The network is built using these components to create a deep architecture capable of learning complex features from input data.

## Key Components

### 1. Convolution2D
- Performs 2D convolution on the input tensor
- Supports different filter sizes, strides, and padding options

### 2. Batch Normalization
- Normalizes the activations of the previous layer
- Helps in faster convergence and reduces internal covariate shift

### 3. ReLU Activation
- Applies the Rectified Linear Unit activation function
- Introduces non-linearity to the model

### 4. Residual Block
- Core building block of ResNet
- Includes skip connections to allow gradients to flow through the network

### 5. Global Average Pooling
- Reduces spatial dimensions of the tensor
- Helps in reducing the number of parameters

### 6. Fully Connected Layer
- Connects every neuron in one layer to every neuron in another layer

### 7. Softmax Activation
- Converts the output to probability distribution
- Used in the final layer for classification tasks

## Implementation Details

The ResNet is implemented as a Java class with static methods for each component. The main steps in building the network are:

1. Create an input tensor
2. Apply initial convolution and activation
3. Stack multiple residual blocks
4. Perform global average pooling
5. Apply fully connected layer
6. Use softmax activation for final output

## Usage

To use the ResNet implementation:

1. Set the input shape and number of classes
2. Call the `buildResNet` method
3. The method returns the final output probabilities



## Code Breakdown

Here's a breakdown of the main methods in the ResNet class:

- `convolution2d`: Implements 2D convolution operation
- `padTensor`: Adds padding to the input tensor
- `batchNormalization`: Performs batch normalization
- `relu`: Applies ReLU activation function
- `resnetBlock`: Implements a residual block
- `addTensors`: Adds two tensors element-wise
- `globalAveragePooling2d`: Performs global average pooling
- `fullyConnectedLayer`: Implements a fully connected layer
- `softmaxActivation`: Applies softmax activation
- `buildResNet`: Constructs the complete ResNet architecture

Each method is implemented to work with 3D tensors (height x width x channels) to process image-like inputs.

The main method demonstrates how to use the ResNet implementation by creating a sample input and running it through the network.

Note: This implementation uses random weights and biases for demonstration purposes. In a real-world scenario, these would be learned through training on a dataset.
