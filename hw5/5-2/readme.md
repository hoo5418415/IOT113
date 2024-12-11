# MNIST Classification with Dense NN and CNN

This repository demonstrates the application of CRISP-DM methodology to solve the MNIST handwritten digit classification problem using two neural network architectures:

1. **Dense Neural Network (Dense NN)**
2. **Convolutional Neural Network (CNN)**

## Dataset

The MNIST dataset is a standard dataset containing grayscale images of handwritten digits (0-9) with 60,000 training samples and 10,000 testing samples. Each image is 28x28 pixels.

## CRISP-DM Methodology

### 1. Business Understanding

The goal is to classify handwritten digits accurately using neural networks to demonstrate the capability of Dense NN and CNN for image classification tasks.

### 2. Data Understanding

The dataset is preloaded using `tensorflow.keras.datasets.mnist`. The dataset contains images and their corresponding labels (0-9).

### 3. Data Preparation

- Normalized pixel values to the range [0, 1].
- Reshaped data to include a channel dimension.
- One-hot encoded the labels for classification.
- Split the training data into training and validation sets.

### 4. Modeling

#### Dense Neural Network
- Input: Flattened 28x28 image (784 features).
- Architecture: Fully connected layers with ReLU activation and dropout for regularization.
- Output: Softmax layer with 10 units (one for each digit).

#### Convolutional Neural Network
- Input: 28x28 grayscale image with 1 channel.
- Architecture: Two convolutional layers with max pooling, followed by fully connected layers with dropout.
- Output: Softmax layer with 10 units.

### 5. Evaluation

Both models are evaluated on the test set using accuracy and classification reports. EarlyStopping and ModelCheckpoint callbacks are employed to enhance performance.

### 6. Deployment

The best models are saved using Keras' `.keras` format and can be reused for predictions or integrated into applications.
