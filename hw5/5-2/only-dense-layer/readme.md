# chatgpt prompt
Write a README for an MNIST classification project using a fully connected neural network. Follow the CRISP-DM methodology, including sections for data collection, preprocessing, transformation, model training, and evaluation. Use TensorFlow's Keras API with a simple model consisting of a flatten layer, two hidden dense layers, and an output layer. Include details on requirements, how to run the script, and expected output.

# MNIST Classification using a Fully Connected Neural Network

This project uses a simple neural network to solve the MNIST classification problem using only dense (fully connected) layers.

## Steps Followed (CRISP-DM Methodology)

### 1. Data Collection
- We use the MNIST dataset, which is a collection of 28x28 grayscale images of handwritten digits from 0 to 9.
- The dataset is loaded using `tensorflow.keras.datasets.mnist`.

### 2. Data Preprocessing
- Images are normalized to a range between 0 and 1 for better training efficiency.
- Labels are one-hot encoded to convert them into categorical format suitable for classification.

### 3. Data Transformation
- The input images are flattened into 1D vectors to be compatible with dense layers.

### 4. Data Mining (Model Training)
- A fully connected neural network is created using the `Sequential` API from `tensorflow.keras`.
- The model consists of:
  - A `Flatten` layer to convert 28x28 images into 1D vectors.
  - Two hidden dense layers with ReLU activation.
  - An output layer with 10 units and softmax activation.
- The model is compiled with the `adam` optimizer and categorical crossentropy loss function.
- The model is trained for 10 epochs with a batch size of 32.

### 5. Evaluation
- The model is evaluated on the test set, and the accuracy is printed.