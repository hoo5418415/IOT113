# Iris Classification using TensorFlow and TensorBoard

## CRISP-DM Steps

### Step 1: Business Understanding
The objective of this project is to classify Iris flowers into one of the three species (setosa, versicolor, virginica) using petal and sepal dimensions. We aim to build a neural network model using TensorFlow that achieves a high level of accuracy for this classification task.

### Step 2: Data Understanding
We use the popular Iris dataset, which consists of 150 samples of iris flowers, with the following features:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

The target variable has three classes, representing the three species of iris flowers.

### Step 3: Data Preparation
We prepare the data as follows:
- Split the dataset into training and testing sets (80-20 split).
- Standardize the feature values to have zero mean and unit variance, which is essential for training neural networks effectively.

```python
import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import TensorBoard
import datetime

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### Step 4: Modeling
We define a neural network model using TensorFlow's Keras API:
- Input layer: 4 neurons (for the 4 features)
- Two hidden layers with 32 neurons each and ReLU activation
- Dropout layer to reduce overfitting
- One hidden layer with 16 neurons and ReLU activation
- Output layer with 3 neurons and softmax activation (for the 3 classes)

```python
# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(4,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])
```

### Step 5: Model Evaluation Setup
The model is compiled using:
- Adam optimizer: An adaptive learning rate optimizer
- Sparse categorical cross-entropy loss: Suitable for multi-class classification with integer labels
- Accuracy metric: To evaluate the performance during training and testing

```python
# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

### Step 6: Training
We train the model using the training set, with 100 epochs, and validate it using the test set. A TensorBoard callback is used to visualize training metrics.

```python
# Set up TensorBoard callback
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train the model
model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), callbacks=[tensorboard_callback])
```

### Step 7: Evaluation
We evaluate the model's performance on the test set to determine its accuracy.

```python
# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc}')
```

### Step 8: Deployment and Monitoring
To visualize the training process, TensorBoard can be used. Run the following command in your terminal to start TensorBoard:

```sh
tensorboard --logdir=logs/fit
```

This will allow you to monitor the loss and accuracy during training and help you understand the model's learning behavior.

### Summary
This project follows the CRISP-DM process to build a neural network model for Iris flower classification. TensorBoard is used to visualize the training metrics, providing valuable insights into the model's performance during the training phase.
