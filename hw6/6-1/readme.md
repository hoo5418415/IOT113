# Face Mask Detection with Transfer Learning

This project utilizes transfer learning with the VGG19 model to classify images into two categories: **with mask** and **without mask**. The dataset is sourced from a GitHub repository, and the implementation follows the CRISP-DM methodology.

## Steps

### Step 1: Build the Model
- The base model is **VGG19**, pretrained on the ImageNet dataset.
- The top layers are replaced with a Flatten layer, a Dense layer, and an output layer with two neurons for binary classification.
- The pretrained layers are frozen to retain their feature extraction capabilities.

### Step 2: Load the Dataset
- The dataset is cloned from [Face-Mask-Detection](https://github.com/chandrikadeb7/Face-Mask-Detection).
- It contains two folders: `with_mask` and `without_mask`.
  - Images in `with_mask` are labeled as **1**.
  - Images in `without_mask` are labeled as **0**.
- The dataset is split into:
  - **80%** for training
  - **10%** for validation
  - **10%** for testing

### Step 3: Train the Model
- The model is trained for 10 epochs with a batch size of 32.
- Accuracy is evaluated on the test set.

### Step 4: Predict and Visualize
- The first 10 test images are displayed along with their predicted and true labels.
- External images can be classified using an image URL.

## Requirements

- TensorFlow
- scikit-learn
- matplotlib
- numpy
- requests
- PIL (Pillow)

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/chandrikadeb7/Face-Mask-Detection.git
   ```
2. Install the required Python packages:
   ```bash
   pip install tensorflow scikit-learn matplotlib numpy requests pillow
   ```
3. Run the Python script:
   ```bash
   python face_mask_detection.py
   ```

## Example Usage

### Predicting External Images

To classify an external image, replace `<your_image_url_here>` in the script with the desired image URL:
```python
image_url = "<your_image_url_here>"
print(f"The image is classified as: {classify_image(image_url)}")
```

The output will indicate whether the image is classified as **with mask** or **without mask**.

## Dataset

The dataset is sourced from [Face-Mask-Detection](https://github.com/chandrikadeb7/Face-Mask-Detection), which contains:
- **with_mask:** Images of people wearing masks.
- **without_mask:** Images of people without masks.

## Results

- The model achieves an accuracy of approximately **XX.XX%** on the test set (replace with actual results).
- Below is a visualization of predictions:

![Sample Predictions](sample_predictions.png)