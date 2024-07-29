# Dogs vs. Cats Classification

This project involves building a Convolutional Neural Network (CNN) to classify images of dogs and cats. The dataset is sourced from Kaggle and includes a set of labeled images used to train and evaluate the model.

## Key Features:

- **Data Preparation**: Downloads and extracts images from Kaggle, preprocesses them by resizing and normalizing.
- **Model Architecture**: Defines a CNN with multiple convolutional layers, batch normalization, max pooling, dense layers, and dropout for regularization.
- **Training**: Compiles and trains the model using TensorFlow and Keras, with monitoring of both training and validation performance.
- **Evaluation**: Plots accuracy and loss metrics to assess model performance.
- **Prediction**: Includes functionality to classify new images as either a dog or a cat.

## Requirements:

- Python 3.x
- TensorFlow
- Keras
- NumPy
- OpenCV
- Matplotlib
- Kaggle API key

## Setup

Kaggle API Key:

Obtain your Kaggle API key from Kaggle.

Save it as a file named kaggle.json.

Place kaggle.json in the ~/.kaggle/ directory. Create the directory if it does not exist:

bash
Copy code
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
### Download the Dataset:

The dataset will be downloaded and extracted automatically by the script. Ensure the Kaggle API key is correctly configured.



1. **Install Dependencies**:

   Ensure you have the necessary Python packages installed. You can install them using pip:

   ```bash
   pip install tensorflow keras numpy opencv-python matplotlib kaggle
