# Dog-vs-Cat-Image-Classification

 Dataset Link:-https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbjVqeDhVb1ZuQWRqUWFDSnhxblYweGo1ZG5nZ3xBQ3Jtc0ttcEd3UVh5NTJwem43STBIZG95R194cl9lYlN0TFo3LWxfZ3l5QWY4WG93MDZWaWdOakVFbVVSd3VicFhrNGlOaUt5OExDNFE4VFVpMko3UjhaYmpYZUI5ZXBMS0JIN0loMU40TGtrckJWak5FczRidw&q=https%3A%2F%2Fwww.kaggle.com%2Fdatasets%2Fsalader%2Fdogs-vs-cats&v=0K4J_PTgysc

 Project Summary
This project involves building a Convolutional Neural Network (CNN) to classify images of dogs and cats. The dataset is sourced from Kaggle and includes a set of labeled images used to train and evaluate the model.

Key Features:
Data Preparation: Downloads and extracts images from Kaggle, preprocesses them by resizing and normalizing.
Model Architecture: Defines a CNN with multiple convolutional layers, batch normalization, max pooling, dense layers, and dropout for regularization.
Training: Compiles and trains the model using TensorFlow and Keras, with monitoring of both training and validation performance.
Evaluation: Plots accuracy and loss metrics to assess model performance.
Prediction: Includes functionality to classify new images as either a dog or a cat.
Requirements:
Python 3.x
TensorFlow
Keras
NumPy
OpenCV
Matplotlib
Kaggle API key
Usage:
Install Dependencies: Install required Python packages using pip.
Kaggle API Setup: Obtain and configure your Kaggle API key.
Download and Prepare Data: The script handles dataset download and extraction.
Train the Model: Run the training script to build and evaluate the model.
Classify Images: Use the trained model to make predictions on new images.
This project provides a practical example of using deep learning for image classification tasks, demonstrating data preparation, model training, and evaluation.ry Python packages installed. You can install them using pip:

bash
Copy code
pip install tensorflow keras numpy opencv-python matplotlib kaggle
Kaggle API Key:

Obtain your Kaggle API key from Kaggle.

Save it as a file named kaggle.json.

Place kaggle.json in the ~/.kaggle/ directory. Create the directory if it does not exist:

bash
Copy code
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
Download the Dataset:

The dataset will be downloaded and extracted automatically by the script. Ensure the Kaggle API key is correctly configured.

Code Overview
1. Data Preparation
The dataset is downloaded from Kaggle and unzipped. The images are then divided into training and validation datasets using TensorFlow's image_dataset_from_directory.

2. Model Definition
A Convolutional Neural Network (CNN) is built with the following architecture:

Convolutional Layers: Extract features from the images.
Batch Normalization: Normalize activations to improve training.
Max Pooling Layers: Reduce the spatial dimensions.
Flatten Layer: Flatten the output from the convolutional layers.
Dense Layers: Fully connected layers for classification.
Dropout: Regularization to prevent overfitting.
3. Training
The model is compiled with the Adam optimizer and binary cross-entropy loss function. It is trained for 10 epochs with both training and validation datasets.

4. Evaluation
Training and validation accuracy and loss are plotted using Matplotlib to visualize the model's performance.

5. Prediction
The model is used to classify new images. The input image is resized and reshaped to match the model's expected input shape.

Usage
Train the Model:

Run the script to start training the model:

bash
Copy code
python Dogs-v-Cat-Classification.py
Classify New Images:

Replace the path in cv2.imread('/content/dog.94.jpg') with the path to your image. The model will predict whether the image is of a dog or a cat.

Example
The provided code includes an example of classifying a new image. Ensure the image is in the correct format and resize it to (256, 256) before feeding it into the model.

python
Copy code
import cv2
test_img = cv2.imread('/content/dog.94.jpg')
plt.imshow(test_img)
plt.show()

test_img = cv2.resize(test_img, (256, 256))
test_input = test_img.reshape((1, 256, 256, 3))

prediction = model.predict(test_input)

if prediction[0][0] == 1:
    print("dog image")
else:
    print("cat image")
