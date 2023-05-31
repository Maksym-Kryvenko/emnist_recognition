# Handwritten Character Recognition

This project focuses on recognizing handwritten characters of the Latin alphabet and numbers using Convolutional Neural Network (CNN) and Fully Connected Network (FCN) trained on the EMNIST dataset.

## Author
- [Maksym Kryvenko](https://www.linkedin.com/in/maksymkryvenko/)

## About
The goal of this project is to recognize handwritten characters. The EMNIST dataset, consisting of images representing characters as arrays of pixel intensities normalized between 0 and 1, is used for training the FCN/CNN model. The model architecture consists of two convolutional layers, each followed by a max pooling layer. These layers are then connected to two fully connected dense layers with 256 and 128 neurons respectively. Finally, the output layer with 62 nodes represents the classes of characters, with each node outputting a probability value between 0 and 1.

## Goal
The goal is to predict single character pictures in the specified location provided as input to the script.

## Dataset
The EMNIST ByClass split sample is used for this project, which includes all 62 classes of characters - 10 digits, 26 lowercase letters, and 26 uppercase letters. The dataset consists of 814,255 characters with 62 unbalanced classes.

```python
from emnist import extract_training_samples
from emnist import extract_test_samples

train_images, train_labels = extract_training_samples('byclass')
test_images, test_labels = extract_test_samples('byclass')
```

## Usage
To install all required Python packages:
```
pip install -r ./app/requirements.txt
```

To create and train model (pre-trained model already exists in directory):
```
python ./app/train.py
```

To make predictions for all files in a directory, use the following command:
```
python ./app/inference.py --input /data/test_data
```
**Note: The script takes the path to a directory containing pictures and returns predictions in the format "[character ASCII index in decimal format], [predicted character], [POSIX path to image sample]" and save into .csv file.**

## Input
As input .png and .jpg(jpeg) can be given in specified folder. Pictures sould be black-and-white or at least background and symbol collors should be contrast.

## Resulting Scores
The final trained model achieved the following scores:
```
- Accuracy: 0.8673
- Loss: 0.356
```
