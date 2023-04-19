"""
This Python script generates captions for images and 
converts them into speech using Google Text-to-Speech 
API.It uses a pre-trained model for caption generation
and a pre-trained ResNet50 model for image encoding. 
The script loads preprocessed dictionaries for word-to-index
and index-to-word mappings and defines functions for preprocessing
an image, generating a caption for an image, and converting text to 
speech. The script can be used for generating captions for images and
converting them into speech, which could be helpful for individuals with 
visual impairments or in multimedia applications.
"""

# Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import tensorflow as tf
import pickle

from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Dense, Dropout, Embedding, LSTM
from keras.layers.merge import add

import os
from gtts import *
from playsound import playsound


# Contract for preprocess_image function
# Given: A string representing the path to an image file
# Returns: A NumPy array representing the preprocessed image
# Ensures: The returned array is of shape (1, 224, 224, 3) and contains preprocessed pixel values
def preprocess_image(img_path):
    """
    Preprocesses an image to prepare it for encoding into a feature vector.

    Parameters:
    img_path (str): The path to the image file.

    Returns:
    numpy.ndarray: A 4D NumPy array representing the preprocessed image.
    """
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    assert img.shape == (1, 224, 224, 3), f"Expected (1, 224, 224, 3), but got {img.shape}"

    return img


# Contract for encode_image function
# Given: A string representing the path to an image file
# Returns: A NumPy array representing the image's feature vector
# Ensures: The returned array is of shape (1, 2048)
def encode_image(img_path):
    """
    Encodes an image into a feature vector using a pre-trained ResNet50 model.

    Parameters:
    img_path (str): The path to the image file.

    Returns:
    numpy.ndarray: A 2D NumPy array representing the image's feature vector.
    """
    model_temp = ResNet50(weights="imagenet", input_shape=(224, 224, 3))
    model_resnet = Model(model_temp.input, model_temp.layers[-2].output)

    img = preprocess_image(img_path)
    feature_vector = model_resnet.predict(img)
    feature_vector = feature_vector.reshape(1, feature_vector.shape[1])

    assert feature_vector.shape == (1, 2048), f"Expected (1, 2048), but got {feature_vector.shape}"

    return feature_vector


# Contract for predict_caption function
# Given: A NumPy array representing an image's feature vector
# Returns: A string representing the generated caption
# Ensures: The returned string starts with "startseq", ends with "endseq",
