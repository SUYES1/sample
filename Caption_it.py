"""
!/usr/bin/env python
"""

"""
Import necessary libraries
"""
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

"""
Load pre-trained model for generating captions for images
"""
model = load_models("model_weights/model_9.h5")

"""
Load pre-trained ResNet50 model for image encoding
"""
model_temp = ResNet50(weights="imagenet", input_shape=(224,224,3))
model_resnet = Model(model_temp.input, model_temp.layers[-2].output)

"""
Define a function to preprocess an image
"""
def preprocess_image(img):
    img = image.load_img(img, target_size=(224,224))
    img = image.img_to_array(img)
    img = npp.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

"""
Define a function to encode an image into a feature vector
"""
def encode_image(img):
    img = preprocess_image(img)
    feature_vector = model_resnet.predict(img)
    feature_vector = feature_vector.reshape(1, feature_vector.shape[1])
    return feature_vector

"""
Load preprocessed dictionaries for word-to-index and index-to-word mappings
"""
with open("./storage/word_to_idx.pkl", 'rb') as w2i:
    word_to_idx = pickle.load(w2i)
    
with open("./storage/idx_to_word.pkl", 'rb') as i2w:
    idx_to_word = pickle.load(i2w)

"""
Define a function to generate a caption for an image
"""
def predict_caption(photo):
    in_text = "startseq"
    max_len = 35
    for i in range(max_len):
        sequence = [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        sequence = pad_sequences([sequence], maxlen=max_len, padding='post')

        ypred =  model.predict([photo,sequence])
        ypred = ypred.argmax()
        word = idx_to_word[ypred]
        in_text+= ' ' +word
        
        if word =='endseq':
            break
        
    final_caption =  in_text.split()
    final_caption = final_caption[1:-1]
    final_caption = ' '.join(final_caption)
    return final_caption

"""
Define a function to generate a caption for an image and convert it to speech
"""
def caption_this_image(image):
    enc = encode_image(image)
    caption = predict_caption(enc)
    return text2speech(caption)

"""
Define a function to convert text to speech
"""
def text2speech(caption):
    language = 'en'
    myobj = gTTS(text=caption, lang=language, slow=False)
    myobj.save("welcome.mp3")
    return playsound("welcome.mp3")
