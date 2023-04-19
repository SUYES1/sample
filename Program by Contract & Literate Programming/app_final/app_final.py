"""
 This is a Python Flask web application that generates a caption
 for an uploaded image and converts the caption to speech.
 The user uploads an image file, which is saved to a specified path. 
 The image is then used to generate a caption and a sound file for the caption. 
 The output is rendered to the home page, where the user can view the image, caption, and hear the speech.
"""


# IMPORTS
from flask import Flask, render_template, request
import Caption_it
import os


# FLASK APP SETUP
app = Flask(__name__)


# HOME PAGE
@app.route('/')
def home():
    """
    Renders the home page of the web application.
    """
    return render_template("index.html")


# HANDLE POST REQUESTS
@app.route('/', methods=['POST'])
def process_upload():
    """
    Handles the POST request sent when the user uploads an image.
    """
    # PRECONDITIONS
    assert request.method == 'POST', "Invalid request method"
    assert 'userfile' in request.files, "No file was uploaded"
    
    # GET THE UPLOADED IMAGE FILE
    f = request.files['userfile']
    
    # SET PATH TO SAVE THE IMAGE FILE
    path = "./static/{}".format(f.filename)
    
    # SAVE IMAGE FILE TO PATH
    f.save(path)

    # GENERATE CAPTION FOR THE UPLOADED IMAGE
    caption = Caption_it.caption_this_image(path)
    
    # GENERATE SPEECH FILE FOR CAPTION
    sound_file = Caption_it.text2speech(caption)

    # POSTCONDITIONS
    assert isinstance(caption, str), "Caption is not a string"
    assert isinstance(sound_file, str), "Speech file is not a string"
    assert os.path.exists(sound_file), "Speech file was not created"
    
    # RENDER OUTPUT TO HOME PAGE
    return render_template("index.html", your_result={'caption': caption, 'sound': sound_file})


# RUN THE APPLICATION
if __name__ == '__main__':
    """
    Runs the Flask application with debug and threaded parameters.
    """
    app.run(debug=False, threaded=False)
