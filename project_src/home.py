from PIL import Image
from flask import Flask, render_template, request, redirect, url_for,flash


import os
import shutil

import secrets

import sys

from Model_development.handwritten_to_digit.Inference import startPrediction
from Model_development.line_segmentation.line_segementation_model import start_line_seg
import time


app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(16)
app.config['UPLOAD_FOLDER'] = 'static/src_image'
app.config['TEXT_FOLDER'] = 'static/prediction_text'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


def convertJPG(file):
    try:
        image = Image.open(file)
        jpg_image: Image = image.convert('RGB')
        return jpg_image

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def duplicate_file(source_path, destination_path):
    shutil.copy2(source_path, destination_path)



# index page
@app.route("/", methods=['GET', 'POST'])
def index():
    try:
    # Example usage
    #source_file = None
    #destination_file = os.path.join(app.config["PREVIEW_FOLDER"], "sample.png")
        if request.method == 'POST':
            if 'file' not in request.files:
                return redirect(request.url)
            file = request.files['file']# request user input file
            if file.filename == "":
                return render_template('index.html')
            if file and ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() == 'png'):
                file = convertJPG(file)
            file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'], "sample.jpg"))

            #start_line_seg()
            #startPrediction()
            return redirect(url_for('loading'))
        return render_template('index.html')

    except Exception as e:
        print(f"Please submit a valid file: {e}")
        return render_template('index.html')


@app.route("/loading")
def loading():
    text_file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['TEXT_FOLDER'], "output.txt")
    if os.path.exists(text_file_path):
        with open(text_file_path,'r') as file:
            file_content = file.read()
        return render_template("secondpage.html", file_content=file_content)
    else:
        return "Text not found"


if __name__ == '__main__':
    app.run(debug=True)