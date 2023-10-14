from PIL.Image import Image
from flask import Flask, render_template, request, redirect, url_for

from werkzeug.utils import secure_filename
import os

import secrets

app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(16)
app.config['UPLOAD_FOLDER'] = 'static/files'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


def convertJPG(file):
    try:
        image = Image.open(file)
        jpg_image: Image = image.convert('RGB')
        return jpg_image

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# index page
@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']  # request user input file
        if file and ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() == 'png'):
            file = convertJPG(file)

        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'], "sample.jpg"))
        return redirect(url_for('loading'))
    return render_template("index.html")


@app.route("/loading")
def loading():
    return render_template("secondpage.html")


if __name__ == '__main__':
    app.run(debug=True)