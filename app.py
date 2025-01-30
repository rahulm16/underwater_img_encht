# Importing essential libraries and modules

from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from PIL import Image
from model_enhance import enhance_model

app = Flask(__name__)

# render home page
@app.route('/')
def home():
    title = 'Underwater Image Enhancement Using Deep Learning'
    return render_template('index.html', title=title)

# Handle underwater image enhancement requests
@app.route('/enhance-image', methods=['GET', 'POST'])
def enhance_image():
    title = 'Underwater Image Enhancement Using Deep Learning'

    if request.method == 'POST':
        # Process uploaded underwater image
        file = request.files.get('file')
        file.save('static/upload&result/input.png')
        enhance_model('static/upload&result/input.png')
        return render_template('result.html', prediction="The enhanced Images are Displayed Below:", precaution="Enhancement Complete", title=title)
    return render_template('enhance.html', title=title)

if __name__ == '__main__':
    app.run(debug=True)
