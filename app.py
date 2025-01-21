# Importing essential libraries and modules

from flask import Flask, render_template, request
import numpy as np
import pandas as pd

import requests
import config
import pickle
import io
from PIL import Image
#from reportlab.lib.pagesizes import letter
#from reportlab.pdfgen import canvas

# ==============================================================================================

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

#import joblib  # For saving and loading the model


# Load the saved model
#loaded_model = joblib.load("xgboost_model.pkl")
#print("Model loaded successfully.")


from model_predict  import enhance_model

# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------


app = Flask(__name__)

# render home page


@ app.route('/')
def home():
    title = 'Under Water Enhancement Using Deeplearning'
    return render_template('index.html', title=title)

# render crop recommendation form page

@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Under Water Enhancement Using Deeplearning'

    if request.method == 'POST':
        #if 'file' not in request.files:
         #   return redirect(request.url)

            file = request.files.get('file')

           # if not file:
            #    return render_template('disease.html', title=title)

            #img = Image.open(file)
            file.save('output.png')

            enhance_model('output.png')



            return render_template('rust-result.html', prediction="The enhanced Images are Displayed Below :  ",precaution="Done",title=title)
        #except:
         #   pass
    return render_template('rust.html', title=title)


# render disease prediction result page


# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=True)
