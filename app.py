from flask import Flask, request, render_template, jsonify
from flask_cors import CORS

import pickle
import numpy as np
import os
from model import preprocess_audio, predict_alzheimer

test_audio = "audio.wav"  # Replace with the uploaded audio file path
model_file = "alzheimer_model.h5"  # Replace with the trained model path
""" print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%first print%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
res1,res2,res3 ,res4= predict_alzheimer(test_audio, model_file)
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% result %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print(res1)
print(res2)
print(res3)
print(res4) """


app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"msg": "No file uploaded."})

    file = request.files['file']
    if file.filename == '':
        return jsonify({"msg": "No file selected."})
    if file:
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Preprocess the audio file and make prediction
        preprocessed_audio = preprocess_audio(file_path)
        res1, res2, res3, res4 = predict_alzheimer(
            preprocessed_audio, model_file)
        print(res4)
        # Return the prediction results
        return jsonify({
            "res3": res3,
            "res4": str(res4)
        })
        # return render_template('result.html', res1=res1, res2=res2, res3=res3, res4=res4)


if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run()
