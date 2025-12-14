import os
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from werkzeug.utils import secure_filename
from flask import Response


app = Flask(__name__)

# Paths
MODEL_PATH = 'models/trained_model.h5'
UPLOAD_FOLDER = 'static/uploads'

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model once
model = load_model(MODEL_PATH)

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route('/style.css')
def style():
    return Response(
        render_template('style.css'),
        mimetype='text/css'
    )


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']

        if file and file.filename:
            filename = secure_filename(file.filename)
            img_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(img_path)

            img_array = preprocess_image(img_path)
            prediction = model.predict(img_array)

            if prediction > 0.015:
                result = 'Non Autistic'
            else:
                result = 'Autistic'

            return render_template(
                'index.html',
                img_path=img_path,
                filename=filename,
                result=result
            )

    return render_template('index.html', img_path=None, result=None)

if __name__ == '__main__':
    app.run(debug=True)
