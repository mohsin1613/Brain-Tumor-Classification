from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import imutils


app = Flask(__name__)

model = load_model('EfficientNet-with.h5')
image_size = 150 

class_labels = ['glioma_tumor', 'meningioma_tumor',
                'no_tumor', 'pituitary_tumor']
class_serials = {class_label: serial for serial,
                 class_label in enumerate(class_labels)}



def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    thresh = cv2.threshold(image, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]
    new_image = cv2.resize(new_image, (image_size, image_size))
    new_image = np.expand_dims(new_image, axis=-1)
    new_image = new_image / 255.0  # Normalize to [0, 1]
    return new_image



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (image_size, image_size))

    prediction = model.predict(np.expand_dims(img, axis=0))
    class_idx = np.argmax(prediction)
    predicted_class = next(class_label for class_label,
                           serial in class_serials.items() if serial == class_idx)

    if predicted_class == "no_tumor":
        predicted_class = "No Tumor"
    elif predicted_class == "pituitary_tumor":
        predicted_class = "Pituitary Tumor"
    elif predicted_class == "glioma_tumor":
        predicted_class = "Glioma Tumor"
    elif predicted_class == "meningioma_tumor":
        predicted_class = "Meningioma Tumor"
    else:
        predicted_class="Wrong Image"

    # start from here
    predicted_image_path = 'static/predicted_image.jpg'
    cv2.imwrite(predicted_image_path, img)
    return render_template('index.html', predicted_class=predicted_class, predicted_image=predicted_image_path)
    # return jsonify({'class_label': predicted_class})


if __name__ == '__main__':
    app.run(debug=True)
