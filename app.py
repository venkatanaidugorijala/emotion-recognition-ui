from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelEncoder
import os

app = Flask(__name__)

# Load models and resources
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
emotion_classifier = load_model('model.h5')
face_recognition_model = load_model('face_recognition_model.h5')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
label_encoder = LabelEncoder()
label_encoder.fit(np.unique(os.listdir('recognition dataset')))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'frame' not in request.files:
        return jsonify({"error": "No frame provided"}), 400

    file = request.files['frame']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    result = {
        "face_detected": False,
        "emotion": None,
        "confidence": None,
        "name": None
    }

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            emotion_prediction = emotion_classifier.predict(roi)[0]
            emotion_label = emotion_labels[np.argmax(emotion_prediction)]
            confidence = float(np.max(emotion_prediction))

            face_resized = cv2.resize(roi_gray, (100, 100))
            face_resized = np.expand_dims(face_resized, axis=-1)
            face_resized = np.expand_dims(face_resized, axis=0)

            recognition_prediction = face_recognition_model.predict(face_resized)
            recognition_label = label_encoder.inverse_transform([np.argmax(recognition_prediction)])

            result["face_detected"] = True
            result["emotion"] = emotion_label
            result["confidence"] = round(confidence, 3)
            result["name"] = recognition_label[0]
            break

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
