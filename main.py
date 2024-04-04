from flask import Flask, render_template, Response
import cv2
import os
import requests

app = Flask(__name__)

# URL del archivo en Google Drive
google_drive_url = "https://drive.google.com/uc?id=1v9WQiNNR2be91pccZvRxT8CSILsDG-HI"

# Descargar el archivo desde Google Drive
response = requests.get(google_drive_url)
with open('./model/my_face_recognizer_model.xml', 'wb') as f:
    f.write(response.content)

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def generate_frames():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (800, 620))
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        aux_frame = gray.copy()

        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = aux_frame[y:y+h, x:x+w]
            face = cv2.resize(face, (720, 720), interpolation=cv2.INTER_CUBIC)

            # Aquí cargarías el modelo de reconocimiento facial y harías la predicción

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run()
