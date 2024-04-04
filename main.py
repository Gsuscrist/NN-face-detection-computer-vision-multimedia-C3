from flask import Flask, render_template, Response
import cv2
import os

app = Flask(__name__)

data_path = './Data'
image_path = os.listdir(data_path)
print(image_path)

face_recognizer = cv2.face.LBPHFaceRecognizer.create()
face_recognizer.read('./model/my_face_recognizer_model.xml')

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
            result = face_recognizer.predict(face)

            # Si la cara detectada coincide con la tuya
            if result[1] < 30:
                cv2.putText(frame, "{}".format(image_path[result[0]]), (x, y-25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
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
