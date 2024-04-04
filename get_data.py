import os
import cv2

data_path = './Data/Jesus Gordillo 213359/'
if not os.path.exists(data_path):
    os.makedirs(data_path)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

faceClassify = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
count = 0


while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (650, 500))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()

    faces = faceClassify.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0), 2)
        face = auxFrame[y:y+h, x:x+w]
        face = cv2.resize(face, (720,720), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(data_path + 'new2-{}.jpg'.format(count), face)
        count += 1

    cv2.imshow('frame', frame)

    k= cv2.waitKey(1)
    if k == 27 or count >=999:
        break


cap.release()
cv2.destroyAllWindows()

