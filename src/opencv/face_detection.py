import numpy as np
import cv2

# face_cascade = cv2.CascadeClassifier(r'./haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(r'./haarcascade_eye.xml')
image_path = '../../resources/one_baby_face.jpeg'
image_path = '../../resources/target.jpg'
image = cv2.imread(image_path)
# 探测图片中的人脸
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.rectangle(image,(x,y),(x+w,y+w),(0,255,0),2)

faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.15,
    minNeighbors=5,
    minSize=(5, 5),
     flags=cv2.CV_HAAR_SCALE_IMAGE
)

print("发现{0}个人脸!".format(len(faces)))
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + w), (0, 255, 0), 2)
    # cv2.circle(image,((x+x+w)/2,(y+y+h)/2),w/2,(0,255,0),2)

cv2.imshow("one_baby_face", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
