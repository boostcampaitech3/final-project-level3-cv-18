import cv2
from PIL import Image
import os


def crop(path):
    # Read the input image
    img = path
    
    
    global faces

    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load the cascade
    face_cascade = cv2.CascadeClassifier('/opt/ml/opencv/data/haarcascades/haarcascade_frontalface_alt2.xml')

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4,minSize=(110,110))
    
    if len(faces) == 0:
        raise Exception()

    # Draw rectangle around the faces and crop the faces
    for (x, y, w, h) in faces:
        #rectangle = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), 0)
        faces = img[y:int(y+h*1.1),x:x+w]
    
    
    return faces
