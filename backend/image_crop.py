import cv2


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
        faces = img[int(y*0.8):int(y+h*1.1),x:int(x+w*0.95)]
        forhead = img[int(y*0.9):int(y+h*0.3),x:int(x+w*0.95)]
        nose = img[int(y+h*0.4):int(y+h*0.75),x:int(x+w*0.95)]
        chin = img[int(y+h*0.65):int(y+h*1.1),x:int(x+w*0.95)]
        
    return faces
