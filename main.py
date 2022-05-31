import cv2 as cv
face_cascade = cv.CascadeClassifier('haarcascade_fullbody.xml')    
cap =  cv.VideoCapture(0)

while True:
    _, img = cap.read() 
    #print("blabla "+ str(img))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.1,4)
    for(x,y,w,h) in faces:
        cv.rectangle(img,(x,y), (x+w, y+h), (255,0,0),2)
    cv.imshow('img',img)
    k = cv.waitKey(30)
    if k ==27:
        break
cap.release()

