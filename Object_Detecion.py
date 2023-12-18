import cv2

# image
img_file= "car.jpg"
video= cv2.VideoCapture("Dutch motorcycle driving Compilation 2.mp4")

# pre-trained car detection
classifierFile = "cars.xml"
car_detector = cv2.CascadeClassifier(classifierFile)

#create opencv image
img =cv2.imread(img_file)

grayScaled_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)




while True:
    successful_frame_read, frame= video.read()
    
    if successful_frame_read:
        grayScaled_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    else:
        break
    cars= car_detector.detectMultiScale(grayScaled_img)
    for (x,y,w,h) in cars:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("car detector", frame)
    
    key=cv2.waitKey(1) 
    if key ==81 or key ==113:
        break