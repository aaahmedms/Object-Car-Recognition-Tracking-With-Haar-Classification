import cv2

#Create a cascade classifer object
car_cascade = cv2.CascadeClassifier("cascadeclassifiers\\cars.xml")


#start capturing video feed
video_footage = cv2.VideoCapture("SampleVideos\\samplevid.mp4")


while(video_footage.isOpened()):
  ret, frame = video_footage.read()
  if ret:
      #search the coordinates of the image to find the cars
      car_capture = car_cascade.detectMultiScale(
                    frame, 
                    1.05, 
                    3, 
                    0 | cv2.CASCADE_SCALE_IMAGE, 
                    (50, 50)
            )
      
      #create tracking square on image
      for(x,y,w,h) in car_capture:
          outlined_image = cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),3)
          cv2.putText(outlined_image, 'Vehicle', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
      
      cv2.namedWindow('Car Detection Haar Classifier',cv2.WINDOW_NORMAL)
      cv2.resizeWindow('Car Detection Haar Classifier', 600,400)
      cv2.imshow('Car Detection Haar Classifier',frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  else:
      break

video_footage.release()
cv2.destroyAllWindows()
 
