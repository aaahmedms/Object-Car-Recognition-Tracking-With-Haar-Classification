import cv2

#Create a cascade classifer object
car_cascade = cv2.CascadeClassifier("cascadeclassifiers\\cars.xml")


#start capturing video feed
video_footage = cv2.VideoCapture("SampleVideos\\samplevid.mp4")

# ============================================================================

# This portion of the code was created by Dan Mašek from stackoverflow to provide rounded corners for the tracking box
# Source: https://stackoverflow.com/questions/46036477/drawing-fancy-rectangle-around-face/46037335

def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1,y1 = pt1
    x2,y2 = pt2

    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

# ============================================================================

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
          outlined_image = draw_border(frame, (x,y),(x+w,y+h),(0,255,0), 2, 5, 5)
          cv2.putText(outlined_image, 'Vehicle', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
      
      cv2.namedWindow('Car Detection Haar Classifier Edged Boxes',cv2.WINDOW_NORMAL)
      cv2.resizeWindow('Car Detection Haar Classifier Edged Boxes', 600,400)
      cv2.imshow('Car Detection Haar Classifier Edged Boxes',frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  else:
      break

video_footage.release()
cv2.destroyAllWindows()
 
