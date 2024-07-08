import cv2

# Face classifier
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')
eye_detector = cv2.CascadeClassifier('haarcascade_eye.xml')



#Get the webcam
webcam = cv2.VideoCapture(0)

while True:

   # reads the current frame from the webcam video
    successful_frame_read, frame = webcam.read()

    #check's for an error
    if not successful_frame_read:
        break

    # converts the frame to grayscale
    frame_grayscaled = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detects the faces in the frame
    faces = face_detector.detectMultiScale(frame_grayscaled)

    # Run the face detection with each the faces
    for (x, y, w, h) in faces:

        # Draw a rectangle around the faces
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 200, 50), 4)

        # Crop the face from the frame /Get the sub frame(using numpy N-dimentional array slicing)
        the_face = frame[y:y+h , x:x+w,]

         # converts the frame to grayscale
        face_grayscaled = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)


        smiles = smile_detector.detectMultiScale(face_grayscaled, scaleFactor= 1.7, minNeighbors=20)

        eyes = eye_detector.detectMultiScale(face_grayscaled, scaleFactor= 1.1, minNeighbors=30)

         # Find all smiles in the face
        #for (x_, y_, w_, h_) in smiles:

            # Draw a rectangle around the faces

            #cv2.rectangle(the_face, (x_, y_), (x_ + w_, y_ + h_), (50, 50, 200), 4)

         # Find all smiles in the face
        for (x_, y_, w_, h_) in eyes:

            # Draw a rectangle around the faces

            cv2.rectangle(the_face, (x_, y_), (x_ + w_, y_ + h_), (225, 225, 225), 4)
    



        # Find all smiles in the face
        #for (x_, y_, w_, h_) in smiles:

            # Draw a rectangle around the faces

            #cv2.rectangle(the_face, (x_, y_), (x_ + w_, y_ + h_), (50, 50, 200), 4)


            # Lable the face as smiling
        if len(smiles) > 0:
            cv2.putText(frame, 'Smiling', (x, y+h+40), fontScale=3, 
            fontFace=cv2.FONT_HERSHEY_PLAIN, color = (255, 255, 255))    

        if len(eyes) > 0:
            cv2.putText(frame, 'Eye detected', (x, y+h+90), fontScale=3, 
            fontFace=cv2.FONT_HERSHEY_PLAIN, color = (255, 255, 255))    
    




    # displays the current frame
    cv2.imshow('Smile Detector', frame)

    # Display
    cv2.waitKey(1)

# Cleanup
webcam.release()
cv2.destroyAllWindows()    

print("What's up")