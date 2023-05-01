import cv2
import dlib
import numpy as np

# Initialize dlib's face detector and facial landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Initialize video capture from default camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to grayscale for faster processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = detector(gray, 0)

    # Draw a rectangle around each detected face and detect smile
    for face in faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Detect landmarks in the face
        landmarks = predictor(gray, face)

        # Get coordinates of upper and lower lips
        upper_lip = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(50, 53)])
        lower_lip = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(58, 61)])

        # Compute the mean y-coordinate of the upper and lower lips
        y_mean_upper = np.mean(upper_lip[:, 1])
        y_mean_lower = np.mean(lower_lip[:, 1])

        # Compute the distance between the upper and lower lips
        lip_dist = int(y_mean_lower - y_mean_upper)

        # Determine if the person is smiling or not
        if lip_dist > 10:
            cv2.putText(frame, "Not smiling", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Smiling", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()

