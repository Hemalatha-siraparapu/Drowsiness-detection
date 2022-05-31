import cv2
import dlib 
from scipy.spatial import distance

def calculate_EAR(eye):
    a = distance.euclidean(eye[1], eye[5])
    b = distance.euclidean(eye[2], eye[4])
    c = distance.euclidean(eye[0], eye[3])
    ear_aspect_ration = (a+b)/(2.0*c)
    return ear_aspect_ration

cap = cv2.VideoCapture("vid2.mp4")
#cap = cv2.VideoCapture("video3.mp4")
#cap = cv2.VideoCapture("video.mp4")
output = cv2.VideoWriter("output.avi",cv2.VideoWriter_fourcc(*'MPEG'),30,(1080,1920))
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = hog_face_detector(gray)
    for face in faces:

        face_landmarks = dlib_facelandmark(gray, face)
        leftEye = []
        rightEye = []

        for n in range(36,42):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            leftEye.append((x,y))
            next_point = n+1
            if n == 41:
                next_point = 36
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            #cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)
        
        for n in range(42, 48):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            rightEye.append((x,y))
            next_point = n+1
            if n == 47:
                next_point = 42
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            #cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)

        left_ear = calculate_EAR(leftEye)
        right_ear = calculate_EAR(rightEye)

        EAR = (left_ear+right_ear)/2
        EAR = round(EAR,2)
        if EAR<0.24:
            cv2.putText(frame,"DROWSY",(20,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),4)
            cv2.putText(frame,"",(20,200),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            print("Drowsy")
        print(EAR)
    output.write(frame)
    cv2.imshow("output",frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

    cv2.imshow(" ",frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()



            
