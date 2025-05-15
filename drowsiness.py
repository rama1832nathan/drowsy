from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils
import dlib
import cv2
from openpyxl import Workbook
from datetime import datetime
import smtplib, ssl



smtp_server = "smtp.gmail.com"
smtp_port = 587
sender_email = "ramank1832@gmail.com"
receiver_email = "sricharith1810@gmail.com"
password = "vzcq chfl ubgg ajib"

def send_email_alert():
    subject = "Drowsiness Detected!"
    body = "Eyes have been closed for more than 3 seconds. Drowsiness detected."
    message = f"Subject: {subject}\n\n{body}"
    context = ssl.create_default_context()
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls(context=context)
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)



mixer.init()
mixer.music.load("alert.wav")

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

thresh = 0.25
frame_check = 20
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shapes.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

cap = cv2.VideoCapture(0)
flag = 0
eyes_closed_time = 0
eyes_opened_time = 0
eyes_closed_duration = 0

# For Excel workbook
wb = Workbook()
ws = wb.active
ws.append(["Start Time", "End Time", "Duration (s)"])

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)
    
    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        
        if ear < thresh:
            flag += 1
            if flag >= frame_check:
                if eyes_closed_time == 0:
                    eyes_closed_time = datetime.now()
                    eyes_opened_time = None
                    eyes_closed_duration = 0
                    mixer.music.play()
                    print("Eyes Closed Count:", flag)
        else:
            if eyes_closed_time:
                eyes_opened_time = datetime.now()
                eyes_closed_duration = (eyes_opened_time - eyes_closed_time).total_seconds()
                if eyes_closed_duration>3:
                    send_email_alert()

                    
                ws.append([eyes_closed_time, eyes_opened_time, eyes_closed_duration])
                wb.save("drowsiness_data.xlsx")
                eyes_closed_time = 0
                flag = 0
        
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()
