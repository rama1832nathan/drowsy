from flask import Flask, request, jsonify
import base64
import cv2
import numpy as np
from scipy.spatial import distance
from imutils import face_utils
import dlib
from pygame import mixer
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 


# Initialize Pygame mixer and load the alert sound
mixer.init()
mixer.music.load("alert.wav")

# Drowsiness detection logic (your existing code)
thresh = 0.25
frame_check = 20
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shapes.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

flag = 0
eyes_closed_time = 0
eyes_opened_time = 0
eyes_closed_duration = 0

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

@app.route('/detect', methods=['POST'])
def detect_drowsiness():
    global flag, eyes_closed_time, eyes_opened_time, eyes_closed_duration

    # Get the image from the frontend
    data = request.get_json()
    img_data = data['image']
    img_data = img_data.split(',')[1]  # Remove 'data:image/jpeg;base64,' part
    img_bytes = base64.b64decode(img_data)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Convert to grayscale
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

        if ear < thresh:
            flag += 1
            if flag >= frame_check:
                if eyes_closed_time == 0:
                    eyes_closed_time = datetime.now()
                    eyes_opened_time = None
                    eyes_closed_duration = 0
                    mixer.music.play()  # Alert sound
        else:
            if eyes_closed_time:
                eyes_opened_time = datetime.now()
                eyes_closed_duration = (eyes_opened_time - eyes_closed_time).total_seconds()
                eyes_closed_time = 0
                flag = 0

    # Return drowsiness status as JSON
    if eyes_closed_duration > 3:
        return jsonify({"status": "Drowsiness Detected!"})
    else:
        return jsonify({"status": "No Drowsiness Detected"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
