import tensorflow as tf
import cv2 as cv
import numpy as np
import mediapipe as mp
model = tf.keras.saving.load_model()
res=model.input.shape[1]



confidence = 0.9
recognition = 0.5
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(confidence)

cap = cv.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print('CAM NOT OPENED')
        break
    frame = cv.flip(frame, 1)
    image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    results = face_detector.process(image)
    if results.detections:
        for id, detection in enumerate(results.detections):
            bboxC = detection.location_data.relative_bounding_box
            h, w, c = image.shape
            x1, y1, width, height = int(bboxC.xmin*w), int(bboxC.ymin*h)-20, int(bboxC.width*w), int(bboxC.height*h)+20
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height
            face = image[y1:y2, x1:x2]