import streamlit as st
import cv2
import numpy as np
import pyautogui as gui
import time

# Set keypress delay to 0
gui.PAUSE = 0

# Load the pre-trained face model
model_path = './res10_300x300_ssd_iter_140000.caffemodel'
prototxt_path = './deploy.prototxt'

st.title("Web Racing Game with Face Tracking")
st.write("Move your head left or right to steer the car!")

def detect(net, frame):
    detected_faces = []
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            detected_faces.append({'start': (startX, startY), 'end': (endX, endY), 'confidence': confidence})
    return detected_faces

def drawFace(frame, detected_faces):
    for face in detected_faces:
        cv2.rectangle(frame, face['start'], face['end'], (0, 255, 0), 10)
    return frame

def move(detected_faces, bbox):
    for face in detected_faces:
        x1, y1 = face['start']
        x2, y2 = face['end']

        # Left Movement
        if x1 < bbox[0]:
            gui.keyDown('left')
            st.write("Moving Left")
        else:
            gui.keyUp('left')

        # Right Movement
        if x2 > bbox[1]:
            gui.keyDown('right')
            st.write("Moving Right")
        else:
            gui.keyUp('right')

def play(prototxt_path, model_path):
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    cap = cv2.VideoCapture(0)

    # Bounding box coordinates
    frame_width, frame_height = 640, 480
    left_x, top_y = frame_width // 2 - 150, frame_height // 2 - 200
    right_x, bottom_y = frame_width // 2 + 150, frame_height // 2 + 200
    bbox = [left_x, right_x, bottom_y, top_y]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        detected_faces = detect(net, frame)
        frame = drawFace(frame, detected_faces)
        move(detected_faces, bbox)
        cv2.imshow('Camera Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if st.button("Start Game"):
    play(prototxt_path, model_path)
