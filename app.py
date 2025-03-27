import cv2
import numpy as np
import pyautogui as gui
import time
import streamlit as st

# Set keypress delay to 0
gui.PAUSE = 0

# Streamlit page settings
st.title("Face Tracking Racing Game")
st.write("Use your face to control the car!")
run_game = st.button("Start Game")

# Loading the model
model_path = './res10_300x300_ssd_iter_140000.caffemodel'
prototxt_path = './deploy.prototxt'

def detect(net, frame):
    detected_faces = []
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            detected_faces.append({'start': (startX, startY), 'end': (endX, endY), 'confidence': confidence})
    return detected_faces

def move(detected_faces, bbox):
    for face in detected_faces:
        x1, y1 = face['start']
        x2, y2 = face['end']
        # Left movement
        if x1 < bbox[0]:
            gui.keyDown('left')
        else:
            gui.keyUp('left')

        # Right movement
        if x2 > bbox[1]:
            gui.keyDown('right')
        else:
            gui.keyUp('right')

def play_game():
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    cap = cv2.VideoCapture(0)
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))

    # Coordinates of the bounding box on the frame
    left_x, top_y = frame_width // 2 - 150, frame_height // 2 - 200
    right_x, bottom_y = frame_width // 2 + 150, frame_height // 2 + 200
    bbox = [left_x, right_x, bottom_y, top_y]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        detected_faces = detect(net, frame)
        
        # Draw bounding box around detected faces
        for face in detected_faces:
            cv2.rectangle(frame, face['start'], face['end'], (0, 255, 0), 3)
        
        # Move based on face position
        move(detected_faces, bbox)

        # Display the webcam feed
        st.image(frame, channels="BGR")
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if run_game:
    play_game()
