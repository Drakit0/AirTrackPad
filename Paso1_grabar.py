import cv2
import mediapipe as mp
import os
import csv

cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("No se abre la cámra")
    exit()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

output_folder = "gestures_data"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

csv_file_path = os.path.join(output_folder, "gestures_landmarks.csv")
is_file_new = not os.path.exists(csv_file_path)

csv_file = open(csv_file_path, mode='a', newline='')
csv_writer = csv.writer(csv_file)

if is_file_new:
    header = ["gesture"] + [f"landmark_x_{i+1}" for i in range(21)] + [f"landmark_y_{i+1}" for i in range(21)] + [f"landmark_z_{i+1}" for i in range(21)]
    csv_writer.writerow(header)

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    print("Realiza el gesto de 'left click'.")
    print("Cuando termines de hacer el gesto, presiona 'q' para cambiar al siguiente gesto.")
    
    current_gesture = "left_click"  
    
    while cam.isOpened():
        ret, frame = cam.read()
        if not ret:
            print("No se pudo obtener frame")
            continue

        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        results = hands.process(frame)
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmarks = [current_gesture]  
                for landmark in hand_landmarks.landmark:
                    landmarks.append(landmark.x)
                    landmarks.append(landmark.y)
                    landmarks.append(landmark.z)
                
                csv_writer.writerow(landmarks)

        cv2.imshow('Gesture Data Collection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Cambiar de gesto
            if current_gesture == "left_click":
                current_gesture = "right_click"
                print("Realiza el gesto de 'right click' y vuelve a darle a 'q'.")
            else:
                break  

csv_file.close()
cam.release()
cv2.destroyAllWindows()
