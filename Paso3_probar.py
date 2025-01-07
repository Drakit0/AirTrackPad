import cv2
import mediapipe as mp
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


model = joblib.load('gesture_nn_model.pkl') # Para redes neuronales
model = joblib.load('gesture_logistic_model.pkl') # Para la regresión logística
scaler = joblib.load('gesture_scaler.pkl')
label_encoder = joblib.load('gesture_label_encoder.pkl')

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("No se pudo abrir la cámara")
    exit()

landmark_columns = [f"landmark_x_{i+1}" for i in range(21)] + \
                    [f"landmark_y_{i+1}" for i in range(21)] + \
                    [f"landmark_z_{i+1}" for i in range(21)]

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cam.isOpened():
        ret, frame = cam.read()
        if not ret:
            print("No se pudo obtener frame")
            break

        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        results = hands.process(frame)
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmarks = [landmark.x for landmark in hand_landmarks.landmark] + \
                            [landmark.y for landmark in hand_landmarks.landmark] + \
                            [landmark.z for landmark in hand_landmarks.landmark]

                print("Landmarks sin normalizar:", landmarks)

                landmarks_df = pd.DataFrame([landmarks], columns=landmark_columns)

                try:
                    landmarks_scaled = scaler.transform(landmarks_df)
                except Exception as e:
                    print(f"Error al normalizar los landmarks: {e}")
                    continue

                print("Landmarks normalizados:", landmarks_scaled)

                prediction = model.predict(landmarks_scaled)

                print("Predicción numérica:", prediction)

                gesture = label_encoder.inverse_transform(prediction)
                print("Gesto detectado:", gesture[0])

                cv2.putText(frame, f"Gesto: {gesture[0]}", (10, frame.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('Gesture Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cam.release()
cv2.destroyAllWindows()


