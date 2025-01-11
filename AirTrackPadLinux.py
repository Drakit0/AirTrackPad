import cv2
import time
import numpy as np
from Utils import draw_landmarks
from picamera2 import Picamera2, Preview
from hand_tracking.HandsDetector import HandDetector
from movement_follower.FPSComplete import landmark_completor
from movement_classifier.Classifier import NeuralClassifier
from actions_handler.ActionsManager import ActionManager


if __name__ == "__main__":
    
    gesture_actions = {
            0:"Left Click",
            1:"Right Click",
            2:"Double Click",
            3:"Zoom In",
            4:"Zoom Out",
            5:"Scroll Up",
            6:"Scroll Down",
            7:"Scroll Left",
            8:"Scroll Right",
            9:"Pointing",
            10:"No Gesture"}
    
    model_path = 'movement_classifier/models/gesture_classifier.pkl'
    scaler_path = 'movement_classifier/models/scaler.pkl'
    num_features = 154
    num_classes = 11
    classifier = NeuralClassifier(model_path, scaler_path, num_features, num_classes) # Movement classifier
    
    detector = HandDetector() # Hand detector
    
    frame_width = 640
    frame_height = 480
    manager = ActionManager(frame_width, frame_height) # Action manager
    
    prev_frame = None # For optical flow
    prev_landmarks = None
    
    gesture = 10
    
    # cam = cv2.VideoCapture(0)
    
    picam2 = Picamera2()
    # picam2.preview_configuration.main.size = (4608, 2592)
    picam2.preview_configuration.main.format = "RGB888"
    picam2.preview_configuration.align()
    # picam2.configure("preview")
    picam2.start()
    
    while cam.isOpened():
        
        ret, frame = cam.read()

        if not ret:
            break
        
        frame = cv2.flip(frame, 1) # Mirror view
        
        detector.mediapipe_detection(frame, draw = True)
        detector.sobel_detection(frame)
        detector.check_landmarks()
        accuracy = detector.model_accuracy
        
        if accuracy < 0.80 or (cv2.waitKey(1) & 0xFF == ord('l')): # Try better adjust with optical flow
            t0 = time.time()
            multi_hand_landmarks = landmark_completor(prev_frame, prev_landmarks, frame)
            detector.landmarks.multi_hand_landmarks = multi_hand_landmarks
            draw_landmarks(frame, detector.landmarks.multi_hand_landmarks, detector.mp_drawing, detector.mp_drawing_styles)
                        
        if detector.landmarks.multi_hand_landmarks:
            
            all_features = [] # Extract features from all detected hands
            
            for hand_landmarks in detector.landmarks.multi_hand_landmarks:
                features = []
                for lm in hand_landmarks.landmark:
                    features.extend([lm.x, lm.y, lm.z])
                all_features.extend(features)
                
            if len(all_features) < num_features: # Vecor must be as long as the number of features
                all_features += [0.0] * (num_features - len(all_features))
                
            else:
                all_features = all_features[:num_features]
                
            features = np.array(all_features).reshape(1, -1)
            features = classifier.scaler.transform(features) # Normalize features
            
            predicted_class, confidence = classifier.classify_gesture(features) # Classify gesture
            gesture = predicted_class   
            
            cv2.putText(frame, f"{gesture_actions[gesture]}, {confidence}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            if gesture == 9: # Pointing
                for hand_landmarks in detector.landmarks.multi_hand_landmarks:
                    manager.handle_gesture(gesture, hand_landmarks.landmark)
                        
            else: # Other gestures
                manager.handle_gesture(gesture)
        
        prev_frame = frame.copy()
        prev_landmarks = detector.landmarks
                    
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break