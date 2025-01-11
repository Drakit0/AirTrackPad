import cv2
import time
from utils import draw_landmarks
from hand_tracking.detector import HandDetector
from movement_follower.completor import landmark_completor

if __name__ == "__main__":
    
    detector = HandDetector()
    
    cam = cv2.VideoCapture(0)
    
    prev_frame = None
    prev_landmarks = None
    
    while cam.isOpened():
        
        ret, frame = cam.read()
        
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        detector.mediapipe_detection(frame, draw=True)
        detector.sobel_detection(frame)
        detector.check_landmarks()
        accuracy = detector.model_accuracy
        
        if accuracy < 0.80 or (cv2.waitKey(1) & 0xFF == ord('l')): # Try better adjust with optical flow
            t0 = time.time()
            multi_hand_landmarks = landmark_completor(prev_frame, prev_landmarks, frame)
            detector.landmarks.multi_hand_landmarks = multi_hand_landmarks
            draw_landmarks(frame, detector.landmarks.multi_hand_landmarks, detector.mp_drawing, detector.mp_drawing_styles)
            
            # print("Time taken: ", time.time() - t0)
            
            
            
        prev_frame = frame.copy()
        prev_landmarks = detector.landmarks
        
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break