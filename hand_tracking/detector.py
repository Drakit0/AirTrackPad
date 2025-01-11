import cv2
import numpy as np
import mediapipe as mp

class HandDetector:
    
    def __init__(self, device="computer", min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.device = device
        self.model_accuracy = 1
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = mp.solutions.hands.Hands( # MediaPipe Hands model
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        self.landmarks = None # Mediapipe landmarks
        self.sobel_edges = [] # Sobel edges
        self.accuracy = 0.0 # Model accuracy
    
    def mediapipe_detection(self, frame, draw=False):

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.landmarks = self.hands.process(rgb_frame)

        if self.landmarks.multi_hand_landmarks and draw: # Draw Landmarks on image, sometimes affects sobel (hence model accuracy)
            for hand_landmarks in self.landmarks.multi_hand_landmarks:
                
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )

    def sobel_detection(self, frame):

        self.sobel_edges = []

        if not self.landmarks or not self.landmarks.multi_hand_landmarks: # Only check sobel if hands are detected
            return

        frame_h, frame_w = frame.shape[:2]

        for hand_landmarks in self.landmarks.multi_hand_landmarks:
            x_coords = [int(lm.x * frame_w) for lm in hand_landmarks.landmark] # Normalized coords to pixel coords
            y_coords = [int(lm.y * frame_h) for lm in hand_landmarks.landmark]
            
            x_min = min(x_coords) # ROI around landmarks
            x_max = max(x_coords)
            y_min = min(y_coords)
            y_max = max(y_coords)
            
            padding = 10 # ROI with padding
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(frame_w, x_max + padding)
            y_max = min(frame_h, y_max + padding)
            roi = frame[y_min:y_max, x_min:x_max]

            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            sobel_x = cv2.Sobel(gray_roi, cv2.CV_64F, 1, 0, ksize=5) # Sobel
            sobel_y = cv2.Sobel(gray_roi, cv2.CV_64F, 0, 1, ksize=5)
            magnitude = cv2.magnitude(sobel_x, sobel_y)

            edges = np.uint8(magnitude > 50) * 255 # 50 good lighting, 100 low lighting, pherhaps automatize change if accuracy lowers too much

            cv2.imshow("Edges", edges) # Hands ROI edges

            self.sobel_edges.append((edges, (x_min, y_min, x_max, y_max)))


    def check_landmarks(self):

        if not self.landmarks or not self.landmarks.multi_hand_landmarks:
            self.accuracy = 0.0
            return

        for landmarks, sobel_data in zip(self.landmarks.multi_hand_landmarks, self.sobel_edges):
            edges, (x_min, y_min, x_max, y_max) = sobel_data

            if self.device != "computer": # Resize edges for better performance
                edges = cv2.resize(edges, (0,0), fx=0.05, fy=0.05)

            roi_h, roi_w = edges.shape[:2]
            original_roi_width = x_max - x_min
            original_roi_height = y_max - y_min

            if original_roi_width > 0 and original_roi_height > 0: # Reescalate ROI acording to the reescalation of edges
                scale_x = roi_w / float(original_roi_width)
                scale_y = roi_h / float(original_roi_height)
                
            else:
                scale_x = 1.0
                scale_y = 1.0

            matched = 0
            total_landmarks = len(landmarks.landmark)
            
            
            for landmark in landmarks.landmark: # For each landmark, find pos in ROI

                local_x = (landmark.x * roi_w) - x_min # absolute coords to relative ROI coords
                local_y = (landmark.y * roi_h) - y_min

                x_final = int(local_x * scale_x) # Scale back to original size
                y_final = int(local_y * scale_y)


                neighborhood = 3 # How close to a landmark to check
                x_start = max(0, x_final - neighborhood)
                x_end   = min(roi_w, x_final + neighborhood)
                y_start = max(0, y_final - neighborhood)
                y_end   = min(roi_h, y_final + neighborhood)

                region = edges[y_start:y_end, x_start:x_end] 
                
                if np.any(region): # Check for pixels on edges
                    matched += 1

            self.accuracy = matched / total_landmarks
            print(f"Accuracy: {self.accuracy:.2f}") # Accuracy of the model


            