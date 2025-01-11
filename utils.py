import numpy as np
import mediapipe as mp

def roi_extractor(landmarks, frame, padding = 10 ) -> tuple[np.ndarray, int, int, int, int]:
    frame_h, frame_w = frame.shape[:2]
    
    x_coords = [int(lm.x * frame_w) for lm in landmarks.landmark] # Normalized coords to pixel coords
    y_coords = [int(lm.y * frame_h) for lm in landmarks.landmark]
    
    x_min = min(x_coords) # ROI around landmarks
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)
    
    x_min = max(0, x_min - padding) # ROI with padding
    y_min = max(0, y_min - padding)
    x_max = min(frame_w, x_max + padding)
    y_max = min(frame_h, y_max + padding)
    
    return frame[y_min:y_max, x_min:x_max], x_min, y_min, x_max, y_max # ROI, x_min, y_min, x_max, y_max

def draw_landmarks(frame, landmarks, mp_drawing, mp_drawing_styles):
    for hand_landmarks in landmarks.multi_hand_landmarks:
        
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp.solutions.hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )
