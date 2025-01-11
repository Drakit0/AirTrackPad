import cv2
import numpy as np
import mediapipe as mp
from utils import roi_extractor

def landmark_completor(prev_frame, prev_landmarks, current_frame, device="computer"):

    if device != "computer": # Resize edges for easier processing
        winSize = (5, 5)
    else:
        winSize = (15, 15)
        
    maxLevel = 2
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)

    for hand_landmarks in prev_landmarks.multi_hand_landmarks:
        
        prev_roi, px_min, py_min, _, _ = roi_extractor(hand_landmarks, prev_frame)
        curr_roi, cx_min, cy_min, _, _ = roi_extractor(hand_landmarks, current_frame)

        if prev_roi.size == 0 or curr_roi.size == 0: # Skip if no ROI
            continue

        adjusted_prev_roi = cv2.resize(prev_roi, (curr_roi.shape[1], curr_roi.shape[0])) # Resize prev_roi to curr_roi size
        adjusted_prev_gray = cv2.cvtColor(adjusted_prev_roi, cv2.COLOR_BGR2GRAY)
        adjusted_curr_gray = cv2.cvtColor(curr_roi, cv2.COLOR_BGR2GRAY)

        prev_points = []
        
        for lm in hand_landmarks.landmark: # Landmarks to ROI coords
            px = int(lm.x * prev_frame.shape[1])  
            py = int(lm.y * prev_frame.shape[0])  
            prev_points.append([px - px_min, py - py_min])
            
        prev_points = np.array(prev_points, dtype=np.float32).reshape(-1, 1, 2)

        # Lukas-Kanade optical flow
        new_points, status, _ = cv2.calcOpticalFlowPyrLK(adjusted_prev_gray, adjusted_curr_gray, prev_points, None, winSize=winSize, maxLevel=maxLevel, criteria=criteria)

        if new_points is not None:
 
            for i, new_pt in enumerate(new_points): # Update landmarks with  correct optical flow
                if status[i][0]:
                    new_x = int(new_pt[0][0] + cx_min)
                    new_y = int(new_pt[0][1] + cy_min)
                    
                else: # maintain old coords
                    new_x = int(prev_points[i][0][0] + cx_min)
                    new_y = int(prev_points[i][0][1] + cy_min)

                # Normalize coords
                hand_landmarks.landmark[i].x = new_x / float(current_frame.shape[1])
                hand_landmarks.landmark[i].y = new_y / float(current_frame.shape[0])
                hand_landmarks.landmark[i].z = 0.0 # No info
                
        else:

            for i, lm in enumerate(hand_landmarks.landmark): # Total fail
                old_x = int(lm.x * prev_frame.shape[1])
                old_y = int(lm.y * prev_frame.shape[0])
                
                hand_landmarks.landmark[i].x = old_x / float(current_frame.shape[1])
                hand_landmarks.landmark[i].y = old_y / float(current_frame.shape[0])
                hand_landmarks.landmark[i].z = 0.0 # No info

    return prev_landmarks # Return SAME landmarks as it wasn't possible to create new ones