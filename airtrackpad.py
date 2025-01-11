from hand_tracking.detector import HandDetector
import cv2

if __name__ == "__main__":
    
    detector = HandDetector()
    
    cam = cv2.VideoCapture(0)
    
    while cam.isOpened():
        
        ret, frame = cam.read()
        
        frame = cv2.flip(frame, 1)
        
        if not ret:
            break
        
        detector.mediapipe_detection(frame)
        detector.sobel_detection(frame)
        detector.check_landmarks()
        accuracy = detector.model_accuracy
        # Display accuracy on the frame
        cv2.putText(frame, f'Accuracy: {accuracy:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break