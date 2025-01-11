import os
import cv2
import joblib
import numpy as np
import mediapipe as mp
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

class GestureTrainer:
    def __init__(self, gestures, model_path='movement_classifier/models/gesture_classifier.pkl', scaler_path='movement_classifier/models/scaler.pkl', data_path='movement_classifier/models/gesture_data.npy'):
        """
        Initializes the GestureTrainer system.
        
        Parameters:
        - gestures: List of gesture names corresponding to classes.
        - model_path: Path to save the trained classifier model.
        - scaler_path: Path to save the scaler.
        - data_path: Path to save the collected gesture data.
        """
        self.gestures = gestures
        self.num_classes = len(gestures)
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.data_path = data_path
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.data = []  # List to hold (features, label) tuples

    def collect_data(self, num_samples_per_gesture=100):
        """
        Collects gesture data using the camera.
        
        Parameters:
        - num_samples_per_gesture: Number of samples to collect for each gesture.
        """
        with self.mp_hands.Hands(static_image_mode=False,
                                 max_num_hands=2,
                                 min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5) as hands:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Error: Could not open webcam.")
                return
            
            for idx, gesture in enumerate(self.gestures):
                print(f"\nPrepare to record gesture '{gesture}'.")
                print("Press 's' to start recording, 'e' to end recording.")
                recording = False
                collected_samples = 0

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("Failed to grab frame.")
                        break

                    # Flip the image horizontally for a mirror view
                    frame = cv2.flip(frame, 1)
                    # Convert the BGR image to RGB
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Process the image and find hands
                    results = hands.process(image_rgb)
                    
                    # Draw hand landmarks on the image
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            self.mp_drawing.draw_landmarks(
                                frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    cv2.putText(frame, f"Gesture: {gesture}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    status = "Recording" if recording else "Press 's' to Start"
                    cv2.putText(frame, status, (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                    
                    cv2.imshow('Gesture Trainer', frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('s') and not recording:
                        print("Recording started...")
                        recording = True
                        collected_samples = 0
                        self.current_features = []
                    elif key == ord('e') and recording:
                        print("Recording stopped.")
                        recording = False
                        break

                    if recording and results.multi_hand_landmarks:
                        # Extract features from all detected hands
                        all_features = []
                        for hand_landmarks in results.multi_hand_landmarks:
                            features = []
                            for lm in hand_landmarks.landmark:
                                features.extend([lm.x, lm.y, lm.z])
                            all_features.extend(features)
                        
                        # Ensure the feature vector has the correct length
                        expected_length = 154  # 21 landmarks * x,y,z for 2 hands ######CHANGE THIS TO UPDATE THE NUMBER OF FEATURES 21*3*2 = gestures*14
                        if len(all_features) < expected_length:
                            # Pad with zeros if less than expected
                            all_features += [0.0] * (expected_length - len(all_features))
                        else:
                            all_features = all_features[:expected_length]
                        
                        self.data.append((all_features, idx))
                        collected_samples += 1
                        print(f"Collected {collected_samples}/{num_samples_per_gesture} samples for '{gesture}'", end='\r')
                        
                        if collected_samples >= num_samples_per_gesture:
                            print(f"\nCollected required samples for '{gesture}'.")
                            recording = False
                            break

            cap.release()
            cv2.destroyAllWindows()
            self.save_data()

    def save_data(self):
        """
        Saves the collected gesture data to a file.
        """
        if not self.data:
            print("No data to save.")
            return
        
        features = np.array([sample[0] for sample in self.data])
        labels = np.array([sample[1] for sample in self.data])

        np.save(self.data_path, {'features': features, 'labels': labels})
        print(f"Saved collected data to {self.data_path}.")

    def load_data(self):
        """
        Loads the gesture data from the saved file.
        """
        if not os.path.exists(self.data_path):
            print(f"Data file {self.data_path} not found.")
            return None, None
        
        data = np.load(self.data_path, allow_pickle=True).item()
        features = data['features']
        labels = data['labels']

        print(f"Loaded data from {self.data_path}: {features.shape[0]} samples.")
        return features, labels

    def train_model(self, X, y):
        """
        Trains the MLPClassifier with the provided data.
        
        Parameters:
        - X: Feature matrix.
        - y: Labels.
        """
        print("Starting model training...")
        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split the data
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Define the MLPClassifier
        model = MLPClassifier(hidden_layer_sizes=(10,), activation='relu',
                              solver='adam', max_iter=500, random_state=42)
        
        # Train the model
        model.fit(X_train, y_train)
        print("Model training completed.")
        
        # Evaluate the model
        y_pred = model.predict(X_val)
        print("Validation Classification Report:")
        print(classification_report(y_val, y_pred, target_names=self.gestures))
        
        # Save the model and scaler
        joblib.dump(model, self.model_path)
        joblib.dump(scaler, self.scaler_path)
        print(f"Model saved to {self.model_path} and scaler saved to {self.scaler_path}.")

    def train_from_saved_data(self):
        """
        Loads the saved data and trains the model.
        """
        X, y = self.load_data()

        if X is not None and y is not None:
            self.train_model(X, y)
        else:
            print("No data available for training.")

    def reset_data(self):
        """
        Clears the collected data and deletes the data file.
        """
        self.data = []
        if os.path.exists(self.data_path):
            os.remove(self.data_path)
            print(f"Deleted data file {self.data_path}.")
        else:
            print("No data file to delete.")

