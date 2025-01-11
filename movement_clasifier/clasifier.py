import cv2
import mediapipe as mp
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

class AirTrackPad:
    def __init__(self, model_path='gesture_classifier.pkl', scaler_path='scaler.pkl', num_features=84, num_classes=9):
        """
        Initializes the AirTrackPad system.
        
        Parameters:
        - model_path: Path to the saved classifier model.
        - scaler_path: Path to the saved scaler.
        - num_features: Number of input features (default is 84 for 42 landmarks * 2 hands * x,y).
        - num_classes: Number of gesture classes.
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.num_features = num_features
        self.num_classes = num_classes
        self.model = None
        self.scaler = None
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self._initialize_model()
    
    def _initialize_model(self):
        """
        Loads the scaler and model if they exist. Otherwise, initializes them.
        """
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            print("Loaded existing model and scaler.")
        else:
            print("Model or scaler not found. Please train the model first.")
    
    def train_model(self, X, y, test_size=0.2, random_state=42):
        """
        Trains the MLPClassifier with the provided data.
        
        Parameters:
        - X: Feature matrix.
        - y: Labels.
        - test_size: Proportion of the dataset to include in the test split.
        - random_state: Controls the shuffling applied to the data before applying the split.
        """
        print("Starting model training...")
        # Normalize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Split the data
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Define the MLPClassifier
        self.model = MLPClassifier(hidden_layer_sizes=(10,), activation='relu',
                                   solver='adam', max_iter=500, random_state=random_state)
        
        # Train the model
        self.model.fit(X_train, y_train)
        print("Model training completed.")
        
        # Evaluate the model
        y_pred = self.model.predict(X_val)
        print("Validation Classification Report:")
        print(classification_report(y_val, y_pred))
        
        # Save the model and scaler
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        print(f"Model saved to {self.model_path} and scaler saved to {self.scaler_path}.")
    
    def preprocess_features(self, landmarks):
        """
        Converts Mediapipe landmarks to a feature vector.
        
        Parameters:
        - landmarks: List of Mediapipe landmarks.
        
        Returns:
        - features: Normalized feature vector.
        """
        features = []
        for lm in landmarks:
            features.extend([lm.x, lm.y, lm.z])
        # If less than required features, pad with zeros
        if len(features) < self.num_features:
            features += [0.0] * (self.num_features - len(features))
        else:
            features = features[:self.num_features]
        # Convert to NumPy array
        features = np.array(features).reshape(1, -1)
        # Scale features
        features = self.scaler.transform(features)
        return features
    
    def classify_gesture(self, features):
        """
        Classifies the gesture based on the input features.
        
        Parameters:
        - features: Normalized feature vector.
        
        Returns:
        - predicted_class: The predicted gesture class.
        - confidence: Probability of the predicted class.
        """
        probabilities = self.model.predict_proba(features)[0]
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]
        return predicted_class, confidence
    
    def execute_action(self, gesture_class):
        """
        Executes the corresponding action based on the gesture class.
        
        Parameters:
        - gesture_class: The classified gesture.
        """
        # Define your gesture-action mapping here
        gesture_actions = {
                    0:"Left Click",
                    1:"Right Click",
                    2:"Double Click",
                    3:"Zoom In",
                    4:"Zoom Out",
                    5:"Scroll Up",
                    6:"Scroll Down",
                    7:"Scroll Left",
                    8:"Scroll Right"
                    }
        action = gesture_actions.get(gesture_class, "Unknown Gesture")
        print(f"Action Executed: {action}")
        # Implement actual actions using libraries like pynput or autopy if needed
    
    def run(self):
        """
        Starts the real-time gesture recognition and executes corresponding actions.
        """
        if self.model is None or self.scaler is None:
            print("Model and scaler must be trained and loaded before running.")
            return
        
        # Initialize Mediapipe Hands
        with self.mp_hands.Hands(static_image_mode=False,
                                 max_num_hands=2,
                                 min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5) as hands:
            # Initialize video capture
            cap = cv2.VideoCapture(0)
            # Optionally reduce the resolution for better performance
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            
            if not cap.isOpened():
                print("Error: Could not open webcam.")
                return
            
            print("Starting video capture. Press 'Esc' to exit.")
            while True:
                success, image = cap.read()
                if not success:
                    print("Failed to grab frame.")
                    break
                
                # Flip the image horizontally for a mirror view
                image = cv2.flip(image, 1)
                # Convert the BGR image to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # Process the image and find hands
                results = hands.process(image_rgb)
                
                gesture = "No Gesture"
                confidence = 0.0
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw hand landmarks on the image
                        self.mp_drawing.draw_landmarks(
                            image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Extract features from all detected hands
                    all_features = []
                    for hand_landmarks in results.multi_hand_landmarks:
                        features = []
                        for lm in hand_landmarks.landmark:
                            features.extend([lm.x, lm.y, lm.z])
                        all_features.extend(features)
                    
                    # Ensure the feature vector has the correct length
                    if len(all_features) < self.num_features:
                        all_features += [0.0] * (self.num_features - len(all_features))
                    else:
                        all_features = all_features[:self.num_features]
                    
                    # Convert to NumPy array and reshape
                    features = np.array(all_features).reshape(1, -1)
                    # Normalize features
                    features = self.scaler.transform(features)
                    
                    # Classify gesture
                    predicted_class, conf = self.classify_gesture(features)
                    gesture = f"Gesture: {predicted_class} ({conf:.2f})"
                    confidence = conf
                    
                    # Execute corresponding action
                    self.execute_action(predicted_class)
                
                # Display gesture and confidence on the image
                cv2.putText(image, gesture, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                
                # Show the image
                cv2.imshow('AirTrackPad', image)
                
                # Exit on pressing 'Esc'
                if cv2.waitKey(5) & 0xFF == 27:
                    break
            
            cap.release()
            cv2.destroyAllWindows()