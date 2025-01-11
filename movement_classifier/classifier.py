import os
import joblib
import numpy as np
import mediapipe as mp
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


class NeuralClassifier:
    
    def __init__(self, model_path='gesture_classifier.pkl', scaler_path='scaler.pkl', num_features=154, num_classes=11):
        """
        Initializes the AirTrackPad system.
        
        Args
        ----
            - model_path: Path to the saved classifier model.
            - scaler_path: Path to the saved scaler.
            - num_features: Number of input features 14*num_clases.
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
        
        Args
        ----
            - X: Feature matrix.
            - y: Labels.
            - test_size: Proportion of the dataset to include in the test split.
            - random_state: Controls the shuffling applied to the data before applying the split.
        """
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X) # Normalize features
        
        X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state, stratify=y) # Train test split
        self.model = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', solver='adam', max_iter=500, random_state=random_state) # Define MLPClassifier
        
        self.model.fit(X_train, y_train) # Train
        print("Model training completed.")
        y_pred = self.model.predict(X_val) # Predict
        
        print("Validation Classification Report:")
        print(classification_report(y_val, y_pred)) # Evaluate
        
        joblib.dump(self.model, self.model_path) # Save model
        joblib.dump(self.scaler, self.scaler_path)
        
        print(f"Model saved to {self.model_path} and scaler saved to {self.scaler_path}.")
    
    def preprocess_features(self, landmarks):
        """
        Converts Mediapipe landmarks to a feature vector.
        
        Args
        ----
            - landmarks: List of Mediapipe landmarks.
        
        Returns
        -------
            - features: Normalized feature vector.
        """
        
        features = []
        for lm in landmarks:
            features.extend([lm.x, lm.y, lm.z])
            
        if len(features) < self.num_features:
            features += [0.0] * (self.num_features - len(features))
            
        else:
            features = features[:self.num_features]
        
        features = np.array(features).reshape(1, -1) # Convert to NumPy array
        
        features = self.scaler.transform(features)# Scale features
        
        return features
    
    def classify_gesture(self, features: np.ndarray):
        """
        Classifies the gesture based on the input features.
        
        Args
        ----
            - features: Normalized feature vector.
        
        Returns
        -------
            - predicted_class: The predicted gesture class.
            - confidence: Probability of the predicted class.
        """
        
        probabilities = self.model.predict_proba(features)[0]
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]
        
        return predicted_class, confidence
    
    