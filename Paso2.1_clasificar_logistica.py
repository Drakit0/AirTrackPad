import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

data = pd.read_csv('gestures_data/gestures_landmarks.csv')

print("Primeras filas del DataFrame de características (X):")
print(data.head())

X = data.drop(columns=['gesture'])  
y = data['gesture']  

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy (Regresión Logística): {accuracy * 100:.2f}%")

joblib.dump(model, 'gesture_logistic_model.pkl')
joblib.dump(scaler, 'gesture_scaler.pkl')
joblib.dump(label_encoder, 'gesture_label_encoder.pkl')

print("Predicciones (primeras 10):", y_pred[:30])
print("Etiquetas reales (primeras 10):", y_test[:30])
