import time
from trainer import GestureTrainer 

# gestures = [
#     "Left Click",
#     "Right Click",
#     "Double Click",
#     "Zoom In",
#     "Zoom Out",
#     "Scroll Up",
#     "Scroll Down",
#     "Scroll Left",
#     "Scroll Right"
# ]

# trainer = GestureTrainer(gestures=gestures)
# trainer.collect_data(num_samples_per_gesture=200)

# t_0 = time.time()
# trainer.train_from_saved_data()
# print(f"Training took {time.time() - t_0} seconds.")

from clasifier import AirTrackPad  # Assuming the class is saved in air_track_pad.py

def main():
    air_track_pad = AirTrackPad(model_path="movement_classifier/models/gesture_classifier.pkl", scaler_path="movement_classifier/models/scaler.pkl")
    
    if not air_track_pad.model or not air_track_pad.scaler:
        print("Model and scaler not found. Please train the model first using GestureTrainer.")
        return
    
    air_track_pad.run()

if __name__ == "__main__":
    main()