import time
from trainer import GestureTrainer 

gestures = [
    "Left Click",
    "Right Click",
    "Double Click",
    "Zoom In",
    "Zoom Out",
    "Scroll Up",
    "Scroll Down",
    "Scroll Left",
    "Scroll Right",
    "Pointing",
    "No Gesture"]

trainer = GestureTrainer(gestures=gestures)
trainer.collect_data(num_samples_per_gesture=200)

# t_0 = time.time()
trainer.train_from_saved_data()
# print(f"Training took {time.time() - t_0} seconds.")

