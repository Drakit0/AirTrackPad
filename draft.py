import os
import cv2
import time
from picamera2 import Picamera2


# picam2 = Picamera2()

# picam2.start()

# Initialize the image counter
# image_counter = 1

# Create the processed_data directory
# processed_data_dir = "raw_data"
# os.makedirs(processed_data_dir, exist_ok=True)

# while True:
#     Capture a frame from the camera
#     frame = picam2.capture_array()

#     # Display the frame using OpenCV
#     cv2.imshow("Camera Feed", frame)

#     Wait for a key press
#     key = cv2.waitKey(1) & 0xFF

#     If 'f' is pressed, capture and save the image
#     if key == ord('f'):
#         filename = f'{processed_data_dir}/{image_counter:03d}.jpg'
#         cv2.imwrite(filename, frame)
#         print(f'Captured {filename}')
#         image_counter += 1

#     If 'q' is pressed, exit the loop
#     elif key == ord('q'):
#         break

# Release resources
# cv2.destroyAllWindows()
# picam2.stop()


# Initialize the camera
picam2 = Picamera2()

# Configure the camera
picam2.configure(picam2.create_still_configuration())
file_name = "raw_data"
os.makedirs(file_name, exist_ok=True)

# Start the camera
picam2.start()

# Capture 5 images
for i in range(30):
    
    while True:
        pepe = input("Press 'f' to capture: ").lower()
        if pepe == 'f' or pepe == 'q':
            break
        
    if pepe == 'q':
        break
    
    filename = f'{file_name}/image_{i+1}.jpg'
    picam2.capture_file(filename)
    print(f'Captured {filename}')
    time.sleep(1)  # Wait for 1 second between captures

# Stop the camera
picam2.stop()

# picam2.start_and_record_video('mamaguebaso.mp4', duration = 10)