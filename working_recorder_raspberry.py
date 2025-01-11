from picamera2 import Picamera2, Preview
import cv2
import numpy as np
import skimage as ski

picam2 = Picamera2()
# picam2.preview_configuration.main.size = (4608, 2592)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
# picam2.configure("preview")
picam2.start()

out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (640, 480))

while True:
    frame = picam2.capture_array()
    
    frame = cv2.flip(frame, 1)
    sobel_processed = ski.filters.sobel(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) # , mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    sobel_processed = cv2.cvtColor((sobel_processed * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    
    # gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    cv2.imshow('frame', sobel_processed )
    
    # out.write(frame)
    if cv2.waitKey(1) == ord('q'):
        break

picam2.stop()
out.release()
cv2.destroyAllWindows()