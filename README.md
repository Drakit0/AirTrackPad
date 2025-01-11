# 🌟 *Air Trackpad* 🌟

## 🖐️ Introduction

In this project, we utilize *MediaPipe, **OpenCV, and a **Raspberry Pi* to create an innovative air trackpad system. The system captures hand gestures through a camera and translates them into actions such as cursor movement, clicking, scrolling, and zooming. The project combines hand detection, gesture recognition, and computer vision techniques to deliver a seamless hands-free interface.

This project was developed as the *final assignment for the Computer Vision course*, showcasing advanced real-time video processing and gesture classification.

---

## ⚙️ Methodology

### ✋ Hand Detection and Tracking

We use *MediaPipe*'s Hand Tracking solution to detect and track key hand landmarks in real-time. The system extracts 3D coordinates for critical points such as fingertips and palm centers. These landmarks form the foundation for gesture classification.

**The detection process includes:**
1. 📸 *Frame Capture*: Real-time video is captured using OpenCV.
2. 🧹 *Preprocessing*: Frames are normalized and filtered to reduce noise.
3. 🖐️ *Landmark Detection*: MediaPipe identifies and tracks hand landmarks.

---

### 🤚 Gesture Recognition

The system recognizes complex gestures by analyzing the configuration and motion of hand landmarks. Gestures are mapped to actions such as:
- 🖱️ *Cursor Movement*: Index finger position defines cursor location.
- 🖱️ *Clicks*: Pinch gestures trigger left or right clicks.
- 📜 *Scrolling*: Vertical or horizontal two-finger swipes scroll the screen.
- 🔍 *Zooming*: Expanding or contracting finger distances adjusts zoom levels.

**To ensure robust detection:**
- We stabilize landmarks with *Optical Flow* and *Kalman Filters* to smooth abrupt movements.
- Gestures are validated using time-based queues to avoid false positives.

---

### 📍 Coordinate Mapping

Detected gestures are mapped to screen coordinates based on the display resolution. This ensures the system dynamically adapts to varying screen sizes and camera positions.

---

### 🧠 Advanced Techniques

1. **Image Stabilization**:
   - Motion smoothing with *Kalman Filters*.
   - Real-time adjustments for lighting and background inconsistencies.
2. **Gesture Validation**:
   - *Frechet Distance* is used to validate trajectories for complex gestures.
   - Each gesture pattern is matched to predefined models.

---

## ✅ Results

The *Air Trackpad* performs reliably under various conditions:
- 🖐️ Gestures are accurately detected in real-time.
- 🚀 Cursor movements are smooth, thanks to stabilization techniques.
- 🖱️ Clicks, scrolling, and zoom actions are intuitive and responsive.

However, in challenging environments with excessive lighting variations or fast hand movements, minor delays or misclassifications may occur.

---

## 🚀 Future Improvements

To enhance the project:
1. 🖱️ Add support for more gestures, such as window switching or custom gestures for application shortcuts.
2. ✂️ Implement real-time background removal to improve detection accuracy in cluttered environments.
3. ⚡ Optimize performance on Raspberry Pi by processing only a region of interest around the detected hand.

---

## 🎥 Video Demo

Check out the video demonstration of the Air Trackpad:  
👉 [Video Demo](#) 

---

## 👩‍💻 Authors

This project was developed by:
- *Lydia Ruiz Martínez* ([LydiaRuizMartinez](https://github.com/LydiaRuizMartinez))  
- *Pablo Tuñón Laguna* ([Drakit0](https://github.com/Drakit0))

---

## 💡 Acknowledgments

Special thanks to the course instructors for their guidance and support!

🎉 *Thank you for exploring our Air Trackpad! We hope you find it inspiring!* 🎉
