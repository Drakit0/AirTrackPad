import time
import pyautogui
import numpy as np
import collections
from threading import Lock

class ActionManager:
    
    def __init__(self, frame_width, frame_height):
        self.action = Lock()
        self.prev_positions = collections.deque(maxlen=5)
        self.frame_width = frame_width
        self.frame_height = frame_height
        
    def handle_gesture(self, gesture, landmarks=None):
        """
        Executes the corresponding action based on the detected gesture.
        """
        if gesture == 0:
            self._left_click()
            
        elif gesture == 1:
            self._right_click()
            
        elif gesture == 2:
            self._double_click()
            
        elif gesture == 3:
            self._zoom_in()
            
        elif gesture == 4:
            self._zoom_out()
            
        elif gesture == 5:
            self._scroll_up()
            
        elif gesture == 6:
            self._scroll_down()
            
        elif gesture == 7:
            self._scroll_left()
            
        elif gesture == 8:
            self._scroll_right()
            
        elif gesture == 9:
            self._pointing(landmarks)
            
        else:
            self._no_gesture()
        
    def _left_click(self):
        """Handles left click gesture."""
        
        self.action.acquire()
        pyautogui.click(button="left")
        print("Left click")
        time.sleep(0.25)
        self.action.release()
        
    def _right_click(self):
        """Handles right click gesture."""
        
        self.action.acquire()
        pyautogui.click(button="right")
        print("Right click")
        time.sleep(0.25)
        self.action.release()
        
    def _double_click(self):
        """Handles double click gesture."""
        
        self.action.acquire()
        pyautogui.doubleClick(button="left")
        print("Double click")
        time.sleep(0.25)
        self.action.release()
        
    def _zoom_in(self):
        """Handles zoom in gesture."""
        
        self.action.acquire()
        pyautogui.hotkey("ctrl", "+")
        print("Zoom in")
        time.sleep(0.25)
        self.action.release()
        
    def _zoom_out(self):
        """Handles zoom out gesture."""
        
        self.action.acquire()
        pyautogui.hotkey("ctrl", "-")
        print("Zoom out")
        time.sleep(0.25)
        self.action.release()
        
    def _scroll_up(self):
        """Handles scroll up gesture."""
        
        self.action.acquire()
        pyautogui.scroll(200)
        print("Scroll up")
        time.sleep(0.25)
        self.action.release()
        
    def _scroll_down(self):
        """Handles scroll down gesture."""
        
        self.action.acquire()
        pyautogui.scroll(-200)
        print("Scroll down")
        time.sleep(0.25)
        self.action.release()
        
    def _scroll_left(self):
        """Handles scroll left gesture."""
        
        self.action.acquire()
        pyautogui.hscroll(-10)
        print("Scroll left")
        time.sleep(0.25)
        self.action.release()
        
    def _scroll_right(self):
        """Handles scroll right gesture."""
        
        self.action.acquire()
        pyautogui.hscroll(10)
        print("Scroll right")
        time.sleep(0.25)
        self.action.release()
        
    def _pointing(self, hand_landmarks):
        """Handles pointing gesture by moving the mouse cursor."""
        
        screen_x, screen_y = self.map_coordinates(hand_landmarks)
        
        self.prev_positions.append((screen_x, screen_y)) # Smoothing

        avg_x = int(np.mean([pos[0] for pos in self.prev_positions]))
        avg_y = int(np.mean([pos[1] for pos in self.prev_positions]))

        pyautogui.moveTo(avg_x, avg_y, duration=0.1) # Move cursor
        print(f"Pointing to ({avg_x}, {avg_y})")
        
    def map_coordinates(self, hand_landmarks):
        """
        Maps the coordinates from webcam frame to screen size.
        """
        
        index_finger_tip = hand_landmarks[8]
        x = int(index_finger_tip.x * self.frame_width)
        y = int(index_finger_tip.y * self.frame_height)
        
        screen_width, screen_height = pyautogui.size()
        screen_x = np.interp(x, (0, self.frame_width), (0, screen_width))
        screen_y = np.interp(y, (0, self.frame_height), (0, screen_height))
        
        return int(screen_x), int(screen_y)
        
    def _no_gesture(self):
        """Handles no gesture."""
        
        print("No gesture")
        

        