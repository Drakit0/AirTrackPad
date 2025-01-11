import time
import keyboard
import pyautogui
import threading

# Global states
state = {
    "left_index_middle_thumb": False,  # Indicates the gesture for left mouse click
    "right_index_middle_thumb": False,  # Indicates the gesture for right mouse click
    "closed_hand": False,  # Indicates the closed hand gesture for dragging
    "two_fingers_direction": None,  # Tracks two-finger gestures for scrolling or zooming
    "last_left_click": 0  # Tracks the last time a left click was performed (for double click)
}

# Gesture functions
def click(state):
    """Handles left and right clicks based on gestures."""
    while True:
        # Handle left mouse click (single or double)
        if state["left_index_middle_thumb"]:
            if time.time() - state["last_left_click"] < 0.3:
                pyautogui.doubleClick(button="left")
                print("Double left click")
            else:
                pyautogui.click(button="left")
                print("Single left click")
            state["last_left_click"] = time.time()
            state["left_index_middle_thumb"] = False

        # Handle right mouse click
        if state["right_index_middle_thumb"]:
            pyautogui.click(button="right")
            print("Right click")
            state["right_index_middle_thumb"] = False

        time.sleep(0.1)

def drag(state):
    """Handles drag gesture (not implemented yet)."""
    while True:
        if state["closed_hand"] and state["left_index_middle_thumb"]:
            print("Dragging (function not implemented yet).")
        time.sleep(0.1)

def zoom(state):
    """Handles zoom in and zoom out gestures."""
    while True:
        if state["two_fingers_direction"] == "zoom_out":
            pyautogui.hotkey("ctrl", "-")
            print("Zooming out...")
        elif state["two_fingers_direction"] == "zoom_in":
            pyautogui.hotkey("ctrl", "+")
            print("Zooming in...")
        state["two_fingers_direction"] = None
        time.sleep(0.1)

def scroll(state):
    """Handles scrolling gestures."""
    while True:
        direction = state["two_fingers_direction"]
        if direction == "scroll_up":
            pyautogui.scroll(200)
            print("Scrolling up")
        elif direction == "scroll_down":
            pyautogui.scroll(-200)
            print("Scrolling down")
        elif direction == "scroll_left":
            pyautogui.hscroll(-10)
            print("Scrolling left")
        elif direction == "scroll_right":
            pyautogui.hscroll(10)
            print("Scrolling right")
        state["two_fingers_direction"] = None
        time.sleep(0.1)

def switch_windows(state):
    """Handles switching between windows."""
    while True:
        direction = state["two_fingers_direction"]
        if direction == "window_right":
            pyautogui.hotkey("alt", "tab")
            print("Switching to the next window")
        elif direction == "window_left":
            pyautogui.hotkey("shift", "alt", "tab")
            print("Switching to the previous window")
        state["two_fingers_direction"] = None
        time.sleep(0.1)

# Keyboard input management
def handle_keyboard_input(state):
    """Updates the global state based on keyboard input."""
    while True:
        state["left_index_middle_thumb"] = keyboard.is_pressed('a')  # Simulates left click
        state["right_index_middle_thumb"] = keyboard.is_pressed('b')  # Simulates right click
        state["closed_hand"] = keyboard.is_pressed('c')  # Simulates closed hand gesture

        if keyboard.is_pressed('d'):
            state["two_fingers_direction"] = "zoom_in"
        elif keyboard.is_pressed('e'):
            state["two_fingers_direction"] = "zoom_out"
        elif keyboard.is_pressed('f'):
            state["two_fingers_direction"] = "scroll_up"
        elif keyboard.is_pressed('g'):
            state["two_fingers_direction"] = "scroll_down"
        elif keyboard.is_pressed('h'):
            state["two_fingers_direction"] = "scroll_left"
        elif keyboard.is_pressed('i'):
            state["two_fingers_direction"] = "scroll_right"
        elif keyboard.is_pressed('j'):
            state["two_fingers_direction"] = "window_left"
        elif keyboard.is_pressed('k'):
            state["two_fingers_direction"] = "window_right"
        
        if keyboard.is_pressed('esc'):
            print("Exiting the program.")
            break

        time.sleep(0.1)

# Main function
def main():
    print("Starting simulation. Use the following keys to test gestures:")
    print("- 'a': Left click")
    print("- 'b': Right click")
    print("- 'c': Close hand (for dragging)")
    print("- 'd': Zoom in")
    print("- 'e': Zoom out")
    print("- 'f': Scroll up")
    print("- 'g': Scroll down")
    print("- 'h': Scroll left")
    print("- 'i': Scroll right")
    print("- 'j': Switch to the previous window")
    print("- 'k': Switch to the next window")
    print("- 'esc': Exit the program")

    # Create threads for each functionality
    threads = [
        threading.Thread(target=click, args=(state,)),
        threading.Thread(target=drag, args=(state,)),
        threading.Thread(target=zoom, args=(state,)),
        threading.Thread(target=scroll, args=(state,)),
        threading.Thread(target=switch_windows, args=(state,)),
        threading.Thread(target=handle_keyboard_input, args=(state,))
    ]

    # Start the threads
    for thread in threads:
        thread.daemon = True
        thread.start()

    # Keep the program running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting the program.")

if __name__ == "__main__":
    main()
