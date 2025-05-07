"""
This script is used to make masks for users to fit into and saves the original
frame and the mask to images/poses and images/masks respectively.
"""
import os
import cv2 as cv
import numpy as np

# Create directories if they don't exist
os.makedirs("images/poses", exist_ok=True)
os.makedirs("images/masks", exist_ok=True)

# Open a camera for video capturing.
CAMERA = cv.VideoCapture(0)
# Image name to be analyzed.
MASK_NAME = "sergipe_mask"

def get_camera_frame():
    """
    This function returns the user's camera frame.

    Returns:
        frame (numpy.ndarray): A 3-D numpy array of RGB values representing
            the user's camera output.
    """
    # Capture frame-by-frame
    _, frame = CAMERA.read()
    # Flip the frame horizontally for a selfie-view display
    frame = cv.flip(frame, 1)
    return frame

def analyze_camera_frame(frame):
    """
    This function analyzes a given camera frame to make a mask to be displayed
    during the actual game.

    Args:
        frame (numpy.ndarray): A 3-D numpy array of RGB values representing
            the user's camera output.

    Returns:
        tuple: A tuple containing the original frame and the mask.
    """
    # Convert the frame to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    
    # Define range of skin color in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    # Threshold the HSV image to get only skin colors
    mask = cv.inRange(hsv, lower_skin, upper_skin)
    
    # Bitwise-AND mask and original image
    res = cv.bitwise_and(frame, frame, mask=mask)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv.erode(mask, kernel, iterations=2)
    mask = cv.dilate(mask, kernel, iterations=2)
    
    # Convert the mask to BGR for display
    mask_bgr = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    
    # Display the original frame and the mask
    cv.imshow('Original', frame)
    cv.imshow('Mask', mask_bgr)
    
    return hsv, mask

def release_camera():
    """
    Release the camera at the end of the script to ensure that the camera is
    no longer being called.
    """
    CAMERA.release()
    cv.destroyAllWindows()

def main():
    """
    This is the runner function to create and save masks.
    """
    print("Position yourself to create a mask for the SHAPE-SE game.")
    print("Press 'd' to capture the current frame and save it as a mask.")
    print("Press 'q' to quit without saving.")

    while True:
        frame = get_camera_frame()
        frame, mask = analyze_camera_frame(frame)
        
        # Break if user presses 'd' to save or 'q' to quit
        key = cv.waitKey(1) & 0xFF
        if key == ord("d"):
            # Save the current frame and the mask
            frame = cv.cvtColor(frame, cv.COLOR_HSV2BGR)
            cv.imwrite(f"images/poses/{MASK_NAME}.png", frame)
            cv.imwrite(f"images/masks/{MASK_NAME}.png", mask)
            print(f"Mask saved as {MASK_NAME}")
            break
        elif key == ord("q"):
            print("Exiting without saving.")
            break

    release_camera()

if __name__ == "__main__":
    main()
