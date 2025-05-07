"""
Create a transparent Sergipe flag overlay for the game.
"""
import os
import cv2
import numpy as np

# Create directories if they don't exist
os.makedirs("images", exist_ok=True)

def create_sergipe_flag(width=640, height=480, alpha=0.3):
    """
    Create a transparent Sergipe flag overlay.
    
    Args:
        width (int): Width of the flag image.
        height (int): Height of the flag image.
        alpha (float): Transparency level (0.0 to 1.0).
        
    Returns:
        numpy.ndarray: RGBA image of the Sergipe flag.
    """
    # Create a blank RGBA image
    flag = np.zeros((height, width, 4), dtype=np.uint8)
    
    # Fill the top half with blue
    flag[:height//2, :, 0] = 255  # Blue channel
    flag[:height//2, :, 3] = int(255 * alpha)  # Alpha channel
    
    # Fill the bottom half with white
    flag[height//2:, :, 0] = 255  # Blue channel
    flag[height//2:, :, 1] = 255  # Green channel
    flag[height//2:, :, 2] = 255  # Red channel
    flag[height//2:, :, 3] = int(255 * alpha)  # Alpha channel
    
    # Draw a yellow star in the center
    center = (width // 2, height // 2)
    radius = min(width, height) // 8
    cv2.circle(flag, center, radius, (0, 255, 255, int(255 * alpha)), -1)
    
    return flag

def main():
    """
    Create and save the Sergipe flag overlay.
    """
    # Create the flag overlay
    flag_overlay = create_sergipe_flag()
    
    # Save the flag overlay
    cv2.imwrite("images/flag_overlay.png", flag_overlay)
    print("Flag overlay saved to images/flag_overlay.png")

if __name__ == "__main__":
    main()
