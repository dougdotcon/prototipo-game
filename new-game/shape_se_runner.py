"""
Main runner code for SHAPE-SE game.
"""
import sys
import os
import time
from datetime import datetime
from shape_se_controller import OpenCVController
from shape_se_view import PygameViewer
from shape_se_model import ShapeSEGame

# Set up view constants
CAMERA_INDEX = 0
DISPLAY_SIZE = (640, 480)
GAME_DURATION = 300  # 5 minutes in seconds

# Create snapshots directory if it doesn't exist
os.makedirs("snapshots", exist_ok=True)

def game_play():
    """
    Run the main game loop.

    Returns:
        str: The next game state.
    """
    # Get a random mask
    mask, joints_file = game_model.get_random_mask()
    
    # Start the timer
    game_controller.start_timer()
    
    # Main game loop
    while True:
        # Get the current frame
        current_frame = game_controller.get_display_frame()
        
        # Get the current timer value
        current_timer_value = game_controller.get_timer_string()
        
        # Display the frame with the mask and timer
        game_view.display_frame(current_frame, current_timer_value, mask)
        
        # Check for user input
        if game_controller.next_screen() == "quit":
            sys.exit()
        
        # Check if the timer has ended
        if game_controller.determine_end_timer():
            final_frame = current_frame
            break
    
    # Analyze the final frame
    game_model.analyze_frame(final_frame)
    game_model.parse_for_joint_positions()
    accuracy = game_model.compute_accuracy(joints_file)
    
    # Check if the player succeeded
    if game_model.is_success():
        # Capture the player's image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"snapshots/success_{timestamp}.jpg"
        game_controller.capture_image(filename)
        
        # Display the win screen
        game_view.display_win(accuracy)
        return "game_complete"
    else:
        # Display the game over screen
        game_view.display_game_over(accuracy)
        return "game_complete"

def game_complete():
    """
    Display the game complete screen and exit.

    Returns:
        None
    """
    # Release the camera
    game_controller.release_camera()
    
    # Exit the game
    sys.exit()

# Define the game states
GAME_STATES = {
    "game_play": game_play,
    "game_complete": game_complete,
}

if __name__ == "__main__":
    # Create controller, model, and view objects
    game_controller = OpenCVController(CAMERA_INDEX, GAME_DURATION)
    game_view = PygameViewer(DISPLAY_SIZE)
    game_model = ShapeSEGame()
    
    # Initialize the view
    game_view.initialize_view()
    
    # Set the current game state
    current_game_state = "game_play"
    
    # Run the game until completion
    while True:
        current_game_state = GAME_STATES[current_game_state]()
        if current_game_state == "game_complete":
            break
    
    # Close the game
    current_game_state = "game_complete"
    GAME_STATES[current_game_state]()
