"""
SHAPE-SE game controller.
"""
from abc import ABC, abstractmethod
import time
import pygame
import cv2


class ShapeSEController(ABC):
    """
    Create abstract class for the game controller.

    Attributes:
        _start_time (float): Start time of the game.
        _current_time (float): Current time in the countdown timer.
        _game_duration (int): Duration of the game in seconds.
    """

    def __init__(self, game_duration=300):
        """
        Initialize the game controller object with time objects.
        
        Args:
            game_duration (int): Duration of the game in seconds (default: 300 seconds = 5 minutes).
        """
        self._start_time = 0
        self._current_time = 0
        self._game_duration = game_duration

    @property
    def start_time(self):
        """
        Return the start_timer value.
        """
        return self._start_time

    @abstractmethod
    def next_screen(self):
        """
        Get the next screen to display.
        """

    @abstractmethod
    def start_timer(self):
        """
        Start the countdown timer for the game.
        """

    @abstractmethod
    def get_display_frame(self):
        """
        Get the current frame to display.
        """

    @abstractmethod
    def get_timer_string(self):
        """
        Get the current timer value as a string.
        """

    @abstractmethod
    def determine_end_timer(self):
        """
        Determine if the timer has ended.
        """


class OpenCVController(ShapeSEController):
    """
    OpenCV implementation of the ShapeSEController class.

    Attributes:
        _camera_index (int): Index of the camera to use.
        _camera_capture (numpy.ndarray): Current caputured video frame.
    """

    def __init__(self, camera_index, game_duration=300):
        """
        Initialize the OpenCV controller.

        Args:
            camera_index (int): Index of the camera to use.
            game_duration (int): Duration of the game in seconds (default: 300 seconds = 5 minutes).
        """
        super().__init__(game_duration)
        self._camera_index = camera_index
        self._camera_capture = cv2.VideoCapture(self._camera_index)

    @property
    def camera_capture(self):
        """
        Return the camera capture object.
        """
        return self._camera_capture

    @property
    def camera_index(self):
        """
        Return the camera index.
        """
        return self._camera_index

    def release_camera(self):
        """
        Release the camera.
        """
        self._camera_capture.release()

    def next_screen(self):
        """
        This function listens for user keypresses. If ESCAPE or the window X is
        clicked, the function returns "quit" to signify termination of the
        game. If any other key is clicked, function returns "continue" to call
        the next screen to be displayed and if no valid event is detected,
        "stay is returned to ensure that the current screen is maintained.

        Returns:
            str: The next screen to display.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return "quit"
                return "continue"
        return "stay"

    def start_timer(self):
        """
        Start the countdown timer for the game.
        """
        self._start_time = time.time()
        self._current_time = self._start_time

    def get_display_frame(self):
        """
        Get the current frame to display.

        Returns:
            numpy.ndarray: The current frame.
        """
        # Update the current time
        self._current_time = time.time()
        
        # Capture frame-by-frame
        ret, frame = self._camera_capture.read()
        
        # If frame is read correctly ret is True
        if not ret:
            print("Error: Could not read frame")
            return None
        
        # Flip the frame horizontally for a selfie-view display
        frame = cv2.flip(frame, 1)
        
        # Resize the frame to match the display size
        frame = cv2.resize(frame, (640, 480))
        
        # Convert the frame from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        return frame

    def get_timer_string(self):
        """
        Get the current timer value as a string.

        Returns:
            str: The current timer value in the format "MM:SS".
        """
        # Calculate the elapsed time
        elapsed_time = self._current_time - self._start_time
        
        # Calculate the remaining time
        remaining_time = max(0, self._game_duration - elapsed_time)
        
        # Convert to minutes and seconds
        minutes = int(remaining_time // 60)
        seconds = int(remaining_time % 60)
        
        # Return the formatted string
        return f"{minutes:02d}:{seconds:02d}"

    def determine_end_timer(self):
        """
        Determine if the timer has ended.

        Returns:
            bool: True if the timer has ended, False otherwise.
        """
        # Calculate the elapsed time
        elapsed_time = self._current_time - self._start_time
        
        # Return True if the elapsed time is greater than or equal to the game duration
        return elapsed_time >= self._game_duration

    def capture_image(self, filename):
        """
        Capture the current frame and save it to a file.

        Args:
            filename (str): The filename to save the image to.
            
        Returns:
            bool: True if the image was saved successfully, False otherwise.
        """
        # Get the current frame
        frame = self.get_display_frame()
        
        # If frame is None, return False
        if frame is None:
            return False
        
        # Convert the frame from RGB to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Save the frame to a file
        return cv2.imwrite(filename, frame)
