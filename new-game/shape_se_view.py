"""
SHAPE-SE game view.
"""
from abc import ABC, abstractmethod
import pygame
from pygame import mixer
import cv2 as cv
import numpy as np


class ShapeSEView(ABC):
    """
    Create abstract class for game view.

    Attributes:
        _display_size: The pixel dimensions of the intended display to the
            user.
    """

    def __init__(self, display_size):
        """
        Initialize the game view object.

        Args:
            display_size (tuple): Size of the display in pixels.
        """
        self._display_size = display_size

    @abstractmethod
    def initialize_view(self):
        """
        Initialize the game view.
        """

    @abstractmethod
    def display_frame(self, frame, timer_text, camera_mask, flag_overlay=None):
        """
        Display the current frame.

        Args:
            frame (numpy.ndarray): Current frame to display.
            timer_text (str): Current timer value.
            camera_mask (numpy.ndarray): Current camera mask.
            flag_overlay (numpy.ndarray, optional): Sergipe flag overlay.
        """

    @abstractmethod
    def display_win(self, score):
        """
        Display the win screen.

        Args:
            score (int): Current score.
        """

    @abstractmethod
    def display_game_over(self, score):
        """
        Display the game over screen.

        Args:
            score (int): Final player score.
        """


class PygameViewer(ShapeSEView):
    """
    Initializing game window view.

    Attributes:
        _BLACK (tuple): RGB value for black.
        _WHITE (tuple): RGB value for white.
        _GREEN (tuple): RGB value for green.
        _RED (tuple): RGB value for red.
        _FONT (str): Font name.
        _FONT_SIZE (int): Font size.
        _screen (pygame.Surface): The game window.
        _font (pygame.font.SysFont): The font used to display text.
    """

    # RGB values for colors
    _BLACK = (0, 0, 0)
    _WHITE = (255, 255, 255)
    _GREEN = (0, 255, 0)
    _RED = (255, 0, 0)
    # Font settings
    _FONT = "Arial"
    _FONT_SIZE = 32

    def __init__(self, display_size):
        """
        Initialize the pygame viewer.

        Args:
            display_size (tuple): Size of the display in pixels.
        """
        super().__init__(display_size)
        self._screen = None
        self._font = None
        # Load the Sergipe flag with transparency
        self._flag_overlay = cv.imread("images/flag_overlay.png", cv.IMREAD_UNCHANGED)
        if self._flag_overlay is not None:
            # Resize the flag to match the display size
            self._flag_overlay = cv.resize(self._flag_overlay, display_size)
            # Convert to RGBA if it's not already
            if self._flag_overlay.shape[2] == 3:
                # Add alpha channel
                alpha = np.ones((display_size[1], display_size[0], 1), dtype=self._flag_overlay.dtype) * 128
                self._flag_overlay = cv.merge((self._flag_overlay, alpha))
        else:
            print("Warning: Could not load flag overlay image")

    def initialize_view(self):
        """
        Initialize the pygame window and font.
        """
        pygame.init()
        mixer.init()
        self._screen = pygame.display.set_mode(self._display_size)
        pygame.display.set_caption("SHAPE-SE")
        self._font = pygame.font.SysFont(self._FONT, self._FONT_SIZE)

    def _overlay_flag(self, frame):
        """
        Overlay the Sergipe flag on the frame with transparency.

        Args:
            frame (numpy.ndarray): The frame to overlay the flag on.

        Returns:
            numpy.ndarray: The frame with the flag overlay.
        """
        if self._flag_overlay is None:
            return frame

        # Create a copy of the frame
        result = frame.copy()

        # Extract the alpha channel from the flag overlay
        alpha = self._flag_overlay[:, :, 3] / 255.0

        # Reshape alpha for broadcasting
        alpha = alpha[:, :, np.newaxis]

        # Blend the flag with the frame based on alpha
        for c in range(3):  # For each color channel
            result[:, :, c] = (1 - alpha) * frame[:, :, c] + alpha * self._flag_overlay[:, :, c]

        return result

    def display_frame(self, frame, timer_text, camera_mask, flag_overlay=None):
        """
        Display the frame on the game window.

        Args:
            frame (numpy.ndarray): The frame to be displayed.
            timer_text (str): The timer text to be displayed.
            camera_mask (numpy.ndarray): The mask to be overlaid on the frame.
            flag_overlay (numpy.ndarray, optional): Sergipe flag overlay.
        """
        # Apply the camera mask to the frame
        frame = cv.bitwise_and(frame, camera_mask)

        # Overlay the flag if provided
        if flag_overlay is not None:
            frame = self._overlay_flag(frame)

        # Convert the frame to a pygame surface
        frame = pygame.transform.rotate(pygame.surfarray.make_surface(frame), -90)

        # Display the frame
        self._screen.blit(frame, (0, 0))

        # Display the timer
        counting_text = self._font.render(timer_text, 1, self._WHITE)
        counting_rect = counting_text.get_rect(
            topright=self._screen.get_rect().topright
        )
        self._screen.blit(counting_text, counting_rect)

        # Update the display
        pygame.display.update()

    def display_win(self, score):
        """
        Display the win screen.

        Args:
            score (int): The player's score.
        """
        self._screen.fill(self._BLACK)

        # Display the win message
        win_text = self._font.render("Viva Sergipe!", 1, self._GREEN)
        win_rect = win_text.get_rect(center=self._screen.get_rect().center)
        self._screen.blit(win_text, win_rect)

        # Display the score
        score_text = self._font.render(f"Score: {score:.2f}%", 1, self._WHITE)
        score_rect = score_text.get_rect(
            center=(self._screen.get_rect().centerx,
                   self._screen.get_rect().centery + 50)
        )
        self._screen.blit(score_text, score_rect)

        # Update the display
        pygame.display.update()

        # Wait for a key press
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN:
                    waiting = False

    def display_game_over(self, score):
        """
        Display the game over screen.

        Args:
            score (int): The player's score.
        """
        self._screen.fill(self._BLACK)

        # Display the game over message
        game_over_text = self._font.render("Game Over", 1, self._RED)
        game_over_rect = game_over_text.get_rect(center=self._screen.get_rect().center)
        self._screen.blit(game_over_text, game_over_rect)

        # Display the score
        score_text = self._font.render(f"Score: {score:.2f}%", 1, self._WHITE)
        score_rect = score_text.get_rect(
            center=(self._screen.get_rect().centerx,
                   self._screen.get_rect().centery + 50)
        )
        self._screen.blit(score_text, score_rect)

        # Update the display
        pygame.display.update()

        # Wait for a key press
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN:
                    waiting = False
