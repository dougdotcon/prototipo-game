import pygame
import cv2
import numpy as np
import os
import time
from datetime import datetime
from PIL import Image

class Game:
    def __init__(self, vision_processor, config):
        self.vision = vision_processor
        self.config = config
        self.threshold = config.get("threshold", 95)
        self.shape_path = config.get("shape_path", "assets/shape-se.png")
        self.background_path = config.get("background_path", "assets/flag-se.jpg")

        # Initialize pygame
        pygame.init()

        # Set up display
        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        self.width, self.height = self.screen.get_size()
        pygame.display.set_caption("Shape SE - Reconhecimento Corporal")

        # Load images
        self.load_images()

        # Game state
        self.running = True
        self.victory = False
        self.victory_time = 0
        self.current_percentage = 0
        self.in_menu = True  # Start in menu mode

        # Create snapshots directory if it doesn't exist
        if not os.path.exists("snapshots"):
            os.makedirs("snapshots")

    def load_images(self):
        """Load and prepare the shape and background images."""
        # Print debug info
        print(f"Loading shape from: {self.shape_path}")
        print(f"Current working directory: {os.getcwd()}")

        try:
            # Load shape image
            self.shape_img = pygame.image.load(self.shape_path).convert_alpha()
            print(f"Shape image loaded successfully. Size: {self.shape_img.get_size()}")
            self.shape_img = pygame.transform.scale(self.shape_img, (int(self.height * 0.8), int(self.height * 0.8)))
            print(f"Shape image scaled to: {self.shape_img.get_size()}")

            # Create shape mask for calculations
            shape_surface = pygame.Surface(self.shape_img.get_size(), pygame.SRCALPHA)
            shape_surface.blit(self.shape_img, (0, 0))
            shape_array = pygame.surfarray.array3d(shape_surface)
            self.shape_mask = np.any(shape_array > 0, axis=2).astype(np.uint8) * 255

            # Load background image
            self.background = pygame.image.load(self.background_path).convert()
            self.background = pygame.transform.scale(self.background, (self.width, self.height))

            # Position for the shape (centered)
            self.shape_pos = ((self.width - self.shape_img.get_width()) // 2,
                            (self.height - self.shape_img.get_height()) // 2)
        except Exception as e:
            print(f"Error loading images: {e}")
            raise

    def process_frame(self, frame):
        """Process a frame from the webcam."""
        # Get segmentation mask
        body_mask = self.vision.get_segmentation_mask(frame)

        # Resize body mask to match shape mask size
        body_mask_resized = cv2.resize(body_mask, (self.shape_mask.shape[1], self.shape_mask.shape[0]))

        # Calculate fill percentage
        percentage = self.vision.calculate_fill_percentage(body_mask_resized, self.shape_mask)
        self.current_percentage = percentage

        # Check for victory
        if percentage >= self.threshold and not self.victory:
            self.victory = True
            self.victory_time = time.time()
            self.save_snapshot(frame)

        # Reset victory after 3 seconds
        if self.victory and time.time() - self.victory_time > 3:
            self.victory = False

        return frame, body_mask

    def save_snapshot(self, frame):
        """Save a snapshot of the victory moment."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"snapshots/victory_{timestamp}.png"

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Save using PIL
        img = Image.fromarray(rgb_frame)
        img.save(filename)
        print(f"Snapshot saved: {filename}")

    def render_menu(self):
        """Render the menu screen."""
        # Draw background
        self.screen.blit(self.background, (0, 0))

        # Draw title
        title_font = pygame.font.Font(None, 72)
        title_text = title_font.render("Shape SE - Reconhecimento Corporal", True, (255, 255, 255))
        title_rect = title_text.get_rect(center=(self.width // 2, self.height // 4))
        self.screen.blit(title_text, title_rect)

        # Draw instructions
        instruction_font = pygame.font.Font(None, 48)
        instructions = [
            "Posicione seu corpo para preencher o contorno do estado de Sergipe",
            "Tente atingir 95% de preenchimento para vencer",
            "",
            "Pressione ESPAÇO para iniciar",
            "Pressione ESC para sair"
        ]

        for i, instruction in enumerate(instructions):
            text = instruction_font.render(instruction, True, (255, 255, 255))
            text_rect = text.get_rect(center=(self.width // 2, self.height // 2 + i * 60))
            self.screen.blit(text, text_rect)

        pygame.display.flip()

    def render(self, frame, body_mask):
        """Render the game screen."""
        # Convert webcam frame to pygame surface and scale to full screen
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
        frame_surface = pygame.transform.scale(frame_surface, (self.width, self.height))

        # Draw webcam frame as background
        self.screen.blit(frame_surface, (0, 0))

        # Draw background with transparency
        background_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        background_surface.blit(self.background, (0, 0))
        background_surface.set_alpha(100)  # Semi-transparent background
        self.screen.blit(background_surface, (0, 0))

        # Draw shape with progress overlay
        shape_overlay = self.shape_img.copy()

        # Create a progress overlay based on current percentage
        if not self.victory:
            progress_alpha = int(min(self.current_percentage, 90) * 255 / 100)
            progress_surface = pygame.Surface(shape_overlay.get_size(), pygame.SRCALPHA)
            progress_surface.fill((0, 255, 0, progress_alpha))
            shape_overlay.blit(progress_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

        # Draw shape
        self.screen.blit(shape_overlay, self.shape_pos)

        # Draw contorno (outline) on top of everything
        try:
            contorno_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "contorno-mapa-SE.png")
            if not hasattr(self, 'contorno_img'):
                print(f"Loading contorno from: {contorno_path}")
                self.contorno_img = pygame.image.load(contorno_path).convert_alpha()
                self.contorno_img = pygame.transform.scale(self.contorno_img, (int(self.height * 0.8), int(self.height * 0.8)))
            self.screen.blit(self.contorno_img, self.shape_pos)
        except Exception as e:
            print(f"Error loading contorno: {e}")

        # Draw percentage
        font = pygame.font.Font(None, 36)
        percentage_text = font.render(f"Preenchimento: {self.current_percentage:.1f}%", True, (255, 255, 255))
        self.screen.blit(percentage_text, (20, 20))

        # Draw victory message
        if self.victory:
            victory_font = pygame.font.Font(None, 72)
            victory_text = victory_font.render("Parabéns!", True, (255, 255, 0))
            text_rect = victory_text.get_rect(center=(self.width // 2, 100))
            self.screen.blit(victory_text, text_rect)

        pygame.display.flip()

    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE and self.in_menu:
                    self.in_menu = False  # Exit menu and start game

    def run(self):
        """Main game loop."""
        clock = pygame.time.Clock()

        while self.running:
            # Handle events
            self.handle_events()

            if self.in_menu:
                # Render menu
                self.render_menu()
            else:
                # Get frame from webcam
                frame = self.vision.get_frame()
                if frame is None:
                    continue

                # Process frame
                frame, body_mask = self.process_frame(frame)

                # Render
                self.render(frame, body_mask)

            # Cap at 30 FPS
            clock.tick(30)

        # Clean up
        pygame.quit()
        self.vision.release()
