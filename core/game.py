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

        # Get base directory
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Set paths with absolute paths
        self.shape_path = os.path.join(self.base_dir, config.get("shape_path", "assets/shape-se.png"))
        self.background_path = os.path.join(self.base_dir, config.get("background_path", "assets/flag-se.jpg"))
        self.contorno_path = os.path.join(self.base_dir, "contorno-mapa-SE.png")

        print(f"Shape path: {self.shape_path}")
        print(f"Background path: {self.background_path}")
        print(f"Contorno path: {self.contorno_path}")

        # Check if files exist
        print(f"Shape file exists: {os.path.exists(self.shape_path)}")
        print(f"Background file exists: {os.path.exists(self.background_path)}")
        print(f"Contorno file exists: {os.path.exists(self.contorno_path)}")

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
        self.show_background = False  # Background invisible by default
        self.background_opacity = 180  # Default opacity (0-255)

        # Timer variables
        self.start_time = None
        self.elapsed_time = 0
        self.timer_running = False

        # Debug options
        self.show_body_mask = True  # Show body segmentation mask for debugging

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
            print(f"Loading background from: {self.background_path}")
            try:
                self.background = pygame.image.load(self.background_path).convert()
                print(f"Background loaded successfully. Size: {self.background.get_size()}")
                self.background = pygame.transform.scale(self.background, (self.width, self.height))
                print(f"Background scaled to: {self.background.get_size()}")
            except Exception as e:
                print(f"Error loading background: {e}")
                # Create a fallback background (blue color)
                self.background = pygame.Surface((self.width, self.height))
                self.background.fill((0, 0, 128))  # Dark blue
                print("Created fallback background")

            # Load contorno image
            print(f"Loading contorno from: {self.contorno_path}")
            self.contorno_img = pygame.image.load(self.contorno_path).convert_alpha()
            self.contorno_img = pygame.transform.scale(self.contorno_img, (int(self.height * 0.8), int(self.height * 0.8)))
            print(f"Contorno loaded successfully. Size: {self.contorno_img.get_size()}")

            # Position for the shape (centered)
            self.shape_pos = ((self.width - self.shape_img.get_width()) // 2,
                            (self.height - self.shape_img.get_height()) // 2)
        except Exception as e:
            print(f"Error loading images: {e}")
            import traceback
            traceback.print_exc()
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
        
        # Obter o percentual de excesso (corpo fora do contorno)
        self.excesso_percentual = getattr(self.vision, 'excesso_percentual', 0)

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
        title_rect = title_text.get_rect(center=(self.width // 2, self.height // 5))
        self.screen.blit(title_text, title_rect)

        # Draw instructions
        instruction_font = pygame.font.Font(None, 48)
        instructions = [
            "Posicione seu corpo para preencher o contorno do estado de Sergipe",
            f"Tente atingir {self.threshold}% de preenchimento para vencer",
            "",
            "Controles:",
            "ESPAÇO - iniciar o jogo",
            "B - alternar visibilidade do fundo",
            "+ / - - ajustar opacidade do fundo",
            "T - pausar/retomar o temporizador",
            "R - reiniciar o temporizador",
            "M - mostrar/ocultar visualização do corpo",
            "ESC - sair"
        ]

        # Calcular espaçamento proporcional à altura da tela
        line_spacing = int(self.height * 0.045)  # Reduzido de 60 para 4.5% da altura da tela
        start_y = self.height // 3  # Começa um pouco mais acima

        for i, instruction in enumerate(instructions):
            text = instruction_font.render(instruction, True, (255, 255, 255))
            text_rect = text.get_rect(center=(self.width // 2, start_y + i * line_spacing))
            self.screen.blit(text, text_rect)

        pygame.display.flip()

    def render(self, frame, body_mask=None):
        """Render the game screen.

        Args:
            frame: The webcam frame to display
            body_mask: The segmentation mask for visualization and debugging
        """
        # Use the class variable for body mask visibility
        # Convert webcam frame to pygame surface and scale to full screen
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
        frame_surface = pygame.transform.scale(frame_surface, (self.width, self.height))

        # Draw webcam frame as background
        self.screen.blit(frame_surface, (0, 0))

        # Draw body mask for debugging if available
        if self.show_body_mask and body_mask is not None:
            # Create a small preview of the body mask in the corner
            mask_size = (int(self.width * 0.2), int(self.height * 0.2))
            # Resize the mask to the preview size
            body_mask_small = cv2.resize(body_mask, mask_size)
            # Convert to RGB for visualization (white silhouette on black background)
            body_mask_rgb = np.stack([body_mask_small] * 3, axis=2)
            # Create pygame surface
            mask_surface = pygame.surfarray.make_surface(body_mask_rgb.swapaxes(0, 1))
            # Draw in bottom-right corner
            self.screen.blit(mask_surface, (self.width - mask_size[0] - 10, self.height - mask_size[1] - 10))

        # Draw background with transparency if enabled
        if self.show_background:
            # Create a copy of the background to avoid modifying the original
            background_copy = self.background.copy()
            # Apply the current opacity setting
            background_copy.set_alpha(self.background_opacity)
            self.screen.blit(background_copy, (0, 0))

            # Draw a help text for background controls
            font = pygame.font.Font(None, 24)
            help_text = font.render("Pressione B para alternar fundo, + e - para ajustar opacidade", True, (255, 255, 255))
            self.screen.blit(help_text, (10, self.height - 30))

        # Draw shape with progress overlay
        shape_overlay = self.shape_img.copy()

        # Create a progress overlay based on current percentage and excesso
        if not self.victory:
            # Informações extras para debug
            font = pygame.font.Font(None, 36)
            percentage_text = font.render(f"Preenchimento: {self.current_percentage:.1f}%", True, (255, 255, 255))
            self.screen.blit(percentage_text, (20, 20))
            
            excesso_text = font.render(f"Excesso: {self.excesso_percentual:.1f}%", True, (255, 100, 100))
            self.screen.blit(excesso_text, (20, 60))
            
            # Mostrar informações de debug adicionais
            if hasattr(self.vision, 'overlap_area') and hasattr(self.vision, 'total_shape_area'):
                debug_text = font.render(
                    f"Área sobreposta: {self.vision.overlap_area} / {self.vision.total_shape_area} pixels", 
                    True, (255, 255, 100)
                )
                self.screen.blit(debug_text, (20, 100))
            
            # Redimensionar a máscara do corpo para o tamanho da forma
            if body_mask is not None:
                # Criar uma cópia do contorno para manipulação
                contorno = self.contorno_img.copy()
                
                # Redimensionar a máscara do corpo
                body_mask_resized = cv2.resize(body_mask, (shape_overlay.get_width(), shape_overlay.get_height()))
                
                # Aplicar operações morfológicas para melhorar a visualização
                kernel = np.ones((7, 7), np.uint8)
                body_mask_dilated = cv2.dilate(body_mask_resized, kernel, iterations=3)
                
                # Criar máscaras para partes dentro e fora do contorno
                shape_mask = pygame.surfarray.array_alpha(shape_overlay)
                body_mask_np = body_mask_dilated > 0

                # Método alternativo 
                # Criar uma superfície para visualização da área coberta (verde)
                green_surface = pygame.Surface(shape_overlay.get_size(), pygame.SRCALPHA)
                green_surface.fill((0, 0, 0, 0))  # Inicialmente transparente
                
                # Desenhar corpo como verde brilhante
                for y in range(body_mask_np.shape[0]):
                    for x in range(body_mask_np.shape[1]):
                        if body_mask_np[y, x] and x < shape_mask.shape[1] and y < shape_mask.shape[0] and shape_mask[y, x] > 0:
                            # Dentro do contorno - verde
                            pygame.draw.circle(green_surface, (0, 255, 0, 200), (x, y), 3)  # Círculos verdes

                # Desenhar excesso como vermelho 
                red_surface = pygame.Surface(shape_overlay.get_size(), pygame.SRCALPHA)
                red_surface.fill((0, 0, 0, 0))  # Inicialmente transparente
                
                for y in range(body_mask_np.shape[0]):
                    for x in range(body_mask_np.shape[1]):
                        if body_mask_np[y, x] and (x >= shape_mask.shape[1] or y >= shape_mask.shape[0] or shape_mask[y, x] == 0):
                            # Fora do contorno - vermelho
                            pygame.draw.circle(red_surface, (255, 0, 0, 200), (x, y), 3)  # Círculos vermelhos
                
                # Aplicar as superfícies coloridas
                self.screen.blit(green_surface, self.shape_pos)
                self.screen.blit(red_surface, self.shape_pos)
                
        # Draw shape
        self.screen.blit(shape_overlay, self.shape_pos)

        # Draw contorno (outline) on top of everything with maximum contrast
        # Make sure the contorno is visible by setting a high contrast color
        contorno_overlay = self.contorno_img.copy()

        # Apply a color filter to make it more visible (bright red with maximum opacity)
        color_surface = pygame.Surface(contorno_overlay.get_size(), pygame.SRCALPHA)
        color_surface.fill((255, 0, 0, 255))  # Bright red with maximum opacity
        contorno_overlay.blit(color_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

        # Draw the contorno with a slight offset to create a shadow effect
        shadow_offset = 2
        self.screen.blit(contorno_overlay, (self.shape_pos[0] + shadow_offset, self.shape_pos[1] + shadow_offset))

        # Draw the main contorno
        color_surface.fill((255, 255, 255, 255))  # Pure white with maximum opacity
        contorno_overlay = self.contorno_img.copy()
        contorno_overlay.blit(color_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        self.screen.blit(contorno_overlay, self.shape_pos)

        # Draw percentage and timer
        font = pygame.font.Font(None, 36)
        
        # Draw timer if it's running
        if self.timer_running and self.start_time is not None:
            # Calculate current time
            current_time = time.time() - self.start_time + self.elapsed_time
            # Format time as minutes:seconds
            minutes = int(current_time // 60)
            seconds = int(current_time % 60)
            timer_text = font.render(f"Tempo: {minutes:02d}:{seconds:02d}", True, (255, 255, 255))
            self.screen.blit(timer_text, (self.width - timer_text.get_width() - 20, 20))

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
                    # Start the timer when the game begins
                    self.start_time = time.time()
                    self.elapsed_time = 0
                    self.timer_running = True
                elif not self.in_menu:  # Only in game mode
                    # Toggle background visibility with B key
                    if event.key == pygame.K_b:
                        self.show_background = not self.show_background
                        print(f"Background visibility: {self.show_background}")

                    # Increase background opacity with + key
                    elif event.key == pygame.K_PLUS or event.key == pygame.K_KP_PLUS or event.key == pygame.K_EQUALS:
                        self.background_opacity = min(255, self.background_opacity + 15)
                        print(f"Background opacity: {self.background_opacity}")

                    # Decrease background opacity with - key
                    elif event.key == pygame.K_MINUS or event.key == pygame.K_KP_MINUS:
                        self.background_opacity = max(0, self.background_opacity - 15)
                        print(f"Background opacity: {self.background_opacity}")

                    # Pause/resume timer with T key
                    elif event.key == pygame.K_t:
                        if self.timer_running:
                            # Pause timer
                            self.elapsed_time += time.time() - self.start_time
                            self.timer_running = False
                            print("Timer paused")
                        else:
                            # Resume timer
                            self.start_time = time.time()
                            self.timer_running = True
                            print("Timer resumed")

                    # Reset timer with R key
                    elif event.key == pygame.K_r:
                        self.start_time = time.time()
                        self.elapsed_time = 0
                        self.timer_running = True
                        print("Timer reset")

                    # Toggle body mask visualization with M key
                    elif event.key == pygame.K_m:
                        self.show_body_mask = not self.show_body_mask
                        print(f"Body mask visualization: {self.show_body_mask}")

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
