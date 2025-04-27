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
        self.excesso_percentual = 0  # Inicializar excesso
        self.victory_score = 0  # Pontuação final
        self.victory_animation_frames = []  # Frames para animação de vitória
        self.current_animation_frame = 0  # Frame atual da animação
        self.victory_stars = []  # Para animação de estrelas

        # Timer variables
        self.start_time = None
        self.elapsed_time = 0
        self.timer_running = False

        # Debug options
        self.show_body_mask = True  # Show body segmentation mask for debugging

        # Create snapshots directory if it doesn't exist
        if not os.path.exists("snapshots"):
            os.makedirs("snapshots")

        self.contorno_pulse = 0  # Para animação pulsante do contorno

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
        
        # Obter a cobertura interna (quanto do shape está preenchido de verde)
        self.cobertura_interna = getattr(self.vision, 'cobertura_interna', 0)
        
        # Verificar condições de vitória: 
        # Novo critério: Focar na cobertura interna (verde) para vencer, 
        # com uma tolerância maior para o excesso
        if (self.cobertura_interna >= self.threshold and 
            self.excesso_percentual < 50 and  # Tolerância maior para excesso
            not self.victory):
            self.victory = True
            self.victory_time = time.time()
            self.save_snapshot(frame)
            
            # Calcular pontuação com base no tempo e precisão
            tempo_segundos = self.elapsed_time + (time.time() - self.start_time) if self.start_time else 0
            precisao = self.cobertura_interna / (1 + self.excesso_percentual/100)  # Menos penalidade para excesso
            self.victory_score = int(10000 * (self.cobertura_interna/100) * max(0, 1 - tempo_segundos/60))
            
            # Inicializar animação de vitória
            self.initialize_victory_animation()

        # Reset victory after 10 seconds (mais tempo para ler a mensagem)
        if self.victory and time.time() - self.victory_time > 10:
            self.victory = False
            self.in_menu = True  # Voltar para o menu

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
        
        # Armazenar o caminho para exibir na tela de vitória
        self.victory_snapshot = filename

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
        """Render the game screen."""
        # Convert webcam frame to pygame surface and scale to full screen
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
        frame_surface = pygame.transform.scale(frame_surface, (self.width, self.height))

        # Draw webcam frame as background
        self.screen.blit(frame_surface, (0, 0))

        # Draw body mask for debugging if available (desativado)
        if self.show_body_mask and body_mask is not None and False:
            mask_size = (int(self.width * 0.2), int(self.height * 0.2))
            body_mask_small = cv2.resize(body_mask, mask_size)
            body_mask_rgb = np.stack([body_mask_small] * 3, axis=2)
            mask_surface = pygame.surfarray.make_surface(body_mask_rgb.swapaxes(0, 1))
            self.screen.blit(mask_surface, (self.width - mask_size[0] - 10, self.height - mask_size[1] - 10))

        # Draw background with transparency if enabled
        if self.show_background:
            background_copy = self.background.copy()
            background_copy.set_alpha(self.background_opacity)
            self.screen.blit(background_copy, (0, 0))

            font = pygame.font.Font(None, 24)
            help_text = font.render("Pressione B para alternar fundo, + e - para ajustar opacidade", True, (255, 255, 255))
            self.screen.blit(help_text, (10, self.height - 30))

        # Informações na tela
        font = pygame.font.Font(None, 36)
        percentage_text = font.render(f"Preenchimento: {self.current_percentage:.1f}%", True, (255, 255, 255))
        self.screen.blit(percentage_text, (20, 20))
        
        excesso_text = font.render(f"Excesso: {self.excesso_percentual:.1f}%", True, (255, 100, 100))
        self.screen.blit(excesso_text, (20, 60))
        
        cobertura_text = font.render(f"Cobertura interna: {self.cobertura_interna:.1f}%", True, (100, 255, 100))
        self.screen.blit(cobertura_text, (20, 100))

        # Processar a visualização do preenchimento
        if not self.victory and body_mask is not None:
            # Criar uma superfície transparente para o quadrado do mapa
            shape_surface = pygame.Surface((self.shape_img.get_width(), self.shape_img.get_height()), pygame.SRCALPHA)
            # Inicialmente transparente
            shape_surface.fill((0, 0, 0, 0))
            
            # Obter a máscara da forma e do corpo
            body_mask_resized = cv2.resize(body_mask, (self.shape_img.get_width(), self.shape_img.get_height()))
            body_mask_processed = cv2.dilate(body_mask_resized, np.ones((7, 7), np.uint8), iterations=3)
            body_bool = body_mask_processed > 0
            
            # Usar o contorno como máscara para a forma
            contorno_surface = pygame.Surface(self.contorno_img.get_size(), pygame.SRCALPHA)
            contorno_surface.fill((0, 0, 0, 0))  # Inicializar transparente
            contorno_surface.blit(self.shape_img, (0, 0))  # Usar shape_img em vez do contorno
            shape_array = pygame.surfarray.array_alpha(contorno_surface)
            shape_bool = shape_array > 10  # Limiar para detectar a forma
            
            # Calcular interseção (verde) e excesso (vermelho)
            inside_mask = np.logical_and(body_bool, shape_bool)
            outside_mask = np.logical_and(body_bool, np.logical_not(shape_bool))
            
            # Criar uma única superfície de visualização
            visual_surface = pygame.Surface(shape_surface.get_size(), pygame.SRCALPHA)
            
            # Preencher áreas - FUNDO CINZA MUITO TRANSPARENTE
            for y in range(visual_surface.get_height()):
                for x in range(visual_surface.get_width()):
                    if y < shape_bool.shape[0] and x < shape_bool.shape[1]:
                        # Dentro da forma não preenchida - cinza transparente
                        if shape_bool[y, x] and not inside_mask[y, x]:
                            visual_surface.set_at((x, y), (150, 150, 150, 30))
                        # Dentro da forma E preenchida - verde
                        elif inside_mask[y, x]:
                            visual_surface.set_at((x, y), (0, 255, 0, 180))
                        # Fora da forma mas com corpo - vermelho
                        elif outside_mask[y, x]:
                            visual_surface.set_at((x, y), (255, 0, 0, 180))
            
            # Colocar na tela
            self.screen.blit(visual_surface, self.shape_pos)
            
            # Desenhar apenas UM contorno preto por cima de tudo
            pygame.draw.rect(self.screen, (0, 0, 0, 0), 
                            (self.shape_pos[0], self.shape_pos[1], 
                            visual_surface.get_width(), visual_surface.get_height()), 1)
            
            # Desenhar o contorno usando a imagem original
            contorno_final = self.contorno_img.copy()
            # Colorir o contorno de PRETO (não branco, não cinza)
            for y in range(contorno_final.get_height()):
                for x in range(contorno_final.get_width()):
                    if contorno_final.get_at((x, y))[3] > 50:  # Se não for transparente
                        contorno_final.set_at((x, y), (0, 0, 0, 255))  # Preto sólido
            
            # Colocar o contorno na tela
            self.screen.blit(contorno_final, self.shape_pos)
            
            # Mostrar instrução de acordo com o progresso
            if self.cobertura_interna < self.threshold:
                instrucao = font.render("Complete de verde o contorno para vencer!", True, (0, 255, 50))
                self.screen.blit(instrucao, (self.width//2 - instrucao.get_width()//2, self.height - 50))

        # Draw timer if it's running
        if self.timer_running and self.start_time is not None:
            # Calculate current time
            current_time = time.time() - self.start_time + self.elapsed_time
            # Format time as minutes:seconds
            minutes = int(current_time // 60)
            seconds = int(current_time % 60)
            font = pygame.font.Font(None, 36)
            timer_text = font.render(f"Tempo: {minutes:02d}:{seconds:02d}", True, (255, 255, 255))
            self.screen.blit(timer_text, (self.width - timer_text.get_width() - 20, 20))

        # Draw victory message
        if self.victory:
            self.render_victory_screen()

        pygame.display.flip()

    def render_victory_screen(self):
        """Renderiza a tela de vitória com animação."""
        # Criar um overlay semi-transparente
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))  # Preto semi-transparente
        self.screen.blit(overlay, (0, 0))
        
        # Atualizar e desenhar as estrelas
        for star in self.victory_stars:
            star['y'] = (star['y'] + star['speed']) % self.height
            pygame.draw.circle(self.screen, star['color'], (star['x'], star['y']), star['size'])
        
        # Mensagens de vitória
        font_grande = pygame.font.Font(None, 96)
        font_medio = pygame.font.Font(None, 64)
        font_pequeno = pygame.font.Font(None, 48)
        
        # Texto "VOCÊ VENCEU!"
        victory_text = font_grande.render("VOCÊ VENCEU!", True, (255, 255, 0))
        text_rect = victory_text.get_rect(center=(self.width // 2, self.height // 6))
        # Adicionar sombra
        shadow_text = font_grande.render("VOCÊ VENCEU!", True, (150, 100, 0))
        shadow_rect = shadow_text.get_rect(center=(self.width // 2 + 5, self.height // 6 + 5))
        self.screen.blit(shadow_text, shadow_rect)
        self.screen.blit(victory_text, text_rect)
        
        # Exibir o snapshot do momento da vitória se disponível
        if hasattr(self, 'victory_snapshot') and os.path.exists(self.victory_snapshot):
            try:
                # Carregar a imagem
                snapshot_img = pygame.image.load(self.victory_snapshot)
                # Redimensionar para um tamanho razoável
                max_height = self.height // 3
                width_ratio = max_height / snapshot_img.get_height()
                new_size = (int(snapshot_img.get_width() * width_ratio), max_height)
                snapshot_img = pygame.transform.scale(snapshot_img, new_size)
                # Determinar a posição (centralizado horizontalmente, abaixo do título)
                snapshot_rect = snapshot_img.get_rect(center=(self.width // 2, self.height * 0.35))
                # Desenhar com uma borda
                pygame.draw.rect(self.screen, (255, 255, 255), 
                                 pygame.Rect(snapshot_rect.x - 5, snapshot_rect.y - 5, 
                                             snapshot_rect.width + 10, snapshot_rect.height + 10))
                self.screen.blit(snapshot_img, snapshot_rect)
            except Exception as e:
                print(f"Erro ao mostrar snapshot: {e}")
        
        # Texto de pontuação
        score_text = font_medio.render(f"Pontuação: {self.victory_score}", True, (200, 255, 200))
        score_rect = score_text.get_rect(center=(self.width // 2, self.height * 0.6))
        self.screen.blit(score_text, score_rect)
        
        # Detalhes da pontuação
        details_text = font_pequeno.render(
            f"Cobertura verde: {self.cobertura_interna:.1f}%", 
            True, (180, 255, 180)
        )
        details_rect = details_text.get_rect(center=(self.width // 2, self.height * 0.67))
        self.screen.blit(details_text, details_rect)
        
        # Tempo de conclusão
        if self.start_time:
            tempo_total = self.elapsed_time + (self.victory_time - self.start_time)
            minutos = int(tempo_total // 60)
            segundos = int(tempo_total % 60)
            tempo_text = font_pequeno.render(
                f"Tempo: {minutos:02d}:{segundos:02d}", 
                True, (180, 180, 255)
            )
            tempo_rect = tempo_text.get_rect(center=(self.width // 2, self.height * 0.74))
            self.screen.blit(tempo_text, tempo_rect)
        
        # Instrução para continuar
        # Pulsar texto para chamar atenção
        alpha = int(128 + 127 * np.sin(time.time() * 4))
        continue_text = font_pequeno.render("Pressione ESPAÇO para jogar novamente", True, (255, 255, 255))
        continue_text.set_alpha(alpha)
        continue_rect = continue_text.get_rect(center=(self.width // 2, self.height * 0.85))
        self.screen.blit(continue_text, continue_rect)

    def initialize_victory_animation(self):
        """Inicializa a animação de vitória com estrelas."""
        # Criar estrelas para animação
        self.victory_stars = []
        for _ in range(50):
            star = {
                'x': np.random.randint(0, self.width),
                'y': np.random.randint(0, self.height),
                'size': np.random.randint(2, 8),
                'speed': np.random.randint(2, 7),
                'color': [
                    np.random.randint(150, 255),
                    np.random.randint(150, 255), 
                    np.random.randint(150, 255)
                ]
            }
            self.victory_stars.append(star)

    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    if self.in_menu:
                        self.in_menu = False  # Exit menu and start game
                        # Start the timer when the game begins
                        self.start_time = time.time()
                        self.elapsed_time = 0
                        self.timer_running = True
                    elif self.victory:
                        # Reiniciar o jogo em caso de vitória
                        self.victory = False
                        self.in_menu = True
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
