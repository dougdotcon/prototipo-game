#!/usr/bin/env python3
"""
View do jogo SHAPE-SE.
Este módulo é responsável pela renderização da interface do jogo.
"""

import pygame
import cv2
import numpy as np
from typing import Optional, Tuple

class ShapeSEView:
    """Classe que representa a view do jogo SHAPE-SE."""

    def __init__(self, screen: pygame.Surface):
        """
        Inicializa a view do jogo.
        
        Args:
            screen: Superfície do Pygame onde o jogo será renderizado
        """
        self.screen = screen
        self.width = screen.get_width()
        self.height = screen.get_height()
        
        # Define cores
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        
        # Carrega fontes
        pygame.font.init()
        self.timer_font = pygame.font.Font(None, 74)
        self.message_font = pygame.font.Font(None, 100)
        
        # Posição do cronômetro (canto superior direito)
        self.timer_pos = (self.width - 150, 20)
        
        # Carrega e redimensiona a bandeira
        self.load_flag()
        
        # Carrega e redimensiona o contorno do mapa
        self.load_map_contour()

    def load_flag(self):
        """Carrega e configura a bandeira de Sergipe."""
        try:
            flag = cv2.imread("novo-game/assets/bandeira.png", cv2.IMREAD_UNCHANGED)
            if flag is None:
                raise FileNotFoundError("Não foi possível carregar a bandeira")
            
            # Redimensiona a bandeira mantendo a proporção
            flag_height = self.height // 4
            aspect_ratio = flag.shape[1] / flag.shape[0]
            flag_width = int(flag_height * aspect_ratio)
            
            self.flag = cv2.resize(flag, (flag_width, flag_height))
            
            # Converte para formato do Pygame
            self.flag_surface = self._convert_rgba_to_pygame_surface(self.flag)
            
            # Posiciona no canto superior esquerdo
            self.flag_pos = (20, 20)
            
        except Exception as e:
            print(f"Erro ao carregar a bandeira: {e}")
            self.flag_surface = None

    def load_map_contour(self):
        """Carrega e configura o contorno do mapa."""
        try:
            contour = cv2.imread("novo-game/assets/contorno-mapa-SE.png")
            if contour is None:
                raise FileNotFoundError("Não foi possível carregar o contorno do mapa")
            
            # Aumenta a espessura do contorno
            kernel = np.ones((5,5), np.uint8)
            contour = cv2.dilate(contour, kernel, iterations=2)
            
            # Redimensiona o contorno
            contour_height = self.height // 2
            aspect_ratio = contour.shape[1] / contour.shape[0]
            contour_width = int(contour_height * aspect_ratio)
            
            self.map_contour = cv2.resize(contour, (contour_width, contour_height))
            
            # Converte para formato do Pygame
            self.map_surface = self._convert_bgr_to_pygame_surface(self.map_contour)
            
            # Posiciona no centro da tela
            self.map_pos = ((self.width - contour_width) // 2,
                           (self.height - contour_height) // 2)
            
        except Exception as e:
            print(f"Erro ao carregar o contorno do mapa: {e}")
            self.map_surface = None

    def _convert_rgba_to_pygame_surface(self, img: np.ndarray) -> pygame.Surface:
        """
        Converte uma imagem RGBA do OpenCV para uma superfície Pygame.
        
        Args:
            img: Imagem no formato RGBA do OpenCV
            
        Returns:
            pygame.Surface: Superfície Pygame correspondente
        """
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        return pygame.image.frombuffer(img.tobytes(), img.shape[1::-1], "RGBA")

    def _convert_bgr_to_pygame_surface(self, img: np.ndarray) -> pygame.Surface:
        """
        Converte uma imagem BGR do OpenCV para uma superfície Pygame.
        
        Args:
            img: Imagem no formato BGR do OpenCV
            
        Returns:
            pygame.Surface: Superfície Pygame correspondente
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return pygame.image.frombuffer(img.tobytes(), img.shape[1::-1], "RGB")

    def render_camera_frame(self, frame: np.ndarray):
        """
        Renderiza o frame da câmera na tela.
        
        Args:
            frame: Frame da câmera no formato BGR
        """
        if frame is not None:
            # Converte o frame para formato Pygame
            frame_surface = self._convert_bgr_to_pygame_surface(frame)
            
            # Renderiza o frame em tela cheia
            self.screen.blit(frame_surface, (0, 0))

    def render_game_elements(self, remaining_time: str):
        """
        Renderiza os elementos do jogo (bandeira, mapa, cronômetro).
        
        Args:
            remaining_time: Tempo restante formatado (MM:SS)
        """
        # Renderiza a bandeira (se carregada)
        if self.flag_surface is not None:
            self.screen.blit(self.flag_surface, self.flag_pos)
        
        # Renderiza o contorno do mapa (se carregado)
        if self.map_surface is not None:
            self.screen.blit(self.map_surface, self.map_pos)
        
        # Renderiza o cronômetro
        timer_surface = self.timer_font.render(remaining_time, True, self.WHITE)
        self.screen.blit(timer_surface, self.timer_pos)

    def render_game_over(self, is_winner: bool, captured_image: Optional[np.ndarray] = None):
        """
        Renderiza a tela de fim de jogo.
        
        Args:
            is_winner: Se o jogador venceu o jogo
            captured_image: Imagem capturada do jogador (opcional)
        """
        # Escurece a tela
        overlay = pygame.Surface((self.width, self.height))
        overlay.fill(self.BLACK)
        overlay.set_alpha(128)
        self.screen.blit(overlay, (0, 0))
        
        # Renderiza a mensagem apropriada
        message = "Viva Sergipe!" if is_winner else "Game Over"
        color = self.GREEN if is_winner else self.RED
        
        message_surface = self.message_font.render(message, True, color)
        message_rect = message_surface.get_rect(center=(self.width//2, self.height//2))
        self.screen.blit(message_surface, message_rect)
        
        # Se o jogador venceu e temos uma imagem capturada, mostra-a
        if is_winner and captured_image is not None:
            # Converte e redimensiona a imagem capturada
            img_height = self.height // 3
            aspect_ratio = captured_image.shape[1] / captured_image.shape[0]
            img_width = int(img_height * aspect_ratio)
            
            resized_image = cv2.resize(captured_image, (img_width, img_height))
            image_surface = self._convert_bgr_to_pygame_surface(resized_image)
            
            # Posiciona a imagem abaixo da mensagem
            image_pos = ((self.width - img_width) // 2,
                        message_rect.bottom + 20)
            self.screen.blit(image_surface, image_pos)