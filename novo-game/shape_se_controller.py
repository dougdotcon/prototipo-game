#!/usr/bin/env python3
"""
Controller do jogo SHAPE-SE.
Este módulo é responsável por gerenciar as interações do usuário e coordenar o modelo e a view.
"""

import pygame
import cv2
from shape_se_model import ShapeSEModel
from shape_se_view import ShapeSEView

class ShapeSEController:
    """Classe que representa o controller do jogo SHAPE-SE."""

    def __init__(self, model: ShapeSEModel, view: ShapeSEView, camera: cv2.VideoCapture):
        """
        Inicializa o controller do jogo.
        
        Args:
            model: Instância do modelo do jogo
            view: Instância da view do jogo
            camera: Instância da câmera
        """
        self.model = model
        self.view = view
        self.camera = camera
        
        # Flag para controlar o estado do jogo
        self.running = True
        
        # Define as teclas de controle
        self.QUIT_KEY = pygame.K_q
        self.START_KEY = pygame.K_SPACE

    def handle_events(self) -> bool:
        """
        Processa os eventos do Pygame.
        
        Returns:
            bool: True se o jogo deve continuar, False se deve terminar
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return False
                
            if event.type == pygame.KEYDOWN:
                if event.key == self.QUIT_KEY:
                    self.running = False
                    return False
                    
                if event.key == self.START_KEY and not self.model.game_started:
                    self.model.start_game()
        
        return True

    def update(self) -> bool:
        """
        Atualiza o estado do jogo.
        
        Returns:
            bool: True se o jogo deve continuar, False se deve terminar
        """
        if not self.running:
            return False
            
        # Processa eventos
        if not self.handle_events():
            return False
            
        # Captura frame da câmera
        ret, frame = self.camera.read()
        if not ret:
            print("Erro ao capturar frame da câmera")
            return False
            
        # Espelha o frame horizontalmente para criar efeito de espelho
        frame = cv2.flip(frame, 1)
        
        if self.model.game_started and not self.model.game_over:
            # Calcula pontuação
            score = self.model.calculate_score(frame)
            self.model.current_score = score
            
            # Verifica se chegou ao fim do jogo
            if not self.model.update():
                # Captura a imagem final do jogador se necessário
                if self.model.is_winner():
                    self.model.capture_player_image(frame)
                return True
        
        return True

    def render(self):
        """Renderiza o frame atual do jogo."""
        # Captura frame da câmera
        ret, frame = self.camera.read()
        if not ret:
            return
            
        # Espelha o frame horizontalmente
        frame = cv2.flip(frame, 1)
        
        # Renderiza o frame da câmera
        self.view.render_camera_frame(frame)
        
        if self.model.game_started:
            # Renderiza elementos do jogo
            self.view.render_game_elements(self.model.get_remaining_time_str())
            
            if self.model.game_over:
                # Renderiza tela de fim de jogo
                self.view.render_game_over(
                    self.model.is_winner(),
                    self.model.captured_image
                )
        else:
            # Renderiza mensagem inicial
            font = pygame.font.Font(None, 74)
            text = font.render("Pressione ESPAÇO para iniciar", True, (255, 255, 255))
            text_rect = text.get_rect(center=(self.view.width//2, self.view.height//2))
            self.view.screen.blit(text, text_rect)
            
            # Renderiza instruções adicionais
            font_small = pygame.font.Font(None, 36)
            quit_text = font_small.render("Pressione Q para sair", True, (255, 255, 255))
            quit_rect = quit_text.get_rect(center=(self.view.width//2, self.view.height//2 + 50))
            self.view.screen.blit(quit_text, quit_rect)