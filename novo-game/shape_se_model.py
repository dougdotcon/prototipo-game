#!/usr/bin/env python3
"""
Modelo do jogo SHAPE-SE.
Este módulo contém a lógica principal do jogo.
"""

import time
import cv2
import numpy as np
from typing import Tuple, Optional

class ShapeSEModel:
    """Classe que representa o modelo do jogo SHAPE-SE."""

    def __init__(self):
        """Inicializa o modelo do jogo."""
        # Tempo total do jogo em segundos (5 minutos)
        self.TOTAL_TIME = 5 * 60
        
        # Estado do jogo
        self.game_started = False
        self.game_over = False
        self.start_time = None
        self.remaining_time = self.TOTAL_TIME
        self.current_score = 0.0
        self.captured_image = None
        
        # Carrega os recursos do jogo
        self.load_game_resources()

    def load_game_resources(self):
        """Carrega os recursos necessários para o jogo."""
        try:
            # Carrega a bandeira de Sergipe com transparência
            self.flag = cv2.imread("novo-game/assets/bandeira.png", cv2.IMREAD_UNCHANGED)
            if self.flag is not None and self.flag.shape[2] == 4:  # Verifica se tem canal alpha
                # Ajusta a transparência (multiplica o canal alpha por 0.5)
                self.flag[:, :, 3] = (self.flag[:, :, 3] * 0.5).astype(np.uint8)
            
            # Carrega o contorno do mapa
            self.map_contour = cv2.imread("novo-game/assets/contorno-mapa-SE.png")
            
            # Define a porcentagem mínima para vitória
            self.WIN_THRESHOLD = 95.0
            
        except Exception as e:
            print(f"Erro ao carregar recursos: {e}")
            raise

    def start_game(self):
        """Inicia o jogo."""
        self.game_started = True
        self.game_over = False
        self.start_time = time.time()
        self.remaining_time = self.TOTAL_TIME
        self.current_score = 0.0
        self.captured_image = None

    def update(self) -> bool:
        """
        Atualiza o estado do jogo.
        
        Returns:
            bool: True se o jogo deve continuar, False se deve terminar
        """
        if not self.game_started:
            return True
            
        # Atualiza o tempo restante
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        self.remaining_time = max(0, self.TOTAL_TIME - int(elapsed_time))
        
        # Verifica se o tempo acabou
        if self.remaining_time <= 0:
            self.end_game()
            return False
            
        return True

    def calculate_score(self, frame: np.ndarray) -> float:
        """
        Calcula a pontuação atual do jogador.
        
        Args:
            frame: Frame atual da câmera
            
        Returns:
            float: Pontuação entre 0 e 100
        """
        # TODO: Implementar lógica de cálculo da pontuação
        # Por enquanto retorna uma pontuação de exemplo
        return 96.0

    def capture_player_image(self, frame: np.ndarray):
        """
        Captura a imagem do jogador.
        
        Args:
            frame: Frame atual da câmera
        """
        if frame is not None:
            self.captured_image = frame.copy()

    def end_game(self):
        """Finaliza o jogo e determina o resultado."""
        self.game_over = True
        self.game_started = False

    def is_winner(self) -> bool:
        """
        Verifica se o jogador venceu o jogo.
        
        Returns:
            bool: True se o jogador venceu, False caso contrário
        """
        return self.current_score >= self.WIN_THRESHOLD

    def get_remaining_time_str(self) -> str:
        """
        Obtém o tempo restante formatado.
        
        Returns:
            str: Tempo restante no formato MM:SS
        """
        minutes = self.remaining_time // 60
        seconds = self.remaining_time % 60
        return f"{minutes:02d}:{seconds:02d}"

    def get_game_state(self) -> Tuple[bool, bool, float, Optional[np.ndarray]]:
        """
        Obtém o estado atual do jogo.
        
        Returns:
            Tuple contendo:
            - bool: Se o jogo está em andamento
            - bool: Se o jogo acabou
            - float: Pontuação atual
            - Optional[np.ndarray]: Imagem capturada do jogador (se houver)
        """
        return (self.game_started, self.game_over, 
                self.current_score, self.captured_image)