#!/usr/bin/env python3
"""
Runner do jogo SHAPE-SE.
Este módulo é responsável por inicializar e executar o jogo.
"""

import pygame
import cv2
from shape_se_model import ShapeSEModel
from shape_se_view import ShapeSEView
from shape_se_controller import ShapeSEController

# Configurações globais
CAMERA_INDEX = 0  # Índice da câmera a ser usada
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
FPS = 60

def main():
    """
    Função principal que inicializa e executa o jogo.
    """
    # Inicializa o Pygame
    pygame.init()
    pygame.display.set_caption("SHAPE-SE")
    
    # Inicializa a câmera
    camera = cv2.VideoCapture(CAMERA_INDEX)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)
    
    # Cria a janela do jogo
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()
    
    try:
        # Inicializa os componentes MVC
        model = ShapeSEModel()
        view = ShapeSEView(screen)
        controller = ShapeSEController(model, view, camera)
        
        # Loop principal do jogo
        while True:
            # Controla o FPS
            clock.tick(FPS)
            
            # Atualiza o jogo
            if not controller.update():
                break
            
            # Renderiza o frame atual
            controller.render()
            
            # Atualiza a tela
            pygame.display.flip()
            
    except Exception as e:
        print(f"Erro durante a execução do jogo: {e}")
        
    finally:
        # Limpa recursos
        camera.release()
        pygame.quit()

if __name__ == "__main__":
    main()