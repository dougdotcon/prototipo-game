#!/usr/bin/env python3
"""
Script para processar as imagens do jogo SHAPE-SE.
Este script modifica a bandeira para ficar mais transparente e o contorno do mapa para ficar mais visível.
"""

import cv2
import numpy as np
import os

def ensure_directory_exists(path):
    """Cria o diretório se ele não existir."""
    if not os.path.exists(path):
        os.makedirs(path)

def process_flag(input_path, output_path):
    """
    Processa a bandeira de Sergipe para torná-la mais transparente.
    
    Args:
        input_path: Caminho da imagem original
        output_path: Caminho onde a imagem processada será salva
    """
    # Lê a imagem original
    flag = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if flag is None:
        raise FileNotFoundError(f"Não foi possível carregar a imagem: {input_path}")
    
    # Se a imagem não tem canal alpha, adiciona um
    if flag.shape[2] == 3:
        flag = cv2.cvtColor(flag, cv2.COLOR_BGR2BGRA)
    
    # Reduz a opacidade para 50%
    flag[:, :, 3] = (flag[:, :, 3] * 0.5).astype(np.uint8)
    
    # Salva a imagem processada
    cv2.imwrite(output_path, flag)
    print(f"Bandeira processada salva em: {output_path}")

def process_map_contour(input_path, output_path):
    """
    Processa o contorno do mapa para torná-lo mais visível.
    
    Args:
        input_path: Caminho da imagem original
        output_path: Caminho onde a imagem processada será salva
    """
    # Lê a imagem original
    contour = cv2.imread(input_path)
    if contour is None:
        raise FileNotFoundError(f"Não foi possível carregar a imagem: {input_path}")
    
    # Converte para escala de cinza
    gray = cv2.cvtColor(contour, cv2.COLOR_BGR2GRAY)
    
    # Aplica threshold para binarizar a imagem
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Encontra os contornos
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Cria uma nova imagem preta
    result = np.zeros_like(contour)
    
    # Desenha os contornos com maior espessura
    cv2.drawContours(result, contours, -1, (255, 255, 255), 3)
    
    # Aplica um leve blur para suavizar as bordas
    result = cv2.GaussianBlur(result, (3, 3), 0)
    
    # Salva a imagem processada
    cv2.imwrite(output_path, result)
    print(f"Contorno do mapa processado salvo em: {output_path}")

def main():
    """Função principal que processa as imagens."""
    # Define os caminhos
    input_dir = "shape-se"
    output_dir = "novo-game/assets"
    
    # Garante que o diretório de saída existe
    ensure_directory_exists(output_dir)
    
    try:
        # Processa a bandeira
        process_flag(
            os.path.join(input_dir, "bandeira.png"),
            os.path.join(output_dir, "bandeira.png")
        )
        
        # Processa o contorno do mapa
        process_map_contour(
            os.path.join(input_dir, "contorno-mapa-SE.png"),
            os.path.join(output_dir, "contorno-mapa-SE.png")
        )
        
        print("Processamento de imagens concluído com sucesso!")
        
    except Exception as e:
        print(f"Erro durante o processamento das imagens: {e}")

if __name__ == "__main__":
    main()