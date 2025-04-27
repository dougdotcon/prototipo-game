Resumo do Projeto Shape SE
O Shape SE é um minigame de reconhecimento corporal onde o jogador deve posicionar seu corpo para preencher o contorno do mapa do estado de Sergipe. O jogo utiliza a webcam para capturar a imagem do jogador e, através de segmentação corporal com MediaPipe, verifica se o jogador está preenchendo adequadamente o contorno do estado.

Principais características:
Reconhecimento corporal em tempo real usando MediaPipe SelfieSeg
Detecção de preenchimento da silhueta do estado de Sergipe
Feedback visual com overlay verde que indica o progresso
Validação de acerto quando o jogador preenche 95% ou mais da área
Captura de imagem do momento da vitória
Interface fullscreen com a bandeira de Sergipe como fundo
Como jogar:
Posicione a câmera a aproximadamente 2 metros de distância
Alinhe seu corpo com o contorno do estado de Sergipe mostrado na tela
Tente preencher o contorno até atingir 95% ou mais
Quando conseguir, uma mensagem de "Parabéns!" será exibida e uma foto será salva
Pressione ESC para sair do jogo
Estrutura do projeto:
core/vision.py: Gerencia a captura de vídeo e segmentação corporal
core/game.py: Implementa a lógica principal do jogo e interface gráfica
run.py: Script principal para iniciar o jogo
config.json: Configurações ajustáveis como ID da câmera e limiar de acerto
assets/: Contém as imagens do contorno do estado e da bandeira
snapshots/: Pasta onde são salvas as imagens de vitória
Ajustes possíveis:
Você pode personalizar o jogo editando o arquivo config.json para ajustar:

ID da câmera
Limiar de acerto (padrão: 95%)
Caminhos das imagens
Modo espelho (para inverter horizontalmente a imagem da câmera)
Para sair do jogo a qualquer momento, pressione a tecla ESC.