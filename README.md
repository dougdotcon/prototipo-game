# 🎮 Vision Games - Jogos Interativos com Reconhecimento Corporal

Este repositório contém uma coleção de jogos interativos que utilizam visão computacional e reconhecimento corporal para criar experiências envolventes e educativas.

## 🎯 Jogos Disponíveis

### 1. Hole in the Camera
Inspirado no famoso game show "Hole in the Wall", este jogo desafia os jogadores a se encaixarem em diferentes formatos que aparecem na tela. 

**Experiência do Jogador:**
- 7 rounds por partida
- 10 segundos para se encaixar em cada forma
- Pontuação de 0-100 baseada na precisão do encaixe
- Pontuação final é a média dos 7 rounds
- Ideal para competições entre amigos

### 2. Shape SE
Um jogo educativo e cívico focado no estado de Sergipe.

**Experiência do Jogador:**
- O jogador deve posicionar seu corpo para preencher o contorno do mapa de Sergipe
- A silhueta do estado é exibida sobre a bandeira de Sergipe
- Meta: atingir 95% de precisão no preenchimento
- Cronômetro regressivo de 5 minutos
- Ao vencer: exibe "Viva Sergipe" e captura uma foto do momento
- Em caso de derrota: exibe "Game Over"

## 🔧 Requisitos Comuns

- Câmera web funcional
- Python 3.x instalado
- Boa iluminação no ambiente
- Espaço suficiente para movimentação (recomendado: 2 metros de distância da câmera)
- Fundo claro/branco para melhor detecção

## 🚀 Como Executar

### Hole in the Camera
1. Instale as dependências:
```bash
pip install pygame opencv-python scipy torch torchvision numpy
```

2. Baixe o modelo de pose corporal:
- Acesse: [Google Drive - Body Pose Model](https://drive.google.com/drive/folders/1Nb6gQIHucZ3YlzVr5ME3FznmF4IqrJzL?usp=sharing)
- Baixe o arquivo "body_pose_model.pth"
- Coloque na pasta `hole-camera/deep_pose/`

3. Execute o jogo:
```bash
cd hole-camera
python hole_in_the_camera_runner.py
```

### Shape SE
1. Instale as dependências:
```bash
cd shape-se
pip install -r requirements.txt
```

2. Execute o jogo:
```bash
python run.py
```

## 🎯 Dicas para Melhor Experiência

1. **Iluminação:**
   - Mantenha o ambiente bem iluminado
   - Evite contraluz

2. **Vestuário:**
   - Use roupas que contrastem com o fundo
   - Para Hole in the Camera: roupas escuras com fundo claro
   - Para Shape SE: roupas que facilitem a visualização do contorno corporal

3. **Posicionamento:**
   - Mantenha-se a aproximadamente 2 metros da câmera
   - Certifique-se que seu corpo inteiro está visível no quadro
   - Evite objetos ou pessoas no fundo

4. **Ambiente:**
   - Espaço livre para movimentação
   - Fundo limpo e preferencialmente de cor única
   - Boa ventilação (você vai se movimentar bastante!)

## 🤝 Contribuindo

Se quiser criar novos desafios para o Hole in the Camera:

1. Use `create_mask.py` para criar novas formas
2. Execute `create_csv.py` para analisar as posições das juntas
3. Atualize a lista de formas em `hole_in_the_camera_model.py`

Para mais detalhes, consulte o README específico de cada jogo nas suas respectivas pastas.