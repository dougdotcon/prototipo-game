# Shape SE – Reconhecimento Corporal com Mapa de Sergipe 🇧🇷

Um mini-protótipo baseado no projeto **Shape Match**, adaptado para uso educacional e cívico. Neste piloto, o jogador deve posicionar seu corpo de forma a preencher com **95% de precisão** o contorno do **mapa do estado de Sergipe**, exibido sobre a **bandeira do estado**. Ao completar corretamente, o sistema exibe a mensagem "Parabéns!" e registra uma foto.

---

## 📦 Funcionalidades

- Reconhecimento corporal em tempo real via webcam.
- Detecção de preenchimento da silhueta do estado.
- Validação de acerto com **95% ou mais** da área preenchida.
- Feedback visual progressivo e mensagem de conclusão.
- Captura de imagem do momento da vitória.
- Interface fullscreen e visual intuitivo.

---

## 🧠 Tecnologias utilizadas

- **Python 3.12**
- **OpenCV** – captura e manipulação de imagem.
- **MediaPipe (SelfieSeg)** – segmentação corporal.
- **NumPy** – cálculo de interseção.
- **PyGame** – interface gráfica.
- **Pillow** – salvamento de imagens.

---

## ▶️ Como rodar

1. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

2. Posicione a câmera a ~2 metros de distância do jogador.

3. Execute o script:
   ```bash
   python run.py
   ```

4. Mantenha o corpo alinhado ao contorno verde.

5. Ao atingir ≥ 95% de preenchimento, a mensagem de vitória será exibida.

6. A imagem do momento será salva automaticamente em `snapshots/`.

7. Pressione ESC para sair do jogo.

---

## 🔧 Ajustes disponíveis (config.json)

```json
{
  "camera_id": 0,
  "threshold": 95,
  "shape_path": "assets/shape-se.png",
  "background_path": "assets/flag-se.jpg",
  "mirror_mode": true
}
```
