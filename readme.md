
# Shape SE – Reconhecimento Corporal com Mapa de Sergipe 🇧🇷

Um mini-protótipo baseado no projeto **Shape Match**, adaptado para uso educacional e cívico. Neste piloto, o jogador deve posicionar seu corpo de forma a preencher com **95% de precisão** o contorno do **mapa do estado de Sergipe**, exibido sobre a **bandeira do estado**. Ao completar corretamente, o sistema exibe a mensagem “Parabéns!” e registra uma foto.

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
- **MediaPipe (Pose ou SelfieSeg)** – segmentação corporal.
- **NumPy** – cálculo de interseção.
- **PyGame** – interface gráfica.
- **Pillow** – salvamento de imagens.

---

## 🗂 Estrutura de diretórios

```
shape_se/
├── core/
│   ├── vision.py        # Segmentação e análise de preenchimento
│   └── game.py          # Lógica principal do jogo
├── assets/
│   ├── shape-se.png     # Contorno transparente do estado
│   └── flag-se.jpg      # Bandeira como fundo da tela
├── snapshots/           # Pasta com imagens salvas dos acertos
├── config.json          # Configurações ajustáveis
├── run.py               # Script principal
└── README.md
```

---

## ⚙️ Requisitos

- **Sistema Operacional**: Windows 10/11 64-bit ou Ubuntu 22.04 LTS
- **Hardware**:
  - Webcam Full HD (1080p) – Logitech C920 recomendado
  - CPU i5 8ª geração ou Ryzen 5
  - 8 GB RAM
  - Iluminação frontal suave (mínimo 500 lux)
- **Dependências**:
  - Instale com:  
    ```bash
    pip install -r requirements.txt
    ```

---

## ▶️ Como rodar

1. Posicione a câmera a ~2 metros de distância do jogador.
2. Execute o script:
   ```bash
   python run.py
   ```
3. Mantenha o corpo alinhado ao contorno verde.
4. Ao atingir ≥ 95% de preenchimento, a mensagem de vitória será exibida.
5. A imagem do momento será salva automaticamente em `snapshots/`.

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

---
### Protótipo “**Shape SE**” – piloto simplificado do jogo de reconhecimento corporal

| Elemento | Adaptação específica |
|----------|----------------------|
| **Forma-alvo** | PNG transparente com **contorno do estado de Sergipe** (dimensão ~800 × 1 000 px). |
| **Fundo** | Imagem em tela cheia da **bandeira de Sergipe**; forma sobreposta ao centro. |
| **Regra de vitória** | **≥ 95 %** da área da forma coberta pelo corpo num único frame. |
| **Feedback** | Tela “Parabéns!” + captura da foto; reinicia após 3 s. |
| **Sem** | ranking, pontuação, cronômetro. |

---

## 1 · Pipeline técnico

graph TD
A[Webcam 1080p 30 fps] --> B(OpenCV Capture)
B --> C(MediaPipe SelfieSeg)
C --> D(Máscara binária corpo)
D --> E[Convolução AND com shape.png]
E --> F[Porcentagem preenchida]
F -->|>=95%| G[Tela Parabéns + snapshot]
```

- **Segmentação**: `mediapipe.tasks.python.vision` (ou BodySeg).  
- **Cálculo de preenchimento**:  
  ```python
  overlap = np.sum(shape_mask & body_mask)
  ratio   = overlap / np.sum(shape_mask) * 100
  ```
- **Interface**: PyGame fullscreen; dois layers: fundo (bandeira) e overlay verde gradual (alpha 0–90 %) indicando progresso.  
- **Parâmetros ajustáveis** em `config.json`: `threshold`: 95, `camera_id`, `shape_path`, `flag_path`.

---

## 2 · Estrutura de pastas

```
shape_se/
  ├─ core/
  │   ├─ vision.py      # captura + segmentação
  │   └─ game.py        # loop principal
  ├─ assets/
  │   ├─ shape-se.png   # contorno Sergipe
  │   └─ flag-se.jpg    # fundo
  ├─ config.json
  └─ run.py
```

---

## 3 · Passo-a-passo de implementação

| Dia | Tarefa | Resultado |
|-----|--------|-----------|
| 1 | Gerar shape PNG (Path no Inkscape) e bandeira JPG. | Arquivos em `assets/`. |
| 1 | Criar `run.py` – inicia PyGame, carrega imagens, abre webcam. | Tela fixa com shape. |
| 2 | Implementar `vision.py` com MediaPipe; retorna máscara binária. | Imagem corpo segmentada. |
| 2 | Função `check_fill(mask)` calcula % preenchimento. | Console exibe porcentagem. |
| 3 | Overlay de progresso (surface alfa). | Jogador vê preenchimento. |
| 3 | Gatilho ≥ 95 % → salva `snapshots/YYMMDD_HHMM.png`, mostra “Parabéns!”. | Loop completo. |
| 4 | Empacotamento (`pyinstaller --onefile`). | Executável stand-alone Win/Linux. |
| 5 | Teste em cenário real; ajustar iluminação e distância câmera. | Preciso ≥ 95 %. |

---

## 4 · Requisitos mínimos (hardware & ambiente)

- CPU i5 8ª G ou Ryzen 5, 8 GB RAM.  
- Webcam 1080p posicionada a 2 m de altura do tórax.  
- Iluminação frontal ≥ 500 lux.  
- Windows 10/11 64-bit ou Ubuntu 22.04.  
*(baseados na proposta inicial – se mantém válida) citeturn0file0*

---

