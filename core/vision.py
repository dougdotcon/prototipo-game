import cv2
import numpy as np
import mediapipe as mp
import os
import urllib.request
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class VisionProcessor:
    def __init__(self, camera_id=0, mirror_mode=True):
        self.camera_id = camera_id
        self.mirror_mode = mirror_mode
        self.cap = None
        self.initialize_camera()
        self.initialize_selfie_segmenter()
        
        # Inicializar o detector de fundo
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)
        self.background = None
        self.has_background = False
        self.frame_count = 0

    def initialize_camera(self):
        """Initialize the webcam capture."""
        print(f"Initializing camera with ID: {self.camera_id}")
        try:
            self.cap = cv2.VideoCapture(self.camera_id)

            # Try to set resolution, but don't fail if it doesn't work
            try:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Lower resolution to avoid issues
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                print("Camera resolution set to 1280x720")
            except Exception as e:
                print(f"Warning: Could not set camera resolution: {e}")

            if not self.cap.isOpened():
                raise ValueError("Could not open webcam. Check camera_id in config.json")

            print("Camera initialized successfully")
        except Exception as e:
            print(f"Error initializing camera: {e}")
            raise

    def initialize_selfie_segmenter(self):
        """Initialize the MediaPipe Selfie Segmentation model."""
        # Create models directory if it doesn't exist
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        # Define model path
        model_filename = "selfie_segmenter.tflite"
        model_path = os.path.join(models_dir, model_filename)

        # Download model if it doesn't exist
        if not os.path.exists(model_path):
            print("Downloading selfie segmentation model...")
            model_url = 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite'
            urllib.request.urlretrieve(model_url, model_path)
            print("Model downloaded successfully!")

        # Initialize the segmenter with the local model
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.ImageSegmenterOptions(
            base_options=base_options,
            output_category_mask=True
        )
        self.segmenter = vision.ImageSegmenter.create_from_options(options)

    def get_frame(self):
        """Capture a frame from the webcam."""
        ret, frame = self.cap.read()
        if not ret:
            return None

        if self.mirror_mode:
            frame = cv2.flip(frame, 1)

        return frame

    def get_segmentation_mask(self, frame):
        """Get the segmentation mask for the person in the frame using combined methods."""
        # Usar vários métodos de detecção e combinar os resultados
        mask1 = self.get_mediapipe_mask(frame)
        mask2 = self.get_background_subtraction_mask(frame)
        
        # Combinar as máscaras (OR lógico)
        combined_mask = cv2.bitwise_or(mask1, mask2)
        
        # Aplicar operações morfológicas para melhorar a qualidade
        kernel = np.ones((9, 9), np.uint8)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=3)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        # Verificar se a máscara contém dados
        non_zero = cv2.countNonZero(combined_mask)
        total_pixels = combined_mask.shape[0] * combined_mask.shape[1]
        print(f"Combined mask: {non_zero} of {total_pixels} pixels ({non_zero/total_pixels*100:.1f}%)")
        
        return combined_mask

    def get_mediapipe_mask(self, frame):
        """Get mask using MediaPipe."""
        try:
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Create a MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Perform segmentation
            segmentation_result = self.segmenter.segment(mp_image)
            category_mask = segmentation_result.category_mask

            # Convert the mask to a binary numpy array
            mask = np.where(category_mask.numpy_view() == 1, 255, 0).astype(np.uint8)
            
            # Check if mask is mostly background (inverted)
            mask_count = cv2.countNonZero(mask)
            total_pixels = mask.shape[0] * mask.shape[1]
            
            # If more than 70% is marked, it's probably inverted
            if mask_count > 0.7 * total_pixels:
                mask = 255 - mask
                
            return mask
        except Exception as e:
            print(f"MediaPipe error: {e}")
            return np.zeros(frame.shape[:2], dtype=np.uint8)

    def get_background_subtraction_mask(self, frame):
        """Get mask using background subtraction."""
        # Converter para escala de cinza
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # Nos primeiros frames, inicializar o background
        if self.frame_count < 10:
            if self.background is None:
                self.background = gray.copy().astype("float")
            else:
                # Acumular o background (média ponderada)
                cv2.accumulateWeighted(gray, self.background, 0.5)
            
            self.frame_count += 1
            return np.zeros_like(gray)
        
        self.has_background = True
        
        # Calcular diferença absoluta entre o frame atual e o background
        frame_delta = cv2.absdiff(gray, cv2.convertScaleAbs(self.background))
        
        # Aplicar limiarização para obter uma máscara binária - valor reduzido para maior sensibilidade
        thresh = cv2.threshold(frame_delta, 15, 255, cv2.THRESH_BINARY)[1]
        
        # Aplicar operações morfológicas para melhorar a detecção
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Atualizar o modelo de background lentamente
        cv2.accumulateWeighted(gray, self.background, 0.01)
        
        return thresh

    def calculate_fill_percentage(self, body_mask, shape_mask):
        """Calculate the percentage of the shape filled by the body."""
        # Ensure masks are binary
        body_mask_bin = body_mask > 0
        shape_mask_bin = shape_mask > 0

        # Apply stronger morphological operations to improve the body mask
        kernel = np.ones((13, 13), np.uint8)  # Kernel ainda maior para preencher mais áreas
        body_mask_bin = cv2.dilate(body_mask_bin.astype(np.uint8), kernel, iterations=6)  # Mais iterações
        body_mask_bin = cv2.erode(body_mask_bin, kernel, iterations=1)
        body_mask_bin = body_mask_bin > 0  # Convert back to boolean

        # Calculate overlap - isto é o que está dentro da forma
        overlap = np.sum(np.logical_and(shape_mask_bin, body_mask_bin))
        total_shape = np.sum(shape_mask_bin)

        # Calcular quanto do corpo está fora do shape (excesso) 
        # O excesso é definido por pixels do corpo que estão FORA da forma
        body_area = np.sum(body_mask_bin)
        outside_shape = body_area - overlap
        
        # Calcular o excesso como uma proporção da área total da forma
        # Usar o total_shape como denominador para consistência
        excesso_percentual = (outside_shape / total_shape) * 100 if total_shape > 0 else 0
        excesso_percentual = min(100, excesso_percentual)  # Limitar a 100%
        
        # Calcular a cobertura interna (quanto do contorno está preenchido de verde)
        # Esse é o valor mais importante para determinar a vitória
        cobertura_interna = (overlap / total_shape) * 100 if total_shape > 0 else 0
        
        # Armazenar os valores para uso na visualização
        self.excesso_percentual = excesso_percentual
        self.cobertura_interna = cobertura_interna
        self.overlap_area = overlap
        self.total_shape_area = total_shape
        self.body_mask_area = body_area
        
        print(f"Overlap: {overlap} pixels, Shape: {total_shape} pixels, Body: {body_area} pixels")
        print(f"Cobertura interna: {cobertura_interna:.1f}%, Excesso: {excesso_percentual:.1f}%")

        if total_shape == 0:
            return 0

        # A porcentagem de preenchimento agora é mais influenciada pela cobertura interna
        # e menos penalizada pelo excesso (pra facilitar o jogo)
        percentage = cobertura_interna * (1 - excesso_percentual/400)  # Penalidade reduzida pelo excesso
        
        # Apply a boost para facilitar alcançar o threshold
        percentage = min(100, percentage * 1.5)  # Boost aumentado para 50%

        return percentage

    def release(self):
        """Release the webcam."""
        if self.cap is not None:
            self.cap.release()
