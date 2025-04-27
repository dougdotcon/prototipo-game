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
        """Get the segmentation mask for the person in the frame."""
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create a MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Perform segmentation
        segmentation_result = self.segmenter.segment(mp_image)
        category_mask = segmentation_result.category_mask

        # Convert the mask to a binary numpy array
        mask = np.where(category_mask.numpy_view() == 1, 255, 0).astype(np.uint8)
        
        # Aplicar processamento adicional para melhorar a detecção
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        # Inverter a máscara para casos onde a segmentação pode estar invertida
        mask_count = cv2.countNonZero(mask)
        total_pixels = mask.shape[0] * mask.shape[1]
        
        # Se mais de 70% da imagem está marcada, provavelmente está invertida
        if mask_count > 0.7 * total_pixels:
            print("Mask inverted - detected as background instead of foreground")
            mask = 255 - mask
        
        # Verificar se a máscara contém algum pixel não zero
        if cv2.countNonZero(mask) == 0:
            print("Warning: Empty body mask detected!")
        else:
            print(f"Body mask detected: {cv2.countNonZero(mask)} pixels of {total_pixels} ({cv2.countNonZero(mask)/total_pixels*100:.1f}%)")
        
        return mask

    def calculate_fill_percentage(self, body_mask, shape_mask):
        """Calculate the percentage of the shape filled by the body."""
        # Ensure masks are binary
        body_mask_bin = body_mask > 0
        shape_mask_bin = shape_mask > 0

        # Apply stronger morphological operations to improve the body mask
        # This helps with filling gaps in the segmentation
        kernel = np.ones((11, 11), np.uint8)  # Kernel ainda maior
        body_mask_bin = cv2.dilate(body_mask_bin.astype(np.uint8), kernel, iterations=5)  # Mais iterações
        body_mask_bin = cv2.erode(body_mask_bin, kernel, iterations=1)
        body_mask_bin = body_mask_bin > 0  # Convert back to boolean

        # Calculate overlap
        overlap = np.sum(np.logical_and(shape_mask_bin, body_mask_bin))
        total_shape = np.sum(shape_mask_bin)

        # Calcular quanto do corpo está fora do shape (excesso)
        excesso = np.sum(np.logical_and(body_mask_bin, np.logical_not(shape_mask_bin)))
        
        # Armazenar o excesso e sobreposição para uso na visualização
        self.excesso_percentual = min(100, (excesso / total_shape) * 100) if total_shape > 0 else 0
        self.overlap_area = overlap
        self.total_shape_area = total_shape
        self.body_mask_area = np.sum(body_mask_bin)
        
        print(f"Overlap: {overlap} pixels, Shape: {total_shape} pixels, Body: {np.sum(body_mask_bin)} pixels")
        print(f"Fill percentage: {(overlap/total_shape*100) if total_shape > 0 else 0:.1f}%, Excess: {self.excesso_percentual:.1f}%")

        if total_shape == 0:
            return 0

        # Calculate percentage
        percentage = (overlap / total_shape) * 100

        # Apply a stronger boost to make it easier to reach the threshold
        # This helps compensate for imperfect segmentation
        percentage = min(100, percentage * 1.25)  # Aumentando o boost para 25%

        return percentage

    def release(self):
        """Release the webcam."""
        if self.cap is not None:
            self.cap.release()
