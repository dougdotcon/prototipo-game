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
        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        if not self.cap.isOpened():
            raise ValueError("Could not open webcam. Check camera_id in config.json")

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

        return mask

    def calculate_fill_percentage(self, body_mask, shape_mask):
        """Calculate the percentage of the shape filled by the body."""
        # Ensure masks are binary
        body_mask_bin = body_mask > 0
        shape_mask_bin = shape_mask > 0

        # Calculate overlap
        overlap = np.sum(np.logical_and(shape_mask_bin, body_mask_bin))
        total_shape = np.sum(shape_mask_bin)

        if total_shape == 0:
            return 0

        # Calculate percentage
        percentage = (overlap / total_shape) * 100
        return percentage

    def release(self):
        """Release the webcam."""
        if self.cap is not None:
            self.cap.release()
