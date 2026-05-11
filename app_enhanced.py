"""
Enhanced Deepfake Detection Application
- Improved error handling and validation
- Type hints and documentation
- Configuration management
- Logging enhancements
- Production-ready features
- Thread-safe operations
"""

import os
import logging
import warnings
from pathlib import Path
from typing import Tuple, Optional, List
from dataclasses import dataclass
from threading import Lock
from contextlib import contextmanager

import torch
import numpy as np
import cv2
from torch import nn
from torch.autograd import Variable
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import face_recognition
from werkzeug.utils import secure_filename
from flask import Flask, render_template, jsonify, request

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Configuration settings"""
    # File handling
    UPLOAD_FOLDER: str = os.path.join(os.path.dirname(__file__), 'Uploaded_Files')
    MAX_CONTENT_LENGTH: int = 16 * 1024 * 1024  # 16MB
    ALLOWED_EXTENSIONS: set = None
    
    # Model settings
    MODEL_PATH: str = 'model/df_model.pt'
    SEQUENCE_LENGTH: int = 20
    IMAGE_SIZE: int = 112
    NUM_CLASSES: int = 2
    
    # Processing settings
    BATCH_SIZE: int = 1
    USE_GPU: bool = torch.cuda.is_available()
    DEVICE: torch.device = None
    
    # Logging
    LOG_LEVEL: str = 'INFO'
    LOG_FILE: Optional[str] = 'deepfake_detection.log'
    
    def __post_init__(self):
        """Initialize dynamic fields"""
        if self.ALLOWED_EXTENSIONS is None:
            self.ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv'}
        
        self.DEVICE = torch.device('cuda' if self.USE_GPU else 'cpu')
        
        # Create upload folder
        Path(self.UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(config: Config) -> logging.Logger:
    """Configure logging with both file and console handlers"""
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, config.LOG_LEVEL))
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if config.LOG_FILE:
        file_handler = logging.FileHandler(config.LOG_FILE)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


config = Config()
logger = setup_logging(config)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# ============================================================================
# NEURAL NETWORK MODEL
# ============================================================================

class DeepfakeDetectionModel(nn.Module):
    """Deepfake detection model combining ResNeXt50 and LSTM"""
    
    def __init__(
        self,
        num_classes: int = 2,
        latent_dim: int = 2048,
        lstm_layers: int = 1,
        hidden_dim: int = 2048,
        bidirectional: bool = False,
        dropout_rate: float = 0.4
    ):
        super(DeepfakeDetectionModel, self).__init__()
        
        # Feature extraction backbone
        backbone = models.resnext50_32x4d(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            latent_dim, 
            hidden_dim, 
            lstm_layers, 
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(hidden_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
        logger.info(f"Model initialized with {num_classes} classes, "
                   f"latent_dim={latent_dim}, dropout={dropout_rate}")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, channels, height, width)
            
        Returns:
            Tuple of (feature_maps, classification_logits)
        """
        batch_size, seq_length, c, h, w = x.shape
        
        # Process all frames through feature extractor
        x = x.view(batch_size * seq_length, c, h, w)
        feature_maps = self.feature_extractor(x)
        
        # Global average pooling
        x = self.avgpool(feature_maps)
        x = x.view(batch_size, seq_length, -1)
        
        # LSTM temporal processing
        lstm_out, _ = self.lstm(x, None)
        
        # Classification from last LSTM output
        x = self.dropout(lstm_out[:, -1, :])
        logits = self.linear(x)
        
        return feature_maps, logits


# ============================================================================
# VIDEO PROCESSING
# ============================================================================

class VideoDataset(Dataset):
    """Dataset for loading and processing video frames"""
    
    def __init__(
        self,
        video_paths: List[str],
        sequence_length: int = 20,
        transform: Optional[transforms.Compose] = None,
        logger: logging.Logger = None
    ):
        """
        Initialize video dataset
        
        Args:
            video_paths: List of paths to video files
            sequence_length: Number of frames to extract
            transform: PyTorch transforms to apply
            logger: Logger instance
        """
        self.video_paths = video_paths
        self.sequence_length = sequence_length
        self.transform = transform
        self.logger = logger or logging.getLogger(__name__)
    
    def __len__(self) -> int:
        return len(self.video_paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get video frames and return as tensor
        
        Args:
            idx: Index of video
            
        Returns:
            Tensor of shape (1, sequence_length, 3, height, width)
            
        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If no valid frames extracted
        """
        video_path = self.video_paths[idx]
        self.logger.info(f"Processing video: {video_path}")
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        frames = []
        frame_indices = self._get_frame_indices()
        frame_count = 0
        
        try:
            for frame_idx, frame in enumerate(self._extract_frames(video_path)):
                if frame is None:
                    self.logger.warning(f"Skipping None frame at index {frame_idx}")
                    continue
                
                # Check if this frame should be processed
                if frame_count not in frame_indices and len(frames) < self.sequence_length:
                    frame_count += 1
                    continue
                
                try:
                    processed_frame = self._process_frame(frame, frame_idx)
                    if processed_frame is not None:
                        frames.append(processed_frame)
                except Exception as e:
                    self.logger.warning(f"Error processing frame {frame_idx}: {str(e)}")
                    continue
                
                if len(frames) >= self.sequence_length:
                    break
                
                frame_count += 1
        
        except Exception as e:
            self.logger.error(f"Error extracting frames: {str(e)}")
            raise
        
        if len(frames) == 0:
            raise ValueError(f"No valid frames extracted from {video_path}")
        
        # Pad with last frame if needed
        while len(frames) < self.sequence_length:
            frames.append(frames[-1])
        
        frames_tensor = torch.stack(frames[:self.sequence_length])
        self.logger.debug(f"Final frames shape: {frames_tensor.shape}")
        
        return frames_tensor.unsqueeze(0)
    
    def _get_frame_indices(self) -> List[int]:
        """Get indices of frames to sample uniformly across video"""
        # Sample frames uniformly (simple approach)
        return list(range(0, 100, max(1, 100 // self.sequence_length)))
    
    def _extract_frames(self, video_path: str):
        """Generator to extract frames from video"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break
                yield frame
        finally:
            cap.release()
    
    def _process_frame(self, frame: np.ndarray, frame_idx: int) -> Optional[torch.Tensor]:
        """
        Process individual frame
        
        Args:
            frame: OpenCV frame (BGR)
            frame_idx: Frame index for logging
            
        Returns:
            Processed tensor or None if processing failed
        """
        try:
            # Detect faces
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = face_recognition.face_locations(frame_rgb, model='hog')
            
            if len(faces) > 0:
                # Crop to face
                top, right, bottom, left = faces[0]
                frame = frame[top:bottom, left:right, :]
                
                # Resize to required dimensions
                if frame.shape[0] == 0 or frame.shape[1] == 0:
                    self.logger.warning(f"Invalid face crop at frame {frame_idx}")
                    frame = cv2.resize(frame, (config.IMAGE_SIZE, config.IMAGE_SIZE))
                else:
                    frame = cv2.resize(frame, (config.IMAGE_SIZE, config.IMAGE_SIZE))
                
                self.logger.debug(f"Face detected and cropped at frame {frame_idx}")
            else:
                # No face: resize full frame
                frame = cv2.resize(frame, (config.IMAGE_SIZE, config.IMAGE_SIZE))
                self.logger.debug(f"No face detected at frame {frame_idx}, using full frame")
            
            # Apply transforms
            if self.transform:
                transformed = self.transform(frame)
                return transformed
            else:
                # Fallback: basic conversion to tensor
                frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                return frame
        
        except Exception as e:
            self.logger.error(f"Error processing frame {frame_idx}: {str(e)}")
            return None


# ============================================================================
# VIDEO VALIDATION
# ============================================================================

def validate_video(video_path: str) -> Tuple[bool, Optional[str]]:
    """
    Validate video file
    
    Args:
        video_path: Path to video file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check file extension
    ext = Path(video_path).suffix.lstrip('.').lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        return False, f"Unsupported file format: {ext}"
    
    # Check file size
    file_size = os.path.getsize(video_path)
    if file_size == 0:
        return False, "Video file is empty"
    
    # Try to open video
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return False, "Cannot open video file"
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cap.release()
        
        # Check minimum requirements
        if frame_count < 20:
            return False, f"Video too short: {frame_count} frames (minimum 20)"
        
        if width < 64 or height < 64:
            return False, f"Video resolution too low: {width}x{height}"
        
        logger.info(f"Video validated: {frame_count} frames, {fps}fps, {width}x{height}")
        return True, None
    
    except Exception as e:
        logger.error(f"Video validation error: {str(e)}")
        return False, str(e)


# ============================================================================
# MODEL LOADING AND PREDICTION
# ============================================================================

class ModelManager:
    """Manages model loading, caching, and inference"""
    
    def __init__(self, model_path: str, device: torch.device):
        """Initialize model manager"""
        self.model_path = model_path
        self.device = device
        self.model = None
        self._lock = Lock()
        self._load_model()
    
    def _load_model(self) -> None:
        """Load model from disk"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        try:
            with self._lock:
                if self.model is None:
                    self.model = DeepfakeDetectionModel(num_classes=config.NUM_CLASSES)
                    state_dict = torch.load(self.model_path, map_location=self.device)
                    self.model.load_state_dict(state_dict)
                    self.model.to(self.device)
                    self.model.eval()
                    logger.info(f"Model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def predict(self, frames: torch.Tensor) -> Tuple[int, float]:
        """
        Run inference on video frames
        
        Args:
            frames: Tensor of shape (1, 1, sequence_length, 3, height, width)
            
        Returns:
            Tuple of (prediction_class, confidence_percentage)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        if frames.shape[1] == 0:
            raise ValueError("Empty frame sequence")
        
        try:
            with torch.no_grad():
                frames = frames.to(self.device)
                _, logits = self.model(frames)
                probabilities = torch.softmax(logits, dim=1)
                confidence, prediction = torch.max(probabilities, 1)
                
                pred_class = int(prediction.item())
                conf_percent = float(confidence.item()) * 100
                
                logger.info(f"Prediction: class={pred_class}, confidence={conf_percent:.2f}%")
                return pred_class, conf_percent
        
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise


# ============================================================================
# FLASK APPLICATION
# ============================================================================

def create_app(config_obj: Config) -> Flask:
    """
    Create and configure Flask application
    
    Args:
        config_obj: Configuration object
        
    Returns:
        Flask app instance
    """
    app = Flask(__name__, template_folder="templates")
    
    # Configuration
    app.config['UPLOAD_FOLDER'] = config_obj.UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = config_obj.MAX_CONTENT_LENGTH
    
    # Load model
    try:
        model_manager = ModelManager(config_obj.MODEL_PATH, config_obj.DEVICE)
    except FileNotFoundError as e:
        logger.warning(f"Model not found: {str(e)}. Demo mode will be used.")
        model_manager = None
    
    # Create transforms
    normalize_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config_obj.IMAGE_SIZE, config_obj.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # ====================================================================
    # ROUTES
    # ====================================================================
    
    @app.route('/', methods=['GET'])
    def homepage():
        """Serve homepage"""
        try:
            return render_template('index.html')
        except Exception as e:
            logger.error(f"Error rendering homepage: {str(e)}")
            return jsonify({'error': 'Failed to load homepage'}), 500
    
    @app.route('/detect', methods=['POST'])
    def detect():
        """
        API endpoint for deepfake detection
        
        Expected request:
        - POST /detect
        - file: video file
        
        Returns:
            JSON with keys:
            - output: "REAL" or "FAKE"
            - confidence: float 0-100
            - error: error message (if applicable)
        """
        video_path = None
        
        try:
            # Validate request
            if 'video' not in request.files:
                return jsonify({'error': 'No video file provided'}), 400
            
            video_file = request.files['video']
            
            if video_file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            # Save file securely
            filename = secure_filename(video_file.filename)
            if not filename:
                return jsonify({'error': 'Invalid filename'}), 400
            
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            video_file.save(video_path)
            logger.info(f"File saved: {video_path}")
            
            # Validate video
            is_valid, error_msg = validate_video(video_path)
            if not is_valid:
                return jsonify({'error': f'Invalid video: {error_msg}'}), 400
            
            # Run detection
            if model_manager is None:
                return jsonify({
                    'error': 'Model not available. Please deploy model file.'
                }), 503
            
            try:
                # Load video
                dataset = VideoDataset(
                    [video_path],
                    sequence_length=config.SEQUENCE_LENGTH,
                    transform=normalize_transform,
                    logger=logger
                )
                frames = dataset[0]
                
                # Predict
                pred_class, confidence = model_manager.predict(frames)
                
                # Format result
                result = {
                    'output': 'REAL' if pred_class == 1 else 'FAKE',
                    'confidence': round(confidence, 2),
                    'class': pred_class
                }
                
                logger.info(f"Detection complete: {result}")
                return jsonify(result), 200
            
            except ValueError as e:
                logger.error(f"Validation error: {str(e)}")
                return jsonify({'error': str(e)}), 400
            except RuntimeError as e:
                logger.error(f"Runtime error: {str(e)}")
                return jsonify({'error': str(e)}), 500
        
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}", exc_info=True)
            return jsonify({'error': 'Internal server error'}), 500
        
        finally:
            # Cleanup
            if video_path and os.path.exists(video_path):
                try:
                    os.remove(video_path)
                    logger.debug(f"Removed temporary file: {video_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove temporary file: {str(e)}")
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'model_loaded': model_manager is not None,
            'device': str(config.DEVICE)
        }), 200
    
    @app.errorhandler(413)
    def request_entity_too_large(error):
        """Handle file too large error"""
        return jsonify({
            'error': f'File too large. Maximum size: {config.MAX_CONTENT_LENGTH / (1024*1024):.0f}MB'
        }), 413
    
    @app.errorhandler(500)
    def internal_error(error):
        """Handle internal server error"""
        logger.error(f"Internal server error: {str(error)}")
        return jsonify({'error': 'Internal server error'}), 500
    
    return app


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    try:
        app = create_app(config)
        
        logger.info("=" * 70)
        logger.info("Deepfake Detection Application Starting")
        logger.info(f"Device: {config.DEVICE}")
        logger.info(f"Model: {config.MODEL_PATH}")
        logger.info(f"Upload folder: {config.UPLOAD_FOLDER}")
        logger.info("=" * 70)
        
        # Run app
        app.run(
            debug=False,  # Never use debug=True in production
            port=3000,
            threaded=True
        )
    
    except Exception as e:
        logger.critical(f"Failed to start application: {str(e)}", exc_info=True)
        raise
