from flask import Flask, render_template, redirect, request, url_for, send_file
from flask import jsonify, json
from werkzeug.utils import secure_filename
import os
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import face_recognition
from torch.autograd import Variable
import time
import sys
from torch import nn
from torchvision import models
from skimage import img_as_ubyte
import logging
import warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment setup
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# File upload configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'Uploaded_Files')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Constants
im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
sm = nn.Softmax()

class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size*seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm,_ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:,-1,:]))

class validation_dataset(Dataset):
    def __init__(self, video_names, sequence_length=60, transform=None):
        self.video_names = video_names
        self.transform = transform
        self.count = sequence_length

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_path = self.video_names[idx]
        logger.info(f"Processing video: {video_path}")
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        frames = []
        a = int(100 / self.count)
        first_frame = np.random.randint(0,a)
        
        for i, frame in enumerate(self.frame_extract(video_path)):
            if frame is None:
                logger.warning(f"Skipping invalid frame {i}")
                continue
                
            logger.debug(f"Processing frame {i}")
            
            # Convert frame to RGB (face_recognition expects RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = face_recognition.face_locations(frame_rgb)
            
            if len(faces) > 0:
                top, right, bottom, left = faces[0]
                frame = frame[top:bottom, left:right, :]
                logger.debug(f"Face detected in frame {i}")
            else:
                logger.warning(f"No face detected in frame {i}, using full frame")
                frame = cv2.resize(frame, (im_size, im_size))
                
            try:
                transformed_frame = self.transform(frame)
                frames.append(transformed_frame)
            except Exception as e:
                logger.error(f"Error transforming frame {i}: {str(e)}")
                continue
                
            if len(frames) >= self.count:
                break
        
        if len(frames) == 0:
            raise ValueError("No valid frames extracted from video")
        
        frames = torch.stack(frames)
        logger.info(f"Final frames shape: {frames.shape}")
        return frames.unsqueeze(0)

    def frame_extract(self, path):
        vidObj = cv2.VideoCapture(path)
        if not vidObj.isOpened():
            raise ValueError(f"Could not open video file: {path}")
            
        success = True
        while success:
            success, image = vidObj.read()
            if success:
                yield image
        vidObj.release()

def is_video_valid(video_path):
    """Check if video file is valid and can be processed"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return frame_count > 0
    except Exception as e:
        logger.error(f"Video validation error: {str(e)}")
        return False

def predict(model, img, path='./'):
    logger.info(f"Input shape: {img.shape}")
    if img.shape[1] == 0:  # Check if sequence length is 0
        raise ValueError("Empty frame sequence provided to model")
        
    fmap, logits = model(img.to())
    weight_softmax = model.linear1.weight.detach().cpu().numpy()
    logits = sm(logits)
    _, prediction = torch.max(logits, 1)
    confidence = logits[:, int(prediction.item())].item()*100
    logger.info(f'Prediction confidence: {confidence:.2f}%')
    return [int(prediction.item()), confidence]

def detectFakeVideo(videoPath):
    """Main function to detect deepfake in a video"""
    logger.info(f"Starting detection for video: {videoPath}")
    
    if not is_video_valid(videoPath):
        raise ValueError("Invalid video file or unsupported format")
    
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((im_size,im_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ])
    
    path_to_videos = [videoPath]
    video_dataset = validation_dataset(path_to_videos, sequence_length=20, transform=train_transforms)
    
    # Load model
    model = Model(2)
    path_to_model = 'model/df_model.pt'
    
    if not os.path.exists(path_to_model):
        raise FileNotFoundError(f"Model file not found at {path_to_model}")
    
    model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))
    model.eval()
    
    try:
        prediction = predict(model, video_dataset[0], './')
        result = "REAL" if prediction[0] == 1 else "FAKE"
        logger.info(f"Detection result: {result} (confidence: {prediction[1]:.2f}%)")
        return prediction
    except Exception as e:
        logger.error(f"Detection failed: {str(e)}")
        raise

# Flask application setup
app = Flask(__name__, template_folder="templates")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

@app.route('/', methods=['GET'])
def homepage():
    return render_template('index.html')

@app.route('/Detect', methods=['POST'])
def DetectPage():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        video = request.files['video']
        
        if video.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Secure filename and save
        video_filename = secure_filename(video.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
        video.save(video_path)
        logger.info(f"File saved to: {video_path}")
        
        # Validate video before processing
        if not is_video_valid(video_path):
            os.remove(video_path)
            return jsonify({'error': 'Invalid video file or unsupported format'}), 400
        
        # Process video
        try:
            prediction = detectFakeVideo(video_path)
            output = "REAL" if prediction[0] == 1 else "FAKE"
            data = {
                'output': output,
                'confidence': round(prediction[1], 2)
            }
            return jsonify(data)
            
        except Exception as detection_error:
            logger.error(f"Detection error: {str(detection_error)}")
            return jsonify({'error': str(detection_error)}), 500
            
        finally:
            # Clean up
            if os.path.exists(video_path):
                os.remove(video_path)
                logger.info(f"Removed temporary file: {video_path}")
                
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=3000)