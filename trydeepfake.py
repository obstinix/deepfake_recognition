import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

class DeepfakeDetector:
    def __init__(self, model_path='deepfake_detection_model.h5'):
        """
        Initialize the deepfake detector with a pre-trained model.
        
        Args:
            model_path (str): Path to the pre-trained model file.
        """
        try:
            self.model = load_model(model_path)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        # Define input shape expected by the model
        self.input_shape = self.model.input_shape[1:3]
        
    def preprocess_image(self, img_path):
        """
        Preprocess an image for model prediction.
        
        Args:
            img_path (str): Path to the image file.
            
        Returns:
            numpy.ndarray: Preprocessed image array.
        """
        try:
            # Load image
            img = image.load_img(img_path, target_size=self.input_shape)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # Normalize to [0,1]
            return img_array
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            raise
    
    def preprocess_video_frame(self, frame):
        """
        Preprocess a video frame for model prediction.
        
        Args:
            frame (numpy.ndarray): Video frame as numpy array.
            
        Returns:
            numpy.ndarray: Preprocessed frame array.
        """
        try:
            # Resize frame to match model input shape
            frame = cv2.resize(frame, self.input_shape)
            frame = frame.astype('float32') / 255.0
            frame = np.expand_dims(frame, axis=0)
            return frame
        except Exception as e:
            print(f"Error preprocessing frame: {e}")
            raise
    
    def predict_image(self, img_path):
        """
        Predict whether an image is real or fake.
        
        Args:
            img_path (str): Path to the image file.
            
        Returns:
            dict: Prediction results with confidence scores.
        """
        try:
            # Preprocess image
            img_array = self.preprocess_image(img_path)
            
            # Make prediction
            prediction = self.model.predict(img_array)
            
            # Interpret results
            confidence = prediction[0][0]
            is_fake = confidence > 0.5
            result = {
                'is_fake': bool(is_fake),
                'confidence': float(confidence),
                'message': 'Fake' if is_fake else 'Real'
            }
            return result
        except Exception as e:
            print(f"Error predicting image: {e}")
            raise
    
    def predict_video(self, video_path, frame_interval=10):
        """
        Predict whether a video contains deepfake content by sampling frames.
        
        Args:
            video_path (str): Path to the video file.
            frame_interval (int): Analyze every nth frame.
            
        Returns:
            dict: Prediction results with confidence scores and frame analysis.
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Could not open video file")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            predictions = []
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Only process frames at the specified interval
                if frame_count % frame_interval == 0:
                    # Preprocess frame
                    frame_array = self.preprocess_video_frame(frame)
                    
                    # Make prediction
                    prediction = self.model.predict(frame_array)
                    confidence = prediction[0][0]
                    is_fake = confidence > 0.5
                    
                    predictions.append({
                        'frame': frame_count,
                        'is_fake': bool(is_fake),
                        'confidence': float(confidence)
                    })
                
                frame_count += 1
            
            cap.release()
            
            # Calculate overall video prediction
            fake_frames = sum(1 for p in predictions if p['is_fake'])
            total_analyzed = len(predictions)
            fake_ratio = fake_frames / total_analyzed if total_analyzed > 0 else 0
            avg_confidence = sum(p['confidence'] for p in predictions) / total_analyzed if total_analyzed > 0 else 0
            
            result = {
                'is_fake': fake_ratio > 0.5,
                'fake_ratio': float(fake_ratio),
                'average_confidence': float(avg_confidence),
                'total_frames': total_frames,
                'analyzed_frames': total_analyzed,
                'frame_predictions': predictions
            }
            
            return result
        except Exception as e:
            print(f"Error predicting video: {e}")
            raise

# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = DeepfakeDetector()
    
    # Test with an image
    image_path = "test_image.jpg"
    image_result = detector.predict_image(image_path)
    print(f"\nImage Prediction Results:")
    print(f"  - Is Fake: {image_result['is_fake']}")
    print(f"  - Confidence: {image_result['confidence']:.2f}")
    print(f"  - Message: {image_result['message']}")
    
    # Test with a video
    video_path = "test_video.mp4"
    video_result = detector.predict_video(video_path)
    print(f"\nVideo Prediction Results:")
    print(f"  - Is Fake: {video_result['is_fake']}")
    print(f"  - Fake Frame Ratio: {video_result['fake_ratio']:.2f}")
    print(f"  - Average Confidence: {video_result['average_confidence']:.2f}")
    print(f"  - Analyzed {video_result['analyzed_frames']} of {video_result['total_frames']} frames")
