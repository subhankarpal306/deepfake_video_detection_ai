from utils.video_utils import extract_frames
from models.vision.cnn_encoder import CNNEncoder
from models.vision.deepfake_classifier import DeepfakeClassifier
import torch

video_path = "data/sample_video.mp4"  # put any mp4 here

frames = extract_frames(video_path)
print(f"Extracted {len(frames)} frames")

encoder = CNNEncoder()
features = encoder(frames)

model = DeepfakeClassifier()
score = model(features)

print("Deepfake probability:", score.detach().item())

