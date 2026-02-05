from utils.video_utils import extract_frames
from models.vision.cnn_encoder import CNNEncoder
from models.vision.deepfake_classifier import DeepfakeClassifier

from models.graph.graph_builder import build_propagation_graph
from models.graph.virality_predictor import predict_virality

from models.fusion.risk_fusion import compute_risk


def analyze(video_path, social_csv):
    # ---- Vision AI ----
    frames = extract_frames(video_path)
    encoder = CNNEncoder()
    classifier = DeepfakeClassifier()

    features = encoder(frames)
    deepfake_prob = classifier(features).detach().item()

    # ---- Graph AI ----
    graph = build_propagation_graph(social_csv)
    virality_prob = predict_virality(graph)

    # ---- Fusion ----
    risk_score, level = compute_risk(deepfake_prob, virality_prob)

    return {
        "deepfake_probability": deepfake_prob,
        "virality_probability": virality_prob,
        "final_risk_score": risk_score,
        "risk_level": level
    }
