from pipelines.inference_pipeline import analyze

result = analyze(
    video_path="data/sample_video.mp4",
    social_csv="data/raw/social_logs.csv"
)

print("\n===== EARLY WARNING REPORT =====")
for k, v in result.items():
    print(f"{k}: {v}")
