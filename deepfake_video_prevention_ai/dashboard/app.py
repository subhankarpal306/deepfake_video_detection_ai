import streamlit as st
import time
import random
import tempfile
import cv2
import os
from google import genai



# ---------------- GEMINI SETUP ----------------
from google import genai

API_KEY = "AIzaSyCKhTmZS0gnVDptetUcJC_sVNtQ-E8bE1w"
client = genai.Client(api_key=API_KEY)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="DeepGuard", layout="wide")

# ---------------- MOCK BACKEND ----------------
def analyze(video_file, metadata_file):
    time.sleep(2)

    seed = len(video_file.name)
    random.seed(seed)

    deepfake_score = random.uniform(70, 99)
    virality_score = random.uniform(60, 95)

    final_risk_score = (deepfake_score * 0.7) + (virality_score * 0.3)

    risk_level = "LOW"
    if final_risk_score > 40: risk_level = "MEDIUM"
    if final_risk_score > 75: risk_level = "HIGH"

    return {
        "risk_level": risk_level,
        "deepfake_probability": deepfake_score,
        "virality_potential": virality_score,
        "final_risk_score": final_risk_score,
        "audio_artifacts": random.choice(["Detected", "None"]),
        "face_inconsistencies": random.choice(["High", "Low", "None"])
    }

# ---------------- GEMINI RECOMMENDATION ----------------
def ai_recommendation_stream(results):

    prompt = f"""
You are a senior Trust & Safety analyst.

Risk Level: {results['risk_level']}
Deepfake Probability: {results['deepfake_probability']:.1f}%
Virality Potential: {results['virality_potential']:.1f}%
Audio Artifacts: {results['audio_artifacts']}
Face Issues: {results['face_inconsistencies']}

Give concise, actionable platform safety recommendations.
"""

    return client.models.generate_content_stream(
        model="gemini-2.5-flash-lite",
        contents=prompt
    )


def display_ai(results):
    st.subheader("🤖 AI Safety Recommendations")

    stream = ai_recommendation_stream(results)

    text = ""
    box = st.empty()

    for chunk in stream:
        if chunk.text:
            text += chunk.text
            box.markdown(text)

# ---------------- HERO ----------------
st.markdown("""
<h1 style='text-align:center;'>🚨 DeepGuard Early Warning</h1>
<p style='text-align:center;'>Proactive Deepfake Detection & Virality Containment</p>
""", unsafe_allow_html=True)

st.divider()

# ---------------- UPLOAD ----------------
col1, col2 = st.columns(2)

with col1:
    video_file = st.file_uploader("Upload Video", type=["mp4","mov","avi"])

with col2:
    metadata_file = st.file_uploader("Upload Metadata", type=["csv","json"])

st.divider()

# ---------------- RUN BUTTON ----------------
if video_file:
    if st.button("🚨 Run Early Warning Scan"):

        with st.spinner("Analyzing..."):
            results = analyze(video_file, metadata_file)

        st.success("Analysis Complete")

        c1,c2,c3 = st.columns(3)

        # Risk Box
        with c1:
            st.subheader("Risk Level")
            st.error(results["risk_level"])

        # Metrics
        with c2:
            st.metric("Deepfake Probability", f"{results['deepfake_probability']:.1f}%")
            st.metric("Virality Potential", f"{results['virality_potential']:.1f}%")

        # Score
        with c3:
            score = results["final_risk_score"]
            st.metric("Overall Score", f"{score:.1f}/100")
            st.progress(score/100)

        st.divider()

        # Gemini AI
        display_ai(results)

else:
    st.info("Upload a video to start")

st.markdown("---")
st.markdown("Built for Hackathon 🚀")
