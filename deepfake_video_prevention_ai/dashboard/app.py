import streamlit as st
import time
import random
import pandas as pd

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="DeepGuard | Early Warning System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* Dark SaaS Theme Overrides */
    .stApp {
        background-color: #0E1117;
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Typography */
    h1, h2, h3 {
        color: #FAFAFA;
        font-weight: 600;
        letter-spacing: -0.5px;
    }
    p, label, .stMarkdown {
        color: #A1A1AA;
    }
    
    /* Cards/Containers */
    .css-1r6slb0, .stCard {
        background-color: #18181B;
        border: 1px solid #27272A;
        border-radius: 8px;
        padding: 20px;
    }
    
    /* Primary CTA Button */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #DC2626 0%, #B91C1C 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 8px;
        width: 100%;
        transition: all 0.2s ease;
        box-shadow: 0 4px 12px rgba(220, 38, 38, 0.3);
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(90deg, #EF4444 0%, #DC2626 100%);
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(220, 38, 38, 0.4);
    }
    
    /* Input Areas */
    .stFileUploader {
        border: 1px dashed #3F3F46;
        border-radius: 8px;
        padding: 20px;
        background-color: #18181B;
    }
    
    /* Divider */
    hr {
        border-color: #27272A;
        margin: 2rem 0;
    }
    
    /* Metric Cards */
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }
    
    /* Custom Classes */
    .pipeline-step {
        text-align: center;
        padding: 10px;
        background: #18181B;
        border: 1px solid #27272A;
        border-radius: 6px;
        font-size: 0.9rem;
        color: #E4E4E7;
    }
    
    .risk-badge-high {
        background-color: rgba(220, 38, 38, 0.1);
        color: #EF4444;
        padding: 4px 12px;
        border-radius: 100px;
        font-weight: 600;
        border: 1px solid rgba(220, 38, 38, 0.2);
    }
    
    .recommendation-card {
        background-color: #18181B;
        border-left: 4px solid #EF4444;
        padding: 15px;
        margin-bottom: 10px;
        border-radius: 0 4px 4px 0;
    }

    /* Progress Bar Customization */
    .stProgress > div > div > div > div {
        background-color: #EF4444;
    }
</style>
""", unsafe_allow_html=True)

# --- MOCK BACKEND LOGIC (Do not modify signature as per constraints) ---
def analyze(video_file, metadata_file):
    """
    Mock analysis function representing the backend AI logic.
    Returns a dictionary with risk scores and metrics.
    """
    time.sleep(2.5) # Simulate processing time
    
    # Deterministic randomness based on filename length to seem consistent
    seed = len(video_file.name) if video_file else 0
    random.seed(seed)
    
    deepfake_score = random.uniform(70, 99)
    virality_score = random.uniform(60, 95)
    
    # Simple weighted risk calculation
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

# --- UI IMPLEMENTATION ---

# 1. HERO SECTION
col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <h1 style='font-size: 3rem; margin-bottom: 0.5rem; background: -webkit-linear-gradient(45deg, #F87171, #EF4444); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
                DeepGuard Early Warning
            </h1>
            <p style='font-size: 1.2rem; color: #D4D4D8;'>
                Proactive Deepfake Detection & Virality Containment System
            </p>
            <p style='font-size: 0.9rem; color: #71717A;'>
                Protecting public trust by identifying synthetic media before it scales.
            </p>
        </div>
    """, unsafe_allow_html=True)

st.divider()

# 2. PIPELINE STORYTELLING
st.markdown("<h3 style='text-align: center; margin-bottom: 1.5rem;'>System Pipeline</h3>", unsafe_allow_html=True)
p_col1, p_col2, p_col3, p_col4, p_col5, p_col6, p_col7 = st.columns([2, 0.5, 2, 0.5, 2, 0.5, 2])

with p_col1:
    st.markdown("""
    <div class="pipeline-step">
        <div style="font-size: 1.5rem;">👁️</div>
        <strong>Vision AI</strong><br>
        <span style="font-size: 0.8rem; color: #71717A;">Artifact Detection</span>
    </div>
    """, unsafe_allow_html=True)

with p_col3:
    st.markdown("""
    <div class="pipeline-step">
        <div style="font-size: 1.5rem;">🕸️</div>
        <strong>Graph AI</strong><br>
        <span style="font-size: 0.8rem; color: #71717A;">Source Tracking</span>
    </div>
    """, unsafe_allow_html=True)

with p_col5:
    st.markdown("""
    <div class="pipeline-step">
        <div style="font-size: 1.5rem;">⚡</div>
        <strong>Risk Fusion</strong><br>
        <span style="font-size: 0.8rem; color: #71717A;">Multi-modal Analysis</span>
    </div>
    """, unsafe_allow_html=True)

with p_col7:
    st.markdown("""
    <div class="pipeline-step" style="border-color: #DC2626;">
        <div style="font-size: 1.5rem;">🚨</div>
        <strong>Early Warning</strong><br>
        <span style="font-size: 0.8rem; color: #EF4444;">Alert Generation</span>
    </div>
    """, unsafe_allow_html=True)

# Arrows
arrow_style = "<div style='text-align: center; padding-top: 20px; color: #52525B;'>→</div>"
with p_col2: st.markdown(arrow_style, unsafe_allow_html=True)
with p_col4: st.markdown(arrow_style, unsafe_allow_html=True)
with p_col6: st.markdown(arrow_style, unsafe_allow_html=True)

st.divider()

# 3. INPUT SECTION
st.subheader("Step 1: Upload Evidence")

input_col1, input_col2 = st.columns(2)

with input_col1:
    st.markdown("##### 📹 Media File")
    st.markdown("<p style='font-size: 0.8rem;'>Supported: MP4, MOV, AVI. Max 200MB.</p>", unsafe_allow_html=True)
    video_file = st.file_uploader("Upload video content", type=['mp4', 'mov', 'avi'], label_visibility="collapsed")
    if video_file:
        st.success(f"Loaded: {video_file.name} ({round(video_file.size / 1024 / 1024, 2)} MB)")

with input_col2:
    st.markdown("##### 📄 Metadata / Logs")
    st.markdown("<p style='font-size: 0.8rem;'>Supported: CSV, JSON. Max 50MB.</p>", unsafe_allow_html=True)
    metadata_file = st.file_uploader("Upload provenance data", type=['csv', 'json'], label_visibility="collapsed")
    if metadata_file:
        st.success(f"Loaded: {metadata_file.name}")

st.divider()

# 4. PRIMARY CALL-TO-ACTION
c_col1, c_col2, c_col3 = st.columns([1, 2, 1])
analyze_clicked = False

with c_col2:
    if video_file is not None:
        analyze_clicked = st.button("🚨 Run Early Warning Scan")
    else:
        st.info("👆 Please upload a video file to initiate the scan.")

# 5. RESULTS SECTION
if analyze_clicked and video_file:
    with st.spinner("Processing media via Neural Engine..."):
        results = analyze(video_file, metadata_file)

    st.divider()

    # ---------------- HEADER ----------------
    res_col1, res_col2 = st.columns([2, 1])

    with res_col1:
        st.subheader("Analysis Results")

    with res_col2:
        st.markdown(
            f"<div style='text-align: right; color: #71717A;'>"
            f"Analysis ID: {random.randint(10000, 99999)}</div>",
            unsafe_allow_html=True
        )

    # ---------------- METRICS ----------------
    m1, m2, m3 = st.columns(3)

    with m1:
        st.markdown(
            f"""
            <div class='metric'>
                <h2>{results['deepfake_probability']:.2f}</h2>
                <p>Deepfake Probability</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with m2:
        st.markdown(
            f"""
            <div class='metric'>
                <h2>{results['virality_probability']:.2f}</h2>
                <p>Virality Probability</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with m3:
        st.markdown(
            f"""
            <div class='metric'>
                <h2>{results['final_risk_score']:.2f}</h2>
                <p>Final Risk Score</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ---------------- AI RECOMMENDATIONS ----------------
    if openai_api_key and OPENAI_AVAILABLE:
        client = OpenAI(api_key=openai_api_key)

        prompt = f"""
        Risk Level: {risk}
        Deepfake Probability: {result['deepfake_probability']}
        Virality Probability: {result['virality_probability']}

        Give 3 concise recommendations.
        """

        with st.spinner("Generating AI recommendations..."):
            response = client.responses.create(
                model="gpt-4.1-mini",
                input=prompt
            )

        st.subheader("🤖 AI Recommendations")
        st.write(response.output_text)

    elif gemini_api_key and GEMINI_AVAILABLE:
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        with st.spinner("Generating AI recommendations..."):
            response = model.generate_content(prompt)

        st.subheader("🤖 AI Recommendations")
        st.write(response.text)

# ---------------- FOOTER ----------------
st.markdown("<div class='footer'>Preventive AI • Hackathon-Ready • Production-Safe</div>", unsafe_allow_html=True)
