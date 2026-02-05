import sys
import os
import streamlit as st
import plotly.graph_objects as go

# ---------------- PATH FIX ----------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from pipelines.inference_pipeline import analyze

# ---------------- OPTIONAL AI PROVIDERS ----------------
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Deepfake Early Warning System",
    page_icon="🚨",
    layout="wide"
)

# ---------------- CLEAN PROFESSIONAL CSS ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #eef2ff, #f8fafc);
}

.block-container {
    max-width: 1400px;
    padding-top: 2.5rem;
}

.main-title {
    font-size: 3rem;
    font-weight: 900;
    text-align: center;
    background: linear-gradient(90deg, #4f46e5, #9333ea);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    text-align: center;
    color: #475569;
    margin-bottom: 2.5rem;
}

.card {
    background: white;
    border-radius: 16px;
    padding: 2rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
}

.metric {
    text-align: center;
    padding: 1.5rem;
}

.metric h2 {
    font-size: 2.4rem;
    margin: 0;
}

.risk-high { color: #dc2626; }
.risk-medium { color: #d97706; }
.risk-low { color: #16a34a; }

.footer {
    text-align: center;
    color: #64748b;
    margin-top: 4rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="main-title">Deepfake Early Warning System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Preventing viral deepfake misinformation using AI</div>', unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("⚙️ Configuration")

    gemini_api_key = st.text_input(
        "Gemini API Key",
        type="password",
        help="Optional: Used for AI recommendations"
    )

    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Optional: Used for AI recommendations"
    )

# ---------------- INPUT ----------------
st.subheader("📥 Upload Inputs")

c1, c2 = st.columns(2)

with c1:
    video_file = st.file_uploader("Video File", type=["mp4", "mov", "avi"])

with c2:
    csv_file = st.file_uploader("Social Propagation CSV", type=["csv"])

# ---------------- RISK GAUGE ----------------
def risk_gauge(score):
    score *= 100

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "steps": [
                {"range": [0, 40], "color": "#dcfce7"},
                {"range": [40, 70], "color": "#fef3c7"},
                {"range": [70, 100], "color": "#fee2e2"}
            ],
            "bar": {"color": "#4f46e5"}
        }
    ))

    fig.update_layout(height=350, margin=dict(t=30, b=10))
    return fig

# ---------------- ANALYSIS ----------------
if st.button("🚀 Run Risk Assessment", use_container_width=True):

    if not video_file or not csv_file:
        st.error("Please upload both files.")
        st.stop()

    with st.spinner("Running multimodal AI analysis..."):
        os.makedirs("temp", exist_ok=True)

        video_path = "temp/video.mp4"
        csv_path = "temp/social.csv"

        with open(video_path, "wb") as f:
            f.write(video_file.read())

        with open(csv_path, "wb") as f:
            f.write(csv_file.read())

        try:
            result = analyze(video_path=video_path, social_csv=csv_path)
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.stop()

    # ---------------- RESULT ----------------
    st.subheader("🎯 Risk Assessment Result")

    risk = result["risk_level"]

    if risk == "HIGH RISK":
        st.error("🚨 HIGH RISK — Immediate intervention required")
    elif risk == "MEDIUM RISK":
        st.warning("⚠️ MEDIUM RISK — Monitor closely")
    else:
        st.success("✅ LOW RISK — No immediate threat")

    # ---------------- METRICS ----------------
    m1, m2, m3 = st.columns(3)

    with m1:
        st.markdown(f"<div class='metric'><h2>{result['deepfake_probability']:.2f}</h2><p>Deepfake Probability</p></div>", unsafe_allow_html=True)

    with m2:
        st.markdown(f"<div class='metric'><h2>{result['virality_probability']:.2f}</h2><p>Virality Probability</p></div>", unsafe_allow_html=True)

    with m3:
        st.markdown(f"<div class='metric'><h2>{result['final_risk_score']:.2f}</h2><p>Final Risk Score</p></div>", unsafe_allow_html=True)

    # ---------------- GAUGE ----------------
    st.plotly_chart(risk_gauge(result["final_risk_score"]), use_container_width=True)

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
