import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import numpy as np
import cv2
from datetime import datetime

# --- Custom CSS for styling ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 40px 20px;
        border-radius: 20px;
        text-align: center;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .header-container h1 {
        font-size: 3em;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .header-container p {
        font-size: 1.1em;
        margin-top: 10px;
        opacity: 0.95;
    }
    
    .upload-section {
        background: white;
        border-radius: 15px;
        padding: 30px;
        margin-bottom: 25px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
    }
    
    .result-section {
        background: white;
        border-radius: 15px;
        padding: 30px;
        margin-bottom: 25px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
    }
    
    .metric-card h3 {
        margin: 0;
        font-size: 0.9em;
        opacity: 0.9;
    }
    
    .metric-card p {
        margin: 10px 0 0 0;
        font-size: 2em;
        font-weight: 700;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 30px;
        border-radius: 8px;
        font-weight: 600;
        transition: transform 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stFileUploader {
        border: 2px dashed #667eea;
        border-radius: 10px;
    }
    
    .info-box {
        background: #e8f4f8;
        border-left: 4px solid #667eea;
        padding: 15px;
        border-radius: 5px;
        margin: 15px 0;
    }
    
    .success-box {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 15px;
        border-radius: 5px;
        margin: 15px 0;
    }
    </style>
""", unsafe_allow_html=True)

# --- Streamlit Page Config ---
st.set_page_config(page_title="Hand Fracture Detection", page_icon="🩻", layout="wide")

# --- Header ---
st.markdown("""
    <div class="header-container">
        <h1>🩻 Hand Fracture Detection</h1>
        <p>Advanced AI-Powered X-ray Analysis</p>
    </div>
""", unsafe_allow_html=True)

# --- Load the YOLO model ---
@st.cache_resource
def load_model():
    model_path = "best.pt"
    model = YOLO(model_path)
    return model

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    model_loaded = False

# --- Sidebar Info ---
with st.sidebar:
    st.markdown("### 📋 How to Use")
    st.info("""
    1. **Upload** an X-ray image (JPG/PNG)
    2. **Wait** for the AI analysis
    3. **Review** the detected fractures
    4. **Download** the result
    """)
    
    st.markdown("### ⚙️ Detection Settings")
    confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)
    st.caption("Lower = more sensitive, Higher = more specific")

# --- Main Content ---
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown("### 📤 Upload X-ray Image")
    uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "jpeg", "png"])
    st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None and model_loaded:
    # Process image
    image = Image.open(uploaded_file)
    image_array = np.array(image)
    
    with col1:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown("#### 🖼️ Uploaded Image")
        st.image(image, use_column_width=True, caption="Original X-ray")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Save and process
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_image_path = tmp_file.name
    
    # Run prediction
    with st.spinner("🔍 Analyzing image... This may take a moment"):
        results = model.predict(source=temp_image_path, conf=confidence, save=False, show=False)
        result_image = results[0].plot()
        detections = results[0].boxes
    
    # Extract detection data
    num_fractures = len(detections)
    confidences = detections.conf.cpu().numpy() if len(detections) > 0 else []
    avg_confidence = float(np.mean(confidences)) if len(confidences) > 0 else 0.0
    
    # Display results
    with col2:
        st.markdown('<div class="result-section">', unsafe_allow_html=True)
        st.markdown("#### 🩻 Analysis Results")
        
        if num_fractures > 0:
            st.markdown(f'<div class="success-box">✅ <strong>{num_fractures} fracture{"" if num_fractures == 1 else "s"} detected</strong></div>', unsafe_allow_html=True)
            st.image(result_image, use_column_width=True, caption="Detected Fractures")
        else:
            st.markdown('<div class="info-box">✅ No fractures detected</div>', unsafe_allow_html=True)
            st.image(result_image, use_column_width=True, caption="Analysis Result")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Metrics row
    st.markdown("---")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
    with col_m1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Fractures Found</h3>
            <p>{num_fractures}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_m2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Avg Confidence</h3>
            <p>{avg_confidence:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_m3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Threshold</h3>
            <p>{confidence:.0%}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_m4:
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.markdown(f"""
        <div class="metric-card">
            <h3>Scan Time</h3>
            <p>{timestamp}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Download section
    col_dl1, col_dl2, col_dl3 = st.columns(3)
    
    with col_dl1:
        result_image_pil = Image.fromarray(result_image)
        output_path = "fracture_result.jpg"
        result_image_pil.save(output_path)
        with open(output_path, "rb") as f:
            st.download_button(
                label="💾 Download Result",
                data=f.read(),
                file_name=f"fracture_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
                mime="image/jpeg",
                use_container_width=True
            )
    
    with col_dl2:
        # Generate detection report
        report = f"""
Fracture Detection Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY:
- Total Fractures Detected: {num_fractures}
- Average Confidence: {avg_confidence:.1%}
- Detection Threshold: {confidence:.0%}

DETAILS:
"""
        if num_fractures > 0:
            for i, conf in enumerate(confidences, 1):
                report += f"\nFracture {i}: {conf:.1%} confidence"
        else:
            report += "\nNo fractures detected in the X-ray image."
        
        st.download_button(
            label="📄 Download Report",
            data=report,
            file_name=f"fracture_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    with col_dl3:
        if st.button("🔄 Analyze Another", use_container_width=True):
            st.rerun()
    
    # Cleanup
    if os.path.exists(temp_image_path):
        os.remove(temp_image_path)

elif not model_loaded:
    st.error("Please ensure the YOLO model file (best.pt) is in the same directory as this script.")
else:
    st.markdown("### 👇 Get Started")
    st.markdown("""
    <div class="info-box">
    <strong>Welcome!</strong> This application uses advanced AI to detect fractures in hand X-rays. 
    Upload an X-ray image in the left panel to begin the analysis.
    </div>
    """, unsafe_allow_html=True)