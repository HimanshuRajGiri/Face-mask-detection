"""
Face Mask Detection - Streamlit Web App
"""
import os
import io
import sys
import numpy as np
from pathlib import Path
from PIL import Image

import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MaskGuard AI",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0d0d1a 0%, #1a1a2e 50%, #16213e 100%);
    min-height: 100vh;
}

/* Hero banner */
.hero {
    background: linear-gradient(120deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    border-radius: 20px;
    padding: 40px 50px;
    margin-bottom: 30px;
    box-shadow: 0 20px 60px rgba(102,126,234,0.4);
    text-align: center;
}
.hero h1 { color: white; font-size: 3rem; font-weight: 800; margin: 0; letter-spacing: -1px; }
.hero p  { color: rgba(255,255,255,0.85); font-size: 1.2rem; margin: 10px 0 0 0; }

/* Cards */
.card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 25px;
    backdrop-filter: blur(10px);
    margin-bottom: 20px;
}

/* Result badges */
.badge-safe {
    background: linear-gradient(135deg, #11998e, #38ef7d);
    color: white; border-radius: 50px; padding: 15px 35px;
    font-size: 1.8rem; font-weight: 700; display: inline-block;
    box-shadow: 0 10px 30px rgba(56,239,125,0.4);
}
.badge-danger {
    background: linear-gradient(135deg, #f7971e, #ffd200);
    color: #1a1a2e; border-radius: 50px; padding: 15px 35px;
    font-size: 1.8rem; font-weight: 700; display: inline-block;
    box-shadow: 0 10px 30px rgba(255,210,0,0.4);
}

/* Metric cards */
.metric-card {
    background: rgba(102,126,234,0.15);
    border: 1px solid rgba(102,126,234,0.3);
    border-radius: 12px; padding: 20px; text-align: center;
}
.metric-val { font-size: 2.5rem; font-weight: 800; color: #667eea; }
.metric-lbl { font-size: 0.85rem; color: rgba(255,255,255,0.6); margin-top: 5px; }

/* Progress bar override */
.stProgress > div > div > div { background: linear-gradient(90deg, #667eea, #f093fb); border-radius: 10px; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(13,13,26,0.9);
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* Upload area */
[data-testid="stFileUploader"] {
    background: rgba(102,126,234,0.1);
    border: 2px dashed rgba(102,126,234,0.5);
    border-radius: 16px; padding: 20px;
}

.section-title {
    color: white; font-size: 1.5rem; font-weight: 700;
    margin-bottom: 15px; padding-bottom: 10px;
    border-bottom: 2px solid rgba(102,126,234,0.4);
}
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent / "mask_cnn_model.pth"

@st.cache_resource
def load_model():
    try:
        import torch
        import torch.nn as nn

        class MaskCNN(nn.Module):
            def __init__(self):
                super().__init__()
                def conv_block(in_ch, out_ch):
                    return nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                        nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                        nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                        nn.MaxPool2d(2), nn.Dropout2d(0.1),
                    )
                self.features = nn.Sequential(
                    conv_block(3, 32), conv_block(32, 64),
                    conv_block(64, 128),
                )
                self.gap = nn.AdaptiveAvgPool2d(1)
                self.classifier = nn.Sequential(
                    nn.Flatten(), nn.Linear(128, 64),
                    nn.ReLU(inplace=True), nn.Dropout(0.5),
                    nn.Linear(64, 1), nn.Sigmoid(),
                )
            def forward(self, x):
                return self.classifier(self.gap(self.features(x))).squeeze(1)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MaskCNN().to(device)
        ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        return model, device, ckpt.get('val_acc', 0)
    except Exception as e:
        return None, None, 0

def detect_face(pil_img):
    import cv2
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Load Haar cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    
    if len(faces) == 0:
        return None, None
        
    # Take the largest face
    faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
    x, y, w, h = faces[0]
    
    # Crop the face with some padding
    padding = int(w * 0.15)
    y1 = max(0, y - padding)
    y2 = min(img_cv.shape[0], y + h + padding)
    x1 = max(0, x - padding)
    x2 = min(img_cv.shape[1], x + w + padding)
    
    face_crop = img_cv[y1:y2, x1:x2]
    face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
    
    # Draw a nice bounding box on the original image for display
    cv2.rectangle(img_cv, (x, y), (x+w, y+h), (56, 239, 125), 4)
    display_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    
    return face_pil, display_pil

def predict(model, device, pil_img):
    import torch
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tensor = tf(pil_img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = model(tensor).item()
    label = "With Mask" if prob > 0.5 else "Without Mask"
    conf  = prob if prob > 0.5 else 1 - prob
    return label, conf, prob

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 😷 MaskGuard AI")
    st.markdown("---")
    page = st.radio("Navigation", ["🏠 Home & Detection", "📊 Model Info", "ℹ️ About"])
    st.markdown("---")

    model, device, val_acc = load_model()
    if model:
        st.success("✅ CNN Model Loaded")
        st.metric("Val Accuracy", f"{val_acc*100:.1f}%")
    else:
        st.error("❌ Model not found\nRun `train_cnn.py` first!")
    
    st.markdown("---")
    st.caption("Built with PyTorch + Streamlit")

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>😷 MaskGuard AI</h1>
    <p>Deep Learning Face Mask Detection — CNN with 90%+ Accuracy</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  PAGE 1 — Home & Detection
# ══════════════════════════════════════════════════════════════════
if "Home" in page:
    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown('<div class="section-title">📸 Input Image</div>', unsafe_allow_html=True)
        
        tab_upload, tab_camera = st.tabs(["📁 Upload Image", "📷 Live Camera"])
        
        uploaded = None
        
        with tab_upload:
            uploaded_file = st.file_uploader(
                "Drag & drop or click to upload",
                type=["jpg", "jpeg", "png", "bmp", "webp"],
                label_visibility="collapsed"
            )
            if uploaded_file:
                uploaded = uploaded_file
                
        with tab_camera:
            camera_file = st.camera_input("Take a picture")
            if camera_file:
                uploaded = camera_file

        if uploaded:
            pil_img = Image.open(uploaded)
            st.image(pil_img, caption="Input Image", use_container_width=True)

    with col_result:
        st.markdown('<div class="section-title">🔍 Detection Result</div>', unsafe_allow_html=True)

        if not uploaded:
            st.markdown("""
            <div class="card" style="text-align:center; padding: 60px 20px;">
                <div style="font-size:4rem;">👆</div>
                <div style="color:rgba(255,255,255,0.5); margin-top:15px;">
                    Upload an image to detect mask
                </div>
            </div>
            """, unsafe_allow_html=True)
        elif model is None:
            st.error("Model not loaded. Please run `train_cnn.py` first.")
        else:
            with st.spinner("🔍 Analyzing..."):
                # Try to detect and crop a face
                face_pil, display_pil = detect_face(pil_img)
                
                if face_pil is not None:
                    # Predict on the cropped face
                    label, conf, raw_prob = predict(model, device, face_pil)
                    st.image(display_pil, caption="Face Detected", use_container_width=True)
                else:
                    st.warning("⚠️ No clear face detected! Analyzing the entire image, but results may be inaccurate.")
                    label, conf, raw_prob = predict(model, device, pil_img)

            is_mask = label == "With Mask"
            emoji   = "😷" if is_mask else "😶"
            badge_class = "badge-safe" if is_mask else "badge-danger"

            st.markdown(f"""
            <div style="text-align:center; padding: 30px 0;">
                <div style="font-size:5rem; margin-bottom:10px;">{emoji}</div>
                <div class="{badge_class}">{label}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**Confidence Score**")
            st.progress(conf)
            st.markdown(f"<div style='text-align:center; color:white; font-size:1.3rem; font-weight:700;'>{conf*100:.1f}%</div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-val" style="color:#38ef7d;">{raw_prob*100:.1f}%</div>
                    <div class="metric-lbl">P(Mask)</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-val" style="color:#ffd200;">{(1-raw_prob)*100:.1f}%</div>
                    <div class="metric-lbl">P(No Mask)</div>
                </div>
                """, unsafe_allow_html=True)

            # Recommendation
            if is_mask:
                st.success("✅ Great! This person is wearing a mask properly.")
            else:
                st.warning("⚠️ Alert! No mask detected. Please wear a mask!")

    # ── Quick stats ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-title">📈 Project Stats</div>', unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    metrics = [
        ("10,470+", "Total Images", "#667eea"),
        ("90%+",    "CNN Accuracy", "#38ef7d"),
        ("7",       "Models Compared", "#f093fb"),
        ("128×128", "Input Resolution", "#ffd200"),
    ]
    for col, (val, lbl, clr) in zip([s1, s2, s3, s4], metrics):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-val" style="color:{clr};">{val}</div>
            <div class="metric-lbl">{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  PAGE 2 — Model Info
# ══════════════════════════════════════════════════════════════════
elif "Model Info" in page:
    st.markdown('<div class="section-title">📊 Model Comparison</div>', unsafe_allow_html=True)

    import pandas as pd
    data = {
        "Model": ["CNN (Ours)", "SVM (RBF)", "MLP", "Random Forest", "Logistic Regression", "KNN", "Decision Tree", "Gaussian NB"],
        "Accuracy": ["~90%+", "71%", "71%", "68%", "67%", "65%", "63%", "60%"],
        "Type": ["Deep Learning", "Traditional ML", "Traditional ML", "Ensemble", "Traditional ML", "Instance-based", "Traditional ML", "Probabilistic"],
        "Training Speed": ["Slow (GPU)", "Slow", "Medium", "Medium", "Fast", "Fast", "Fast", "Very Fast"],
        "Notes": ["Custom 4-block CNN", "RBF kernel, C=30", "3 hidden layers", "100 estimators", "All CPUs", "k=3", "Default", "Feature indep. assumption fails"],
    }
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")
    col_arch, col_hist = st.columns([1, 1], gap="large")

    with col_arch:
        st.markdown('<div class="section-title">🧠 CNN Architecture</div>', unsafe_allow_html=True)
        layers = [
            ("Input", "3 × 128 × 128", "#667eea"),
            ("Conv Block 1", "Conv→BN→ReLU×2 → MaxPool → 32ch @ 64×64", "#764ba2"),
            ("Conv Block 2", "Conv→BN→ReLU×2 → MaxPool → 64ch @ 32×32", "#9c27b0"),
            ("Conv Block 3", "Conv→BN→ReLU×2 → MaxPool → 128ch @ 16×16", "#e91e63"),
            ("Conv Block 4", "Conv→BN→ReLU×2 → MaxPool → 256ch @ 8×8", "#f44336"),
            ("GAP",          "Global Average Pooling → 256-dim", "#ff9800"),
            ("Dense",        "FC(256→128) → ReLU → Dropout(0.5)", "#ffc107"),
            ("Output",       "FC(128→1) → Sigmoid", "#4caf50"),
        ]
        for name, desc, clr in layers:
            st.markdown(f"""
            <div style="display:flex; align-items:center; margin:8px 0;">
                <div style="background:{clr}; border-radius:8px; padding:6px 12px;
                            color:white; font-weight:600; min-width:120px; text-align:center;">
                    {name}
                </div>
                <div style="margin-left:15px; color:rgba(255,255,255,0.7); font-size:0.85rem;">
                    {desc}
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_hist:
        st.markdown('<div class="section-title">📈 Training History</div>', unsafe_allow_html=True)
        hist_path = Path(__file__).parent / "training_history.png"
        if hist_path.exists():
            st.image(str(hist_path), use_container_width=True)
        else:
            st.info("Training history will appear here after running `train_cnn.py`")

    st.markdown("---")
    st.markdown('<div class="section-title">⚙️ Training Config</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    configs = [
        [("Optimizer", "Adam"), ("Learning Rate", "0.001"), ("Weight Decay", "1e-4")],
        [("Epochs", "30 (early stop)"), ("Batch Size", "32"), ("Patience", "7 epochs")],
        [("Input Size", "128 × 128"), ("Augmentation", "Flip, Rotate, ColorJitter"), ("Loss", "Binary Cross-Entropy")],
    ]
    for col, items in zip([c1, c2, c3], configs):
        with col:
            for k, v in items:
                st.markdown(f"""
                <div class="card" style="padding:12px 16px; margin-bottom:8px;">
                    <div style="color:rgba(255,255,255,0.5); font-size:0.75rem;">{k}</div>
                    <div style="color:white; font-weight:600;">{v}</div>
                </div>
                """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  PAGE 3 — About
# ══════════════════════════════════════════════════════════════════
elif "About" in page:
    st.markdown('<div class="section-title">ℹ️ About This Project</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <p style="color:rgba(255,255,255,0.8); line-height:1.8; font-size:1rem;">
            This project classifies face images into <b>Mask</b> or <b>No-Mask</b> categories using 
            deep learning. Originally built with traditional ML models (SVM, MLP, etc.) achieving ~71% accuracy,
            we upgraded to a custom <b>Convolutional Neural Network (CNN)</b> built in PyTorch 
            achieving <b>90%+ accuracy</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">🛠️ Tech Stack</div>', unsafe_allow_html=True)
    techs = [
        ("🔥 PyTorch", "Deep learning framework for CNN training"),
        ("🌊 Streamlit", "Web application framework"),
        ("📷 OpenCV", "Image processing & Haar cascade face detection"),
        ("🖼️ PIL/Pillow", "Image loading and preprocessing"),
        ("🔢 NumPy", "Numerical computations"),
        ("📊 Matplotlib/Seaborn", "Training visualization"),
        ("🤖 Scikit-learn", "Traditional ML models baseline"),
    ]
    cols = st.columns(2)
    for i, (name, desc) in enumerate(techs):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="card" style="padding:15px 20px; margin-bottom:10px;">
                <b style="color:white;">{name}</b>
                <div style="color:rgba(255,255,255,0.6); font-size:0.85rem; margin-top:4px;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">👨‍💻 Developer</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <p style="color:rgba(255,255,255,0.8); font-size: 1.1rem;">
            Developed by <b>Himanshu Raj Giri</b><br>
            <span style="font-size: 0.9rem; color: rgba(255,255,255,0.5);">CNN + Web App implemented with ❤️</span>
        </p>
    </div>
    """, unsafe_allow_html=True)
