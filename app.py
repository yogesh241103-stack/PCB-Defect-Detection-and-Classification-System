import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pandas as pd

# --- Configuration & Tech Stack ---
CLASS_NAMES = ['Missing_hole', 'Mouse_bite', 'Open_circuit', 'Short', 'Spur', 'Spurious_copper']
MODEL_PATH = "efficientnet_pcb_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Page Setup ---
st.set_page_config(
    page_title="PCB Defect Detector", 
    page_icon="🔍", 
    layout="wide"
)

# --- Custom CSS for Metrics (Adapts to Light/Dark Mode) ---
st.markdown("""
    <style>
    div[data-testid="metric-container"] {
        background-color: rgba(128, 128, 128, 0.1);
        border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 5% 10% 5% 10%;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# --- Main Header ---
st.title("🔍 AI PCB Defect Detection System")
st.markdown("Automated optical inspection powered by deep learning and computer vision.")
st.markdown("---")

# --- Load Model ---
@st.cache_resource(show_spinner="Loading AI Model...")
def load_model():
    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(num_ftrs, len(CLASS_NAMES))
    )
    # Use weights_only=True for safer loading (best practice in newer PyTorch)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    return model.to(device).eval()

try:
    model = load_model()
except Exception as e:
    st.error(f"⚠️ Error loading model: Ensure '{MODEL_PATH}' is in the root directory.")
    st.exception(e)
    st.stop()

inference_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- UI Uploaders ---
with st.container():
    st.subheader("1. Upload Images")
    col1, col2 = st.columns(2)
    with col1:
        template_file = st.file_uploader("Upload Template Image (Defect-Free) 🟢", type=['jpg', 'png', 'jpeg'])
    with col2:
        test_file = st.file_uploader("Upload Test Image (To Inspect) 🔴", type=['jpg', 'png', 'jpeg'])

# --- Processing Pipeline ---
if template_file and test_file:
    st.markdown("---")
    st.subheader("2. Inspection Results")
    
    # Read files
    t_bytes = np.frombuffer(template_file.read(), np.uint8)
    img_bytes = np.frombuffer(test_file.read(), np.uint8)
    
    template_gray = cv2.imdecode(t_bytes, cv2.IMREAD_GRAYSCALE)
    test_gray = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)
    display_img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    # Use a status container to give the user step-by-step feedback
    with st.status("Scanning PCB...", expanded=True) as status:
        st.write("Aligning images...")
        if template_gray.shape != test_gray.shape:
            h, w = template_gray.shape
            test_gray = cv2.resize(test_gray, (w, h))
            display_img = cv2.resize(display_img, (w, h))
        
        st.write("Extracting contours and performing image subtraction...")
        diff = cv2.absdiff(template_gray, test_gray)
        _, thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        kernel = np.ones((3, 3), np.uint8)
        clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        clean = cv2.dilate(clean, kernel, iterations=1)
        
        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        st.write("Classifying defects using EfficientNet...")
        detected_list = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 150:
                x, y, w, h = cv2.boundingRect(cnt)
                
                roi_raw = display_img[max(0,y-15):min(h+y+15, display_img.shape[0]), 
                                      max(0,x-15):min(w+x+15, display_img.shape[1])]
                
                roi_pil = Image.fromarray(cv2.cvtColor(roi_raw, cv2.COLOR_BGR2RGB))
                input_tensor = inference_transform(roi_pil).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    _, pred_idx = torch.max(output, 1)
                    label = CLASS_NAMES[pred_idx.item()]
                
                # Draw boxes
                cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 0, 255), 3)
                
                # Add background to text for better readability
                text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(display_img, (x, y - text_size[1] - 10), (x + text_size[0], y), (0, 0, 255), -1)
                cv2.putText(display_img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                detected_list.append({"Type": label, "Location": f"({x}, {y})"})
        
        status.update(label="Scan Complete!", state="complete", expanded=False)

    # --- UI: Results Display ---
    
    # Highlight Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Status", "Defects Found" if detected_list else "Passed", "Fail" if detected_list else "Pass", delta_color="inverse")
    m2.metric("Total Defects", len(detected_list))
    m3.metric("Resolution", f"{display_img.shape[1]}x{display_img.shape[0]}")
    
    st.write("") # Spacer

    # Use Tabs for a cleaner layout
    tab1, tab2, tab3 = st.tabs(["🎯 Visual Results", "📋 Defect Logs", "🔍 Original Images"])
    
    with tab1:
        st.write("") # Add a little top spacing
        
        # Create 3 columns: Left Pad (20%), Center Image (60%), Right Pad (20%)
        # Adjust these numbers to make the image bigger or smaller!
        left_pad, center_col, right_pad = st.columns([1, 3, 1]) 
        
        with center_col:
            st.image(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB), caption="AI Analyzed PCB Result", use_container_width=True)
    with tab2:
        if detected_list:
            df = pd.DataFrame(detected_list)
            st.dataframe(df, use_container_width=True)
            st.download_button(
                label="📥 Export Logs to CSV",
                data=df.to_csv(index=False),
                file_name="pcb_defect_logs.csv",
                mime="text/csv",
                type="primary"
            )
        else:
            st.success("✅ No defects detected! The PCB looks good.")
            
    with tab3:
        # Show what the user uploaded for comparison
        col_orig1, col_orig2 = st.columns(2)
        with col_orig1:
            st.image(template_file.getvalue(), caption="Template (Reference)", use_container_width=True)
        with col_orig2:
            st.image(test_file.getvalue(), caption="Test (Original)", use_container_width=True)

else:
    # Empty state placeholder
    st.info("👆 Please upload both a Template and a Test image to begin inspection.")