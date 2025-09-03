import os
import cv2
import numpy as np

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image
import io
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Salt Detection & Explainability",
    page_icon="🧂",
    layout="wide"
)

# Title and description
st.title("🧂 Salt Deposit Detection & Explainability")
st.markdown("Upload a seismic image to detect salt deposits and see AI explainability through Grad-CAM")

# Configuration
IMG_SIZE = (128, 128)
SEGMENTATION_MODEL_PATH = "models/pinn_unetpp_best.keras"
CLASSIFIER_MODEL_PATH = "models/unetpp_classifier.h5"

@st.cache_resource
def load_models():
    """Load models with caching to avoid reloading"""
    try:
        # Import custom loss
        from train_pinn_unetpp import physics_informed_loss
        
        # Load segmentation model
        seg_model = load_model(
            SEGMENTATION_MODEL_PATH,
            custom_objects={'physics_informed_loss': physics_informed_loss}
        )
        
        # Load classification model
        clf_model = load_model(CLASSIFIER_MODEL_PATH)
        
        return seg_model, clf_model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

def preprocess_image(uploaded_file):
    """Preprocess uploaded image"""
    try:
        # Read image
        image = Image.open(uploaded_file)
        
        # Convert to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Resize and normalize
        img_resized = cv2.resize(img_array, IMG_SIZE) / 255.0
        input_img = np.expand_dims(img_resized, axis=(0, -1))
        
        return img_array, img_resized, input_img
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, None, None

def get_gradcam_heatmap(model, image_array, last_conv_layer_name):
    """Generate Grad-CAM heatmap"""
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_array)
        loss = predictions[:, :, :, 0]
    
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0] * pooled_grads
    heatmap = tf.reduce_mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1
    return heatmap

def analyze_salt_distribution(binary_mask):
    """Analyze salt distribution and region"""
    h, w = binary_mask.shape
    vertical = "left" if np.sum(binary_mask[:, :w//2]) > np.sum(binary_mask[:, w//2:]) else "right"
    horizontal = "upper" if np.sum(binary_mask[:h//2, :]) > np.sum(binary_mask[h//2:, :]) else "lower"
    return f"{horizontal} {vertical}"

def get_geological_analysis(salt_percentage, region_text, classified_zone):
    """Generate geological analysis"""
    strength = (
        "strong" if salt_percentage >= 30 else
        "moderate" if salt_percentage >= 10 else
        "weak"
    )
    
    viability = (
        "Highly viable for extraction." if salt_percentage >= 40 else
        "Moderately viable with support." if salt_percentage >= 20 else
        "Not currently viable for extraction."
    )
    
    region_patterns = {
        "upper left": "continental shelf edges",
        "upper right": "folded tectonic margins", 
        "lower left": "subduction zones",
        "lower right": "oceanic trench peripheries"
    }
    similar_region = region_patterns.get(region_text, "offshore marine basin")
    
    return strength, viability, similar_region

# Load models
seg_model, clf_model = load_models()

if seg_model is None or clf_model is None:
    st.error("Failed to load models. Please check if model files exist.")
    st.stop()

# File uploader
uploaded_file = st.file_uploader(
    "Choose a seismic image",
    type=['png', 'jpg', 'jpeg'],
    help="Upload a grayscale seismic image for salt detection"
)

if uploaded_file is not None:
    # Preprocess image
    original_img, resized_img, input_img = preprocess_image(uploaded_file)
    
    if original_img is not None:
        # Create columns for layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Original Image")
            st.image(original_img, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            st.subheader("🔍 Processed Image")
            st.image(resized_img, caption=f"Resized to {IMG_SIZE}", use_container_width=True)
        
        # Process button
        if st.button("🚀 Analyze Salt Deposits", type="primary"):
            with st.spinner("Analyzing image..."):
                try:
                    # Segmentation inference
                    pred_mask = seg_model.predict(input_img, verbose=0)[0, :, :, 0]
                    binary_mask = (pred_mask > 0.5).astype(np.uint8)
                    salt_percentage = (np.sum(binary_mask) / binary_mask.size) * 100
                    
                    # Classification inference
                    pred_class = clf_model.predict(input_img, verbose=0)
                    class_label = np.argmax(pred_class)
                    class_names = ["Low salt zone", "Medium salt zone", "High salt zone"]
                    classified_zone = class_names[class_label]
                    
                    # Salt distribution analysis
                    region_text = analyze_salt_distribution(binary_mask)
                    
                    # Geological analysis
                    strength, viability, similar_region = get_geological_analysis(
                        salt_percentage, region_text, classified_zone
                    )
                    
                    # Generate Grad-CAM
                    last_conv_layer = next((layer.name for layer in reversed(seg_model.layers)
                                          if isinstance(layer, tf.keras.layers.Conv2D)), None)
                    
                    if last_conv_layer:
                        heatmap = get_gradcam_heatmap(seg_model, input_img, last_conv_layer)
                        heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
                        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
                        overlay = cv2.addWeighted(
                            cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR), 0.6, 
                            heatmap_colored, 0.4, 0
                        )
                    
                    # Display results
                    st.success("Analysis completed!")
                    
                    # Results section
                    st.subheader("📊 Analysis Results")
                    
                    # Metrics in columns
                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                    
                    with metric_col1:
                        st.metric("Salt Coverage", f"{salt_percentage:.1f}%")
                    
                    with metric_col2:
                        st.metric("Salt Strength", strength.capitalize())
                    
                    with metric_col3:
                        st.metric("Classification", classified_zone)
                    
                    with metric_col4:
                        st.metric("Region", region_text.title())
                    
                    # Visualizations
                    st.subheader("🖼️ Visualizations")
                    
                    vis_col1, vis_col2, vis_col3 = st.columns(3)
                    
                    with vis_col1:
                        st.write("**Predicted Salt Mask**")
                        st.image(binary_mask, caption="Binary Salt Mask", use_container_width=True)
                    
                    with vis_col2:
                        st.write("**Grad-CAM Heatmap**")
                        st.image(heatmap_resized, caption="Attention Heatmap", use_container_width=True)
                    
                    with vis_col3:
                        st.write("**Grad-CAM Overlay**")
                        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                        st.image(overlay_rgb, caption="Explainability Overlay", use_container_width=True)
                    
                    # Detailed analysis
                    st.subheader("🧠 Geological Analysis")
                    
                    analysis_text = f"""
                    **Salt Deposit Analysis:**
                    - **Strength**: {strength.capitalize()} salt presence ({salt_percentage:.2f}% coverage)
                    - **Location**: Concentrated in the {region_text} region
                    - **Classification**: {classified_zone}
                    - **Geological Pattern**: Characteristic of {similar_region}
                    
                    **Extraction Viability:**
                    {viability}
                    
                    **Explainability:**
                    The Grad-CAM heatmap shows which areas of the image the AI model focused on when making its prediction. 
                    Red/hot colors indicate regions the model considers most important for salt detection, while blue/cool 
                    colors show areas of lesser importance.
                    """
                    
                    st.markdown(analysis_text)
                    
                    # Download section
                    st.subheader("💾 Download Results")
                    
                    # Create downloadable images
                    download_col1, download_col2 = st.columns(2)
                    
                    with download_col1:
                        # Save mask as bytes
                        mask_img = Image.fromarray((binary_mask * 255).astype(np.uint8))
                        mask_bytes = io.BytesIO()
                        mask_img.save(mask_bytes, format='PNG')
                        mask_bytes.seek(0)
                        
                        st.download_button(
                            label="📥 Download Salt Mask",
                            data=mask_bytes,
                            file_name="predicted_salt_mask.png",
                            mime="image/png"
                        )
                    
                    with download_col2:
                        # Save overlay as bytes
                        overlay_img = Image.fromarray(overlay_rgb)
                        overlay_bytes = io.BytesIO()
                        overlay_img.save(overlay_bytes, format='PNG')
                        overlay_bytes.seek(0)
                        
                        st.download_button(
                            label="📥 Download Grad-CAM Overlay",
                            data=overlay_bytes,
                            file_name="gradcam_overlay.png", 
                            mime="image/png"
                        )
                
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")

# Sidebar with information
st.sidebar.header("ℹ️ About")
st.sidebar.markdown("""
**Salt Detection AI System**

This application uses Physics-Informed Neural Networks (PINNs) with UNet++ architecture to:

1. **Segment** salt deposits in seismic images
2. **Classify** salt zones (Low/Medium/High)
3. **Explain** predictions using Grad-CAM

**Models Used:**
- Segmentation: PINN UNet++
- Classification: UNet++ Classifier

**Upload Requirements:**
- Image format: PNG, JPG, JPEG
- Preferably grayscale seismic images
- Any size (will be resized to 128x128)
""")

st.sidebar.header("🔧 Technical Details")
st.sidebar.markdown("""
**Grad-CAM Explanation:**
- Red areas: High model attention
- Blue areas: Low model attention
- Shows what the AI "sees" when making predictions

**Salt Coverage Thresholds:**
- Strong: ≥30%
- Moderate: 10-30%
- Weak: <10%

**Extraction Viability:**
- Highly viable: ≥40% coverage
- Moderately viable: 20-40% coverage
- Not viable: <20% coverage
""")
