import os
import subprocess
import streamlit as st

# --------------------------------------------------------------------
# ✅ Ensure system dependencies & Git LFS model files are available
# --------------------------------------------------------------------
repo_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(repo_dir)

try:
   # st.info("Ensuring model files are available (using Git LFS)...")
    # Initialize and pull Git LFS
    subprocess.run(["git", "lfs", "install"], check=True)
    subprocess.run(["git", "lfs", "pull"], check=True)
    #st.success("✅ Model files pulled successfully using Git LFS.")
except subprocess.CalledProcessError as e:
    st.error(f"❌ Git LFS failed: {e.stderr}")
    st.error("Please ensure Git LFS is set up correctly in the repository.")
except FileNotFoundError:
    st.error("⚠️ Git LFS command not found. Make sure it's installed via packages.txt.")

# --------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import cv2  # OpenCV requires libGL (installed via packages.txt)

from vit_model import VisionTransformer
from hybrid_model import HybridCNNTransformer
from utils import create_attention_heatmap, get_data_transforms

# --------------------------------------------------------------------
# Streamlit Page Configuration
# --------------------------------------------------------------------
st.set_page_config(
    page_title="Vision Transformer Explorer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------------------------
# Cached Model Loader
# --------------------------------------------------------------------
@st.cache_resource
def load_model(model_type, model_path, class_info_path, device):
    """Load model with caching."""
    with open(class_info_path, 'r') as f:
        class_info = json.load(f)

    num_classes = len(class_info['classes'])

    if model_type == 'vit':
        model = VisionTransformer(num_classes=num_classes)
    elif model_type == 'hybrid':
        model = HybridCNNTransformer(num_classes=num_classes)
    elif model_type == 'pretrained_vit':
        import timm
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
    else:
        raise ValueError("Invalid model type")

    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model, class_info['classes']

# --------------------------------------------------------------------
# Prediction Function
# --------------------------------------------------------------------
def predict_image(image, model, class_names, device, model_type):
    """Make prediction and return attention visualization."""
    _, val_transform = get_data_transforms()

    input_tensor = val_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        if model_type == 'pretrained_vit':
            logits = model(input_tensor)
            attention_weights = None
            H, W = 14, 14
        elif model_type == 'vit':
            logits, attention_weights = model(input_tensor, return_attention=True)
            H, W = 14, 14
        else:  # hybrid
            logits, attention_weights, H, W = model(input_tensor, return_attention=True)

    probabilities = F.softmax(logits, dim=1)
    predicted_class = torch.argmax(logits, dim=1).item()
    confidence = probabilities[0, predicted_class].item()

    # Attention visualization
    if attention_weights is not None:
        heatmap_overlay, attention_map = create_attention_heatmap(
            input_tensor[0].cpu(), attention_weights, patch_size=16
        )
    else:
        heatmap_overlay, attention_map = None, None

    # Top-K predictions
    top_k = 5
    top_probs, top_indices = torch.topk(probabilities[0], top_k)
    top_classes = [class_names[i] for i in top_indices.cpu().numpy()]
    top_probabilities = top_probs.cpu().numpy()

    return {
        'predicted_class': class_names[predicted_class],
        'confidence': confidence,
        'top_predictions': list(zip(top_classes, top_probabilities)),
        'heatmap_overlay': heatmap_overlay,
        'attention_map': attention_map,
        'all_probabilities': probabilities[0].cpu().numpy()
    }

# --------------------------------------------------------------------
# Streamlit Main App
# --------------------------------------------------------------------
def main():
    st.title("🔍 Vision Transformer Image Classification Dashboard")
    st.markdown("Explore how Vision Transformers see and classify images with attention visualization")

    # Sidebar configuration
    st.sidebar.title("Configuration")

    model_type = st.sidebar.selectbox(
        "Select Model",
        ["vit", "hybrid", "pretrained_vit"],
        format_func=lambda x: {
            "vit": "Custom Vision Transformer",
            "hybrid": "Hybrid CNN + Transformer",
            "pretrained_vit": "Pretrained ViT (timm)"
        }[x]
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_configs = {
        "vit": {"path": "vit_model.pth", "class_info": "vit_class_info.json"},
        "hybrid": {"path": "hybrid_model.pth", "class_info": "hybrid_class_info.json"},
        "pretrained_vit": {"path": "pretrained_vit_model.pth", "class_info": "pretrained_vit_class_info.json"}
    }

    try:
        config = model_configs[model_type]
        model, class_names = load_model(model_type, config["path"], config["class_info"], device)
        st.sidebar.success(f"✅ {model_type.upper()} model loaded successfully!")
        st.sidebar.info(f"Number of classes: {len(class_names)}")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {e}")
        return

    # ----------------------------------------------------------------
    # Main content layout
    # ----------------------------------------------------------------
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📤 Upload Image")

        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image to classify and visualize attention"
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_column_width=True)

            if st.button("🔍 Analyze Image", type="primary"):
                with st.spinner("Analyzing image with Vision Transformer..."):
                    results = predict_image(image, model, class_names, device, model_type)

                st.header("📊 Prediction Results")

                st.metric(
                    label="Predicted Class",
                    value=results['predicted_class'],
                    delta=f"{results['confidence']:.1%} confidence"
                )

                st.progress(int(results['confidence'] * 100))
                st.caption(f"Confidence: {results['confidence']:.3f}")

                st.subheader("Top Predictions")
                for class_name, prob in results['top_predictions']:
                    col_a, col_b = st.columns([3, 1])
                    col_a.write(class_name)
                    col_b.write(f"{prob:.3f}")

    with col2:
        if uploaded_file is not None and 'results' in locals():
            st.header("👁️ Attention Visualization")

            if results['heatmap_overlay'] is not None:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

                ax1.imshow(image)
                ax1.set_title('Original Image')
                ax1.axis('off')

                ax2.imshow(results['heatmap_overlay'])
                ax2.set_title('Attention Heatmap')
                ax2.axis('off')

                st.pyplot(fig)

                st.subheader("Attention Analysis")
                attention_data = results['attention_map']
                if attention_data is not None:
                    st.write("**Attention Statistics:**")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Max Attention", f"{attention_data.max():.3f}")
                    c2.metric("Mean Attention", f"{attention_data.mean():.3f}")
                    c3.metric("Attention Std", f"{attention_data.std():.3f}")
            else:
                st.info("Attention visualization not available for this model")

    # Sidebar training metrics
    st.sidebar.markdown("---")
    st.sidebar.header("Training Metrics")

    plot_files = {
        "vit": "vit_training_metrics.png",
        "hybrid": "hybrid_training_metrics.png",
        "pretrained_vit": "pretrained_vit_training_metrics.png"
    }

    plot_file = plot_files.get(model_type)
    if plot_file and os.path.exists(plot_file):
        st.sidebar.image(plot_file, caption="Training Metrics", use_column_width=True)

    st.sidebar.markdown("---")
    st.sidebar.header("Model Comparison")

    if os.path.exists("results_comparison.csv"):
        df = pd.read_csv("results_comparison.csv")
        st.sidebar.dataframe(df.style.highlight_max(axis=0))

# --------------------------------------------------------------------
# Entrypoint
# --------------------------------------------------------------------
if __name__ == "__main__":
    main()
