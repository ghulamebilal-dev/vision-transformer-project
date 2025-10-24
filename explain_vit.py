import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import argparse
import json
import os

from vit_model import VisionTransformer
from hybrid_model import HybridCNNTransformer
from utils import create_attention_heatmap, get_data_transforms

def load_model_and_classes(model_type, model_path, class_info_path, device):
    """Load trained model and class information."""
    
    with open(class_info_path, 'r') as f:
        class_info = json.load(f)
    
    num_classes = len(class_info['classes'])
    
    if model_type == 'vit':
        model = VisionTransformer(num_classes=num_classes)
    elif model_type == 'hybrid':
        model = HybridCNNTransformer(num_classes=num_classes)
    else:
        raise ValueError("Model type must be 'vit' or 'hybrid'")
    
    # Load model weights
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model, class_info['classes'], class_info['class_to_idx']

def explain_prediction(image_path, model, class_names, device, model_type='vit'):
    """Generate prediction and attention visualization for an image."""
    
    # Load and preprocess image
    _, val_transform = get_data_transforms()
    image = Image.open(image_path).convert('RGB')
    original_image = np.array(image)
    
    # Transform image
    input_tensor = val_transform(image).unsqueeze(0).to(device)
    
    # Get prediction and attention
    with torch.no_grad():
        if model_type == 'vit':
            logits, attention_weights = model(input_tensor, return_attention=True)
            H, W = 14, 14  # For ViT with patch size 16
        else:
            logits, attention_weights, H, W = model(input_tensor, return_attention=True)
    
    # Get prediction
    probabilities = F.softmax(logits, dim=1)
    predicted_class = torch.argmax(logits, dim=1).item()
    confidence = probabilities[0, predicted_class].item()
    
    # Create attention heatmap
    heatmap_overlay, attention_map = create_attention_heatmap(
        input_tensor[0].cpu(), attention_weights, patch_size=16
    )
    
    # Resize original image to match heatmap if needed
    if original_image.shape[:2] != heatmap_overlay.shape[:2]:
        original_image = cv2.resize(original_image, 
                                  (heatmap_overlay.shape[1], heatmap_overlay.shape[0]))
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original image
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Attention heatmap
    axes[0, 1].imshow(heatmap_overlay)
    axes[0, 1].set_title('Attention Heatmap')
    axes[0, 1].axis('off')
    
    # Raw attention map
    im = axes[1, 0].imshow(attention_map, cmap='hot')
    axes[1, 0].set_title('Raw Attention Map')
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0])
    
    # Prediction probabilities
    top_k = 5
    top_probs, top_indices = torch.topk(probabilities[0], top_k)
    
    classes = [class_names[i] for i in top_indices.cpu().numpy()]
    probabilities_list = top_probs.cpu().numpy()
    
    y_pos = np.arange(len(classes))
    axes[1, 1].barh(y_pos, probabilities_list, align='center')
    axes[1, 1].set_yticks(y_pos)
    axes[1, 1].set_yticklabels(classes)
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_xlabel('Probability')
    axes[1, 1].set_title('Top Predictions')
    
    # Add probability values on bars
    for i, v in enumerate(probabilities_list):
        axes[1, 1].text(v + 0.01, i, f'{v:.3f}', va='center')
    
    plt.suptitle(f'Prediction: {class_names[predicted_class]} (Confidence: {confidence:.3f})', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig, predicted_class, confidence, attention_weights

def main():
    parser = argparse.ArgumentParser(description='Explain ViT Predictions with Attention Maps')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model_type', type=str, default='vit', choices=['vit', 'hybrid'],
                       help='Type of model to use')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model weights')
    parser.add_argument('--class_info', type=str, required=True, help='Path to class info JSON')
    parser.add_argument('--output', type=str, default='attention_visualization.png', 
                       help='Output path for visualization')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model and classes
    model, class_names, _ = load_model_and_classes(
        args.model_type, args.model_path, args.class_info, device
    )
    
    # Generate explanation
    fig, predicted_class, confidence, _ = explain_prediction(
        args.image, model, class_names, device, args.model_type
    )
    
    # Save visualization
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {args.output}")
    print(f"Prediction: {class_names[predicted_class]} (Confidence: {confidence:.3f})")
    
    plt.show()

if __name__ == "__main__":
    main()