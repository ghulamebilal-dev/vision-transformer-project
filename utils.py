import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import cv2
from PIL import Image

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def plot_training_metrics(train_losses, val_losses, train_accs, val_accs, save_path='training_metrics.png'):
    """Plot training and validation metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(train_losses, label='Training Loss', alpha=0.7)
    ax1.plot(val_losses, label='Validation Loss', alpha=0.7)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(train_accs, label='Training Accuracy', alpha=0.7)
    ax2.plot(val_accs, label='Validation Accuracy', alpha=0.7)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, save_path='confusion_matrix.png'):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_attention_heatmap(image, attention_weights, patch_size=16):
    """Create attention heatmap overlay on original image."""
    # Convert image to numpy
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
        if image.shape[0] == 3:  # CHW to HWC
            image = image.transpose(1, 2, 0)
    
    # Normalize image
    image = (image - image.min()) / (image.max() - image.min())
    
    # Get attention weights for CLS token (excluding CLS token itself)
    # Use last layer attention
    last_layer_attn = attention_weights[-1]
    cls_attention = last_layer_attn[:, :, 0, 1:].mean(dim=1)  # Average over heads
    
    # Reshape to 2D
    num_patches = int(np.sqrt(cls_attention.shape[-1]))
    attention_map = cls_attention.reshape(-1, num_patches, num_patches)
    attention_map = attention_map.mean(dim=0)  # Average over batch
    
    # Resize attention map to match image size
    attention_map_resized = cv2.resize(attention_map.cpu().numpy(), 
                                     (image.shape[1], image.shape[0]))
    
    # Normalize attention map
    attention_map_resized = (attention_map_resized - attention_map_resized.min()) / \
                          (attention_map_resized.max() - attention_map_resized.min())
    
    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * attention_map_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = heatmap / 255.0
    
    # Overlay heatmap on image
    alpha = 0.5
    overlayed = image * (1 - alpha) + heatmap * alpha
    
    return overlayed, attention_map_resized

def save_model(model, optimizer, epoch, accuracy, path='best_model.pth'):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy
    }, path)

def load_model(model, optimizer=None, path='best_model.pth'):
    """Load model checkpoint."""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint.get('epoch', 0), checkpoint.get('accuracy', 0)

def get_data_transforms():
    """Get data augmentation transforms for training and validation."""
    from torchvision import transforms
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform