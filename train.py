import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
import os
from tqdm import tqdm
import json

from vit_model import VisionTransformer
from hybrid_model import HybridCNNTransformer
from utils import AverageMeter, plot_training_metrics, save_model, get_data_transforms

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == labels).float().mean()
        
        # Update metrics
        losses.update(loss.item(), images.size(0))
        accuracies.update(accuracy.item(), images.size(0))
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{losses.avg:.4f}',
            'Acc': f'{accuracies.avg:.4f}'
        })
    
    return losses.avg, accuracies.avg

def validate_epoch(model, dataloader, criterion, device, epoch):
    """Validate for one epoch."""
    model.eval()
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Val]')
    
    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == labels).float().mean()
            
            # Update metrics
            losses.update(loss.item(), images.size(0))
            accuracies.update(accuracy.item(), images.size(0))
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'Acc': f'{accuracies.avg:.4f}'
            })
    
    return losses.avg, accuracies.avg

def train_model(model_type='vit', num_epochs=50, batch_size=32, learning_rate=1e-4):
    """Main training function."""
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data transforms
    train_transform, val_transform = get_data_transforms()
    
    # Dataset
    train_dataset = datasets.ImageFolder('data/train', transform=train_transform)
    val_dataset = datasets.ImageFolder('data/val', transform=val_transform)
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Model
    num_classes = len(train_dataset.classes)
    if model_type == 'vit':
        model = VisionTransformer(num_classes=num_classes)
        model_save_path = 'vit_model.pth'
    elif model_type == 'hybrid':
        model = HybridCNNTransformer(num_classes=num_classes)
        model_save_path = 'hybrid_model.pth'
    else:
        raise ValueError("Model type must be 'vit' or 'hybrid'")
    
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training metrics
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_accuracy = 0.0
    
    print(f"Training {model_type.upper()} model on {len(train_dataset)} training samples")
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {train_dataset.classes}")
    
    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device, epoch)
        
        # Update scheduler
        scheduler.step()
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Save best model
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            save_model(model, optimizer, epoch, best_accuracy, model_save_path)
            print(f"New best model saved with accuracy: {best_accuracy:.4f}")
        
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Best Val Acc: {best_accuracy:.4f}")
        print("-" * 50)
    
    # Plot training metrics
    plot_training_metrics(train_losses, val_losses, train_accs, val_accs, 
                         f'{model_type}_training_metrics.png')
    
    # Save class names
    class_info = {
        'classes': train_dataset.classes,
        'class_to_idx': train_dataset.class_to_idx
    }
    with open(f'{model_type}_class_info.json', 'w') as f:
        json.dump(class_info, f, indent=2)
    
    return model, train_dataset.classes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Vision Transformer Models')
    parser.add_argument('--model', type=str, default='vit', choices=['vit', 'hybrid'],
                       help='Model type to train')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    
    args = parser.parse_args()
    
    train_model(
        model_type=args.model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )