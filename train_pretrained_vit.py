import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
import timm
import argparse
import json
import os
from tqdm import tqdm

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

def train_pretrained_vit(num_epochs=30, batch_size=32, learning_rate=2e-5, model_name='vit_base_patch16_224'):
    """Fine-tune pretrained ViT from timm library."""
    
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
    
    # Pretrained ViT model
    num_classes = len(train_dataset.classes)
    print(f"Loading pretrained {model_name}...")
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training metrics
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_accuracy = 0.0
    
    print(f"Fine-tuning pretrained ViT on {len(train_dataset)} training samples")
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {train_dataset.classes}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
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
            save_model(model, optimizer, epoch, best_accuracy, 'pretrained_vit_model.pth')
            print(f"New best model saved with accuracy: {best_accuracy:.4f}")
        
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Best Val Acc: {best_accuracy:.4f}")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.2e}")
        print("-" * 50)
    
    # Plot training metrics
    plot_training_metrics(train_losses, val_losses, train_accs, val_accs, 
                         'pretrained_vit_training_metrics.png')
    
    # Save class names
    class_info = {
        'classes': train_dataset.classes,
        'class_to_idx': train_dataset.class_to_idx
    }
    with open('pretrained_vit_class_info.json', 'w') as f:
        json.dump(class_info, f, indent=2)
    
    print(f"Training completed! Best validation accuracy: {best_accuracy:.4f}")
    return model, train_dataset.classes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fine-tune Pretrained Vision Transformer')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--model', type=str, default='vit_base_patch16_224', 
                       choices=['vit_base_patch16_224', 'vit_small_patch16_224'],
                       help='Pretrained ViT model name')
    
    args = parser.parse_args()
    
    train_pretrained_vit(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        model_name=args.model
    )