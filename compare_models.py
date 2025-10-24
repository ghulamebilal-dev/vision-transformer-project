import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
import json
import os

from vit_model import VisionTransformer
from hybrid_model import HybridCNNTransformer
from utils import get_data_transforms

def evaluate_model(model, dataloader, device, model_type='standard'):
    """Evaluate model accuracy and return detailed metrics."""
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"Evaluating {model_type}"):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            if model_type == 'hybrid':
                outputs = model(images)
            elif model_type == 'pretrained_vit':
                outputs = model(images)
            else:  # standard vit
                outputs = model(images)
            
            # Get predictions
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store for detailed analysis
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    accuracy = correct / total
    return accuracy, all_predictions, all_labels, all_probabilities

def count_parameters(model):
    """Count total and trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def load_model(model_config, device):
    """Load model based on configuration."""
    model_type = model_config['type']
    
    # Load class info
    with open(model_config['class_info'], 'r') as f:
        class_info = json.load(f)
    
    num_classes = len(class_info['classes'])
    
    # Initialize model
    if model_type == 'vit':
        model = VisionTransformer(num_classes=num_classes)
    elif model_type == 'hybrid':
        model = HybridCNNTransformer(num_classes=num_classes)
    elif model_type == 'pretrained_vit':
        import timm
        model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load weights
    checkpoint = torch.load(model_config['model_path'], map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model, class_info['classes']

def compare_models():
    """Compare all trained models comprehensively."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("=" * 60)
    
    # Load validation data
    _, val_transform = get_data_transforms()
    val_dataset = datasets.ImageFolder('data/val', transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Model configurations
    models_config = [
        {
            'name': 'Custom Vision Transformer',
            'type': 'vit',
            'model_path': 'vit_model.pth',
            'class_info': 'vit_class_info.json',
            'description': 'ViT implemented from scratch'
        },
        {
            'name': 'Hybrid CNN+Transformer', 
            'type': 'hybrid',
            'model_path': 'hybrid_model.pth',
            'class_info': 'hybrid_class_info.json',
            'description': 'ResNet18 features + Transformer encoder'
        }
    ]
    
    # Add pretrained ViT if available
    if os.path.exists('pretrained_vit_model.pth') and os.path.exists('pretrained_vit_class_info.json'):
        models_config.append({
            'name': 'Pretrained ViT (Fine-tuned)',
            'type': 'pretrained_vit', 
            'model_path': 'pretrained_vit_model.pth',
            'class_info': 'pretrained_vit_class_info.json',
            'description': 'ViT-Base from timm, fine-tuned'
        })
    
    results = []
    detailed_results = []
    
    print(f"Evaluating on {len(val_dataset)} validation samples")
    print(f"Number of classes: {len(val_dataset.classes)}")
    print("=" * 60)
    
    for config in models_config:
        print(f"\n🔍 Evaluating: {config['name']}")
        print(f"   Description: {config['description']}")
        
        try:
            # Check if model files exist
            if not os.path.exists(config['model_path']):
                print(f"   ⚠️  Model file not found: {config['model_path']}")
                continue
            if not os.path.exists(config['class_info']):
                print(f"   ⚠️  Class info not found: {config['class_info']}")
                continue
            
            # Load model
            model, class_names = load_model(config, device)
            
            # Count parameters
            total_params, trainable_params = count_parameters(model)
            
            # Evaluate
            accuracy, predictions, labels, probabilities = evaluate_model(
                model, val_loader, device, config['type']
            )
            
            # Calculate per-class accuracy
            class_correct = {class_name: 0 for class_name in class_names}
            class_total = {class_name: 0 for class_name in class_names}
            
            for i, (pred, true) in enumerate(zip(predictions, labels)):
                class_name = class_names[true]
                class_total[class_name] += 1
                if pred == true:
                    class_correct[class_name] += 1
            
            class_accuracy = {cls: class_correct[cls]/class_total[cls] for cls in class_names}
            
            # Store results
            results.append({
                'Model': config['name'],
                'Accuracy': f"{accuracy:.3f}",
                'Total Parameters': f"{total_params:,}",
                'Trainable Parameters': f"{trainable_params:,}",
                'Type': config['type']
            })
            
            # Store detailed results
            detailed_results.append({
                'model': config['name'],
                'accuracy': accuracy,
                'total_params': total_params,
                'trainable_params': trainable_params,
                'type': config['type'],
                'class_accuracy': class_accuracy,
                'description': config['description']
            })
            
            print(f"   ✅ Accuracy: {accuracy:.3f}")
            print(f"   📊 Parameters: {total_params:,} total, {trainable_params:,} trainable")
            
            # Print worst performing classes
            worst_classes = sorted(class_accuracy.items(), key=lambda x: x[1])[:3]
            print(f"   ⚠️  Most challenging classes: {', '.join([f'{cls}({acc:.2f})' for cls, acc in worst_classes])}")
            
        except Exception as e:
            print(f"   ❌ Error evaluating {config['name']}: {e}")
            continue
    
    # Create and save results DataFrame
    if results:
        df = pd.DataFrame(results)
        df.to_csv('results_comparison.csv', index=False)
        
        # Save detailed results
        detailed_df = pd.DataFrame([{
            'Model': r['model'],
            'Accuracy': r['accuracy'],
            'Total_Parameters': r['total_params'],
            'Type': r['type'],
            'Description': r['description']
        } for r in detailed_results])
        detailed_df.to_csv('detailed_results_comparison.csv', index=False)
        
        # Display results
        print("\n" + "=" * 80)
        print("📊 MODEL COMPARISON RESULTS")
        print("=" * 80)
        print(df.to_string(index=False))
        
        # Find best model
        best_model = max(detailed_results, key=lambda x: x['accuracy'])
        print(f"\n🏆 BEST MODEL: {best_model['model']}")
        print(f"   Accuracy: {best_model['accuracy']:.3f}")
        print(f"   Parameters: {best_model['total_params']:,}")
        
        # Create summary
        print(f"\n📈 SUMMARY:")
        print(f"   Total models evaluated: {len(results)}")
        print(f"   Best accuracy: {best_model['accuracy']:.3f}")
        print(f"   Average accuracy: {sum(r['accuracy'] for r in detailed_results)/len(detailed_results):.3f}")
        
        return df, detailed_results
    else:
        print("❌ No models were successfully evaluated!")
        return None, None

if __name__ == "__main__":
    compare_models()