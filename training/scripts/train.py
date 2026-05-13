import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from training.utils.data_loader import DeepfakeDataset
from training.utils.early_stopping import EarlyStopping
from training.utils.metrics import compute_metrics, print_metrics
from backend.app.ml.models import build_resnet18, build_efficientnet_b3, build_vit

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(all_labels, all_preds)
    return epoch_loss, metrics

def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    return epoch_loss, metrics

def main(config_path, data_dir):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_dataset = DeepfakeDataset(data_dir, split="train", input_size=config['data']['input_size'])
    val_dataset = DeepfakeDataset(data_dir, split="val", input_size=config['data']['input_size'])
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['data']['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=config['data']['num_workers'])
    
    models_to_train = config['model']['models']
    os.makedirs('models/checkpoints', exist_ok=True)
    
    for model_name in models_to_train:
        print(f"\n--- Training {model_name} ---")
        if model_name == "resnet18":
            model = build_resnet18().to(device)
        elif model_name == "efficientnet_b3":
            model = build_efficientnet_b3().to(device)
        elif model_name == "vit_base":
            model = build_vit().to(device)
        else:
            print(f"Unknown model: {model_name}")
            continue
            
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
        early_stopping = EarlyStopping(patience=config['training']['patience'], mode="min")
        
        for epoch in range(config['training']['epochs']):
            print(f"\nEpoch {epoch+1}/{config['training']['epochs']}")
            train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device)
            
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print_metrics(val_metrics, prefix="Validation Metrics")
            
            if early_stopping(val_loss):
                print("Early stopping triggered.")
                break
                
            # Save checkpoint if it's the best model so far
            if early_stopping.counter == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_metrics': val_metrics
                }, f"models/checkpoints/{model_name}_best.pth")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='training/configs/default.yaml')
    parser.add_argument('--data_dir', default='data')
    args = parser.parse_args()
    main(args.config, args.data_dir)
