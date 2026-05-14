import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from training.utils.data_loader import DeepfakeDataset
from training.utils.metrics import compute_metrics, print_metrics
from backend.app.ml.models import build_resnet18, build_efficientnet_b3, build_vit

def main(config_path, data_dir, model_name, checkpoint_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    test_dataset = DeepfakeDataset(data_dir, split="test", input_size=config['data']['input_size'])
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=config['data']['num_workers'])
    
    if model_name == "resnet18":
        model = build_resnet18().to(device)
    elif model_name == "efficientnet_b3":
        model = build_efficientnet_b3().to(device)
    elif model_name == "vit_base":
        model = build_vit().to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")
        
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"Evaluating {model_name}"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    print_metrics(metrics, prefix=f"Evaluation Metrics ({model_name})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='training/configs/default.yaml')
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--model', required=True, help="Model name (e.g. resnet18)")
    parser.add_argument('--checkpoint', required=True, help="Path to model checkpoint")
    args = parser.parse_args()
    main(args.config, args.data_dir, args.model, args.checkpoint)
