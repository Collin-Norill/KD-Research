import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import random
import h5py
from torch.utils.data import Subset, DataLoader
from captum.attr import DeepLift, Saliency, IntegratedGradients, NoiseTunnel, visualization
from dataset.cifar100 import get_cifar100_dataloaders

# Assuming model_dict is defined in your models.py
# from models import model_dict

model_dict = {
    'ResNet32x4': ResNet32x4,
    # Add other models here
}

global layer_outputs
layer_outputs = {}

def hook_fn(module, input, output):
    layer_name = str(module)
    layer_outputs[layer_name] = output.cpu().detach().numpy()

def register_hooks(model):
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Module):
            layer.register_forward_hook(hook_fn)

def save_heatmap_to_disk(heatmap, layer_idx):
    with h5py.File('heatmaps.h5', 'a') as hf:
        hf.create_dataset(f'layer_{layer_idx}', data=heatmap)

def load_heatmaps_from_disk():
    heatmaps = []
    with h5py.File('heatmaps.h5', 'r') as hf:
        for i in range(len(hf.keys())):
            heatmaps.append(hf[f'layer_{i}'][:])
    return heatmaps

def generate_3d_heatmap():
    heatmaps = load_heatmaps_from_disk()
    # Your logic to stack 2D heatmaps into a 3D heatmap

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural Network Analysis')
    parser.add_argument('--model', type=str, required=True, help='Specify the model for analysis')
    parser.add_argument('--model_type', type=str, required=True, help='Specify the model type (e.g., resnet32x4)')
    parser.add_argument('--method', type=str, default='DeepLIFT', choices=['DeepLIFT', 'SHAP', 'Grad-CAM'], help='Method for interpretability')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model_dict.get(args.model_type, None)
    if model is None:
        print(f'Model type {args.model_type} not found in model_dict.')
        exit(1)

    model = model(num_classes=100).to(device)
    model_path = args.model
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    register_hooks(model)

    train_loader, val_loader = get_cifar100_dataloaders(batch_size=32, num_workers=8)

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input, target = input.to(device), target.to(device)
            output = model(input)

            for idx, (layer_name, feature) in enumerate(layer_outputs.items()):
                # Your logic to calculate importance using DeepLIFT, SHAP, or Grad-CAM
                # For example, using DeepLIFT
                if args.method == 'DeepLIFT':
                    dl = DeepLift(model)
                    attributions = dl.attribute(input, target=target)
                    heatmap = np.mean(attributions.cpu().detach().numpy(), axis=(0, 2, 3))

                save_heatmap_to_disk(heatmap, idx)

    generate_3d_heatmap()
