import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import argparse
import json
from models import model_dict  # Assuming models are imported from your models.py


def hook_fn(module, input, output):
    global layer_outputs
    layer_outputs.append(output)

def register_hooks(model):
    for layer in model.children():
        if isinstance(layer, nn.Module):
            layer.register_forward_hook(hook_fn)

def perform_pca(features):
    pca = PCA(n_components=2)
    pca.fit(features)
    return pca.components_

def generate_heatmap(data, layer_idx):
    sns.heatmap(data, annot=True, fmt='.2f')
    plt.title(f'Layer {layer_idx + 1} Important Neurons')
    plt.show()

def generate_3d_plot(data, layer_idx):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2])
    ax.set_title(f'Layer {layer_idx + 1} Important Neurons in 3D')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PCA Analysis for Neural Network Layers')
    parser.add_argument('--model', type=str, required=True, help='Specify the model for analysis')
    parser.add_argument('--dataset', type=str, required=True, help='Specify the test dataset')
    args = parser.parse_args()

    model = model_dict[args.model]()
    model.eval()

    layer_outputs = []
    register_hooks(model)

    important_neurons_info = {}

    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            output = model(input)
            for idx, feature in enumerate(layer_outputs):
                feature = feature.reshape(feature.size(0), -1).cpu().numpy()
                important_features = perform_pca(feature)

                generate_heatmap(important_features, idx)
                generate_3d_plot(important_features, idx)

                layer_info = {
                    'important_neurons': important_features.tolist(),
                    'layer_type': str(type(model[idx])),
                    'weights': model[idx].weight.cpu().numpy().tolist() if hasattr(model[idx], 'weight') else 'N/A',
                    'activation_function': str(model[idx].activation) if hasattr(model[idx], 'activation') else 'N/A'
                }
                important_neurons_info[f'layer_{idx + 1}'] = layer_info

    with open('important_neurons_info.json', 'w') as f:
        json.dump(important_neurons_info, f, indent=4)
