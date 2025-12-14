import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import your modules
from model import KGNN_Diabetes
from knowledge_graph import build_graph_tensors, get_patient_active_nodes, NUM_NODES, NODE_MAP
from data_preprocessing import process_data_for_model # assume you put the preprocessing function here, or copy it directly

# Configuration (must match training)
CONFIG = {
    "num_nodes": NUM_NODES,
    "num_relations": 3,
    "num_direct_features": 11,
    "hidden_dim": 32,
    "csv_path": "diabetes_binary_5050split_health_indicators_BRFSS2015.csv", # your csv path
    "model_path": "best_kgnn_model.pth" # trained model path
}

def load_model_and_data():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available(): device = torch.device("mps")
    
    # 1. Prepare the knowledge graph
    edge_index, edge_types = build_graph_tensors()
    edge_index = edge_index.to(device)
    edge_types = edge_types.to(device)

    # 2. Prepare data (take a small subset for analysis)
    print("Loading data...")
    df = pd.read_csv(CONFIG['csv_path'])
    # We only care about Diabetes in 3-class (Label=2) or 2-class (Label=1)
    # For demonstration, we take a small subset of the full dataset
    df_sample = df.sample(n=2000, random_state=42)
    
    # Reuse the previous data preprocessing logic
    # Note: ensure process_data_for_model can be imported here
    # If you didn't encapsulate it before, copy the function into this file
    p_masks, p_directs, labels = process_data_for_model(df_sample)
    
    num_classes = len(torch.unique(labels)) # Dynamically obtain number of classes

    # 3. Load model
    print(f"Loading model from {CONFIG['model_path']}...")
    model = KGNN_Diabetes(
        num_nodes=CONFIG['num_nodes'],
        num_relations=CONFIG['num_relations'],
        num_direct_features=CONFIG['num_direct_features'],
        num_classes=num_classes, # dynamic setting
        graph_hidden_dim=CONFIG['hidden_dim']
    ).to(device)
    
    model.load_state_dict(torch.load(CONFIG['model_path'], map_location=device))
    model.eval()
    
    return model, p_masks, p_directs, labels, edge_index, edge_types, device

def analyze_attention(model, masks, directs, labels, edge_index, edge_types, device):
    """
    Core analysis function
    """
    masks = masks.to(device)
    directs = directs.to(device)
    
    with torch.no_grad():
        # Get predictions and attention weights
        logits, attn_weights = model(edge_index, edge_types, masks, directs)
        preds = torch.argmax(logits, dim=1)
        
    # attn_weights shape: [Batch_Size, Num_Nodes] (already through Softmax)
    attn_weights = attn_weights.cpu().numpy()
    preds = preds.cpu().numpy()
    true_labels = labels.numpy()
    
    # --- Analysis 1: Global Importance ---
    # Find all samples predicted as 'Diabetic' (assuming label=1 or 2 indicates diabetes, take the maximum)
    diabetes_label = np.max(true_labels) 
    diabetes_indices = np.where(preds == diabetes_label)[0]
    
    if len(diabetes_indices) > 0:
        print(f"\nAnalyzing {len(diabetes_indices)} patients predicted as Diabetic...")
        
        # Compute the average attention distribution for these patients
        avg_weights = np.mean(attn_weights[diabetes_indices], axis=0)
        
        # Plot
        plot_global_importance(avg_weights)
    else:
        print("No diabetes predictions found in sample.")

    # --- Analysis 2: Individual Case Study ---
    # Find a patient who is a true positive (predicted and actual diabetic)
    tp_indices = np.where((preds == diabetes_label) & (true_labels == diabetes_label))[0]
    
    if len(tp_indices) > 0:
        # Pick the first true positive
        patient_idx = tp_indices[0] 
        print(f"\nCase Study: Patient {patient_idx} (True Positive)")
        
        patient_weight = attn_weights[patient_idx]
        patient_mask = masks[patient_idx].cpu().numpy()
        
        plot_local_importance(patient_weight, patient_mask, patient_id=patient_idx)

def plot_global_importance(avg_weights):
    node_names = [NODE_MAP[i] for i in range(len(avg_weights))]
    
    plt.figure(figsize=(12, 6))
    # Sort
    indices = np.argsort(avg_weights)[::-1]
    sorted_weights = avg_weights[indices]
    sorted_names = [node_names[i] for i in indices]
    
    sns.barplot(x=sorted_weights, y=sorted_names, palette="viridis")
    plt.title("Global Feature Importance (Graph Attention) for Diabetic Predictions")
    plt.xlabel("Average Attention Weight")
    plt.ylabel("Medical Concept Node")
    plt.tight_layout()
    plt.savefig("global_interpretability.png")
    plt.show()
    print("Saved global_interpretability.png")

def plot_local_importance(weights, mask, patient_id):
    node_names = [NODE_MAP[i] for i in range(len(weights))]
    
    # Only highlight nodes actually activated for the patient (Mask=1)
    # Or show all nodes to check whether the model attends to inactive nodes (theoretically masked attention should be 0)
    
    plt.figure(figsize=(10, 5))
    colors = ['red' if m==1 else 'gray' for m in mask] # Red indicates the patient actually has this condition
    
    plt.bar(node_names, weights, color=colors)
    plt.xticks(rotation=45, ha='right')
    plt.title(f"Why Patient {patient_id} was classified as Diabetic?\n(Red = Patient has this condition)")
    plt.ylabel("Attention Weight")
    plt.tight_layout()
    plt.savefig(f"patient_{patient_id}_case_study.png")
    plt.show()
    print(f"Saved patient_{patient_id}_case_study.png")

if __name__ == "__main__":
    model, masks, directs, labels, edge_index, edge_types, device = load_model_and_data()
    analyze_attention(model, masks, directs, labels, edge_index, edge_types, device)