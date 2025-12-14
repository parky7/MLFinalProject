import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, balanced_accuracy_score, confusion_matrix, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import time
import wandb
import torch.nn.functional as F

# Import your custom modules
from model import KGNN_Diabetes
from knowledge_graph import build_graph_tensors, get_patient_active_nodes, NUM_NODES
from data_preprocessing import process_data_for_model

# ==========================================
# 1. Dataset definition (Dataset Class)
# ==========================================
class DiabetesDataset(Dataset):
    def __init__(self, patient_masks, direct_features, labels):
        """
        Args:
        - patient_masks: input for the graph branch (0/1 matrix)
        - direct_features: input for the MLP branch (numeric features)
        - labels: target labels (0, 1, 2)
        """
        self.patient_masks = patient_masks
        self.direct_features = direct_features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.patient_masks[idx], 
                self.direct_features[idx], 
                self.labels[idx])

# ==========================================
# 2. Data preprocessing logic
# ==========================================
# Define direct input feature columns (matching the previous assignment)

# import from process_data_for_model

# ==========================================
# 3. Training and evaluation functions
# ==========================================
def train_epoch(model, dataloader, criterion, optimizer, edge_index, edge_types, device):
    model.train()
    total_loss = 0
    
    for batch_masks, batch_direct, batch_labels in dataloader:
        # Move to GPU/MPS/CPU
        batch_masks = batch_masks.to(device)
        batch_direct = batch_direct.to(device)
        batch_labels = batch_labels.to(device)
        
        # Forward pass
        # Note: edge_index and edge_types are static full-graph structures and do not need batching
        optimizer.zero_grad()
        logits, _ = model(edge_index, edge_types, batch_masks, batch_direct)
        
        # Compute loss
        loss = criterion(logits, batch_labels)
        
        # Backpropagate
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

# def evaluate(model, dataloader, num_classes, edge_index, edge_types, device):
#     model.eval()
#     all_preds = []
#     all_labels = []
#
#     if num_classes == 3:
#         current_names = ['No Dia', 'Pre-Dia', 'Diabetes']
#     else:
#         current_names = ['No Dia', 'Diabetes']
#     
#     with torch.no_grad():
#         for batch_masks, batch_direct, batch_labels in dataloader:
#             batch_masks = batch_masks.to(device)
#             batch_direct = batch_direct.to(device)
#             batch_labels = batch_labels.to(device)
#             
#             logits, _ = model(edge_index, edge_types, batch_masks, batch_direct)
#             preds = torch.argmax(logits, dim=1)
#             
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(batch_labels.cpu().numpy())
#             
#     # Compute evaluation metrics
#     print("\n" + "="*40)
#     print("EVALUATION REPORT")
#     print("="*40)
#     
#     # 1. Basic metrics
#     bal_acc = balanced_accuracy_score(all_labels, all_preds)
#     acc = accuracy_score(all_labels, all_preds)
#     print(f"Eval Balanced Acc: {bal_acc:.4f}") # Still print key info to terminal
#     
#     # 2. Generate detailed report (return as dict with output_dict=True)
#     text_report = classification_report(all_labels, all_preds, 
#                                    target_names=current_names, 
#                                    zero_division=0)
#     print(text_report)
#
#     # 3. Confusion matrix
#     cm = confusion_matrix(all_labels, all_preds)
#     print("Confusion Matrix:\n", cm)
#     
#
#     # 4. Build dictionary to return to WandB
#     # We can log recall for each class, which you care about most
#     report = classification_report(all_labels, all_preds, 
#                                 target_names=current_names, 
#                                 output_dict=True,
#                                 zero_division=0)
#     metrics = {
#         "val_loss": 0, # val_loss calculation can be extended here, currently missing
#         "val_balanced_acc": bal_acc,
#         "val_accuracy": acc,
#         "val_macro_f1": report['macro avg']['f1-score'],
#     }
#     # 1. No Diabetes (always present)
#     if 'No Dia' in report:
#         metrics["recall_no_diabetes"] = report['No Dia']['recall']
#         
#     # 2. Diabetes (always present)
#     if 'Diabetes' in report:
#         metrics["recall_diabetes"] = report['Diabetes']['recall'] # <--- focus on this curve
#         metrics["precision_diabetes"] = report['Diabetes']['precision']
#
#     # 3. Pre-Diabetes (only exists for 3-class)
#     if 'Pre-Dia' in report:
#         metrics["recall_pre_diabetes"] = report['Pre-Dia']['recall']
#     
#     return metrics

def evaluate(model, dataloader, num_classes, edge_index, edge_types, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = [] # <--- Added: store probabilities (for ROC-AUC)
    
    # 1. Start timing
    start_time = time.time()
    
    with torch.no_grad():
        for batch_masks, batch_direct, batch_labels in dataloader:
            batch_masks = batch_masks.to(device)
            batch_direct = batch_direct.to(device)
            batch_labels = batch_labels.to(device)
            
            logits, _ = model(edge_index, edge_types, batch_masks, batch_direct)
            
            # Get predicted class
            preds = torch.argmax(logits, dim=1)
            # Get predicted probabilities (needed for ROC-AUC)
            probs = F.softmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy()) # store probabilities
            
    # 2. End timing & compute average time
    end_time = time.time()
    total_time_ms = (end_time - start_time) * 1000
    avg_time_ms = total_time_ms / len(all_labels) # ms/sample
    
    # -------------------------------------------------------
    # 3. Compute all metrics
    # -------------------------------------------------------
    print("\n" + "="*40)
    print("EVALUATION REPORT")
    print("="*40)
    
    # Basic metrics
    acc = accuracy_score(all_labels, all_preds)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    
    # Precision/Recall/F1 
    # Use 'binary' mode to focus on positive class (1=Diabetes), suitable for your 50-50 dataset
    # If multi-class, the code will automatically switch to 'weighted'
    avg_method = 'binary' if num_classes == 2 else 'weighted'
    
    prec = precision_score(all_labels, all_preds, average=avg_method, zero_division=0)
    rec = recall_score(all_labels, all_preds, average=avg_method, zero_division=0)
    f1 = f1_score(all_labels, all_preds, average=avg_method, zero_division=0)
    
    # ROC-AUC calculation
    try:
        if num_classes == 2:
            # Binary: take probability for positive class (column index 1)
            roc_auc = roc_auc_score(all_labels, np.array(all_probs)[:, 1])
        else:
            # Multi-class: need to specify multi_class parameter
            roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    except Exception as e:
        print(f"Warning: ROC-AUC calculation failed ({e})")
        roc_auc = 0.0

    # Print results (for terminal viewing)
    print(f"Prediction Time: {avg_time_ms:.2f} ms/sample")
    print(f"Accuracy:        {acc:.4f}")
    print(f"Balanced Acc:    {bal_acc:.4f}")
    print(f"Precision:       {prec:.4f}")
    print(f"Recall:          {rec:.4f}")
    print(f"F1 Score:        {f1:.4f}")
    print(f"ROC-AUC:         {roc_auc:.4f}")
    
    # -------------------------------------------------------
    # 4. Build return dictionary (for WandB)
    # -------------------------------------------------------
    metrics = {
        "val_loss": 0, 
        "val_accuracy": acc,
        "val_balanced_acc": bal_acc,
        "prediction_time_ms": avg_time_ms,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        
        # Keep previous per-class recall in case it's needed
        "recall_diabetes": rec # For binary, Recall equals Recall(Diabetes)
    }
    
    return metrics
    

# ==========================================
# 4. Main program
# ==========================================
def main():
    # Configuration
    config = {
        "learning_rate": 0.001,
        "batch_size": 64,
        "epochs": 20,
        "sample_size": None, # Set to 1000 for quick tests, None for full dataset
        "dropout": 0.3,
        "hidden_dim": 32,
        # 'cvs_path': "diabetes_binary_5050split_health_indicators_BRFSS2015.csv",
        "cvs_path": "diabetes_binary_health_indicators_BRFSS2015.csv",
        "dataset": "Diabetes_Kaggle"
    }

    # --- Initialize WandB ---
    wandb.init(project="diabetes-binary-kgnn-prediction", config=config)
    # Get config (allow wandb sweeps to override parameters)
    cfg = wandb.config
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available(): # Mac M1/M2 support
        device = torch.device("mps")
    print(f"Using device: {device}")

    # -------------------------------------------------------
    # Step 1: Prepare data
    # -------------------------------------------------------
    try:
        # Try to read data
        p_masks, p_directs, labels = process_data_for_model(pd.read_csv(cfg.cvs_path), sample_size=cfg.sample_size)
    except FileNotFoundError:
        print(f"Error: {cfg.cvs_path} not found. Generating DUMMY data for testing...")
    
    num_classes = len(torch.unique(labels))
    wandb.config.update({"num_classes": num_classes}, allow_val_change=True)

    # Split train/test sets (Stratified split ensures rare classes appear in validation set)
    # Need to split Masks, Directs, Labels
    X_idx = np.arange(len(labels))
    train_idx, val_idx = train_test_split(X_idx, test_size=0.2, stratify=labels, random_state=42)
    
    train_dataset = DiabetesDataset(p_masks[train_idx], p_directs[train_idx], labels[train_idx])
    val_dataset = DiabetesDataset(p_masks[val_idx], p_directs[val_idx], labels[val_idx])
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    # -------------------------------------------------------
    # Step 2: Prepare knowledge graph (static)
    # -------------------------------------------------------
    edge_index, edge_types = build_graph_tensors()
    edge_index = edge_index.to(device)
    edge_types = edge_types.to(device)

    # -------------------------------------------------------
    # Step 3: Initialize model and class weights
    # -------------------------------------------------------
    model = KGNN_Diabetes(
        num_nodes=NUM_NODES,
        num_relations=3,
        num_direct_features=11, # Must match number of direct feature columns
        num_classes=cfg.num_classes,
        dropout_rate=cfg.dropout,
        graph_hidden_dim=cfg.hidden_dim
    ).to(device)

    # --- Monitor model gradients (optional) ---
    wandb.watch(model, log="all", log_freq=10)

    # Compute class weights to handle imbalance
    # Logic: fewer samples -> larger weight
    class_counts = torch.bincount(labels[train_idx])
    total_samples = len(train_idx)
    # Classic inverse frequency weight calculation: Total / (Num_Classes * Count)
    class_weights = total_samples / (cfg.num_classes * class_counts.float())
    class_weights = class_weights.to(device)
    
    print(f"Class Weights Computed: {class_weights}") 
    # Expected: 'No Diabetes' lowest weight, 'Pre-Diabetes' highest

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4) # L2 regularization

    # -------------------------------------------------------
    # Step 4: Training loop
    # -------------------------------------------------------
    print("\nStarting Training...")
    best_acc = 0.0
    best_score = -1.0
    save_path = "best_kgnn_bin_model_metric.pth"
    
    for epoch in range(cfg.epochs):
        start_time = time.time()
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, edge_index, edge_types, device)
        val_metrics = evaluate(model, val_loader, num_classes, edge_index, edge_types, device)
        
        # --- Log to WandB ---
        # Merge train_loss and val_metrics into one dict and upload
        log_dict = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            **val_metrics # Unpack val_metrics: put val_balanced_acc etc. into the log
        }
        wandb.log(log_dict)
        
        print(f"Epoch {epoch+1} | Loss: {train_loss:.4f} | Recall(Dia): {val_metrics['recall_diabetes']:.2f}")

        current_score = val_metrics['val_balanced_acc']
        if current_score > best_score:
            best_score = current_score
            # Save only state_dict (recommended), file is smaller
            torch.save(model.state_dict(), save_path)
            print(f">>> [Saved] New Best Model! Balanced Acc: {best_score:.4f}")

    wandb.finish() # Finish run
    

if __name__ == "__main__":
    main()