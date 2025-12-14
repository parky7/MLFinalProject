import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

class KGNN_Diabetes(nn.Module):
    def __init__(self, 
                 num_nodes=16,          # Number of graph nodes (0-15)
                 num_relations=3,       # Number of relation types (0:Causes, 1:Associated, 2:Is_A)
                 num_direct_features=11,# Number of direct numeric input features (Age, Sex, Income...)
                 graph_hidden_dim=32,   # Hidden dimension for graph neural network
                 direct_hidden_dim=16,  # Encoding dimension for direct features
                 num_classes=3,         # 0:No, 1:Pre, 2:Diabetes
                 dropout_rate=0.3):
        
        super(KGNN_Diabetes, self).__init__()

        # ==================================================================================
        # Branch A: Knowledge Graph Reasoning (R-GCN)
        # Corresponds to Section 6.3.2 in the report
        # ==================================================================================
        
        # 1. Initial node embeddings (lookup table)
        # Each medical concept (e.g., Obesity) starts as a random vector and is updated during training
        self.node_embedding = nn.Embedding(num_nodes, graph_hidden_dim)
        
        # 2. R-GCN Layer 1
        # Aggregate neighbor information and distinguish relation types (relation-specific weights)
        self.rgcn1 = RGCNConv(graph_hidden_dim, graph_hidden_dim, num_relations=num_relations)
        
        # 3. R-GCN Layer 2
        # Deeper semantic reasoning
        self.rgcn2 = RGCNConv(graph_hidden_dim, graph_hidden_dim, num_relations=num_relations)
        
        # 4. Patient-Specific Attention Mechanism
        # Corresponds to Eq. (4) in the report
        # Computes importance weight alpha of each node for the current patient
        self.attention_net = nn.Sequential(
            nn.Linear(graph_hidden_dim, 1),
            nn.Tanh()
        )

        # ==================================================================================
        # Branch B: Direct Feature Encoding (MLP)
        # Handles numeric features such as Age, Sex, Income, BMI (raw)
        # ==================================================================================
        self.direct_encoder = nn.Sequential(
            nn.Linear(num_direct_features, direct_hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(direct_hidden_dim) # Keeps numerical stability
        )

        # ==================================================================================
        # Fusion & Classification
        # Corresponds to Eq. (8) in the report
        # ==================================================================================
        
        # Fusion dimension = graph vector dim + direct feature dim
        fusion_dim = graph_hidden_dim + direct_hidden_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, edge_index, edge_types, patient_masks, direct_features):
        """
        Forward pass logic
        
        Parameters:
        - edge_index, edge_types: Static knowledge graph structure (shared by all patients)
        - patient_masks: [Batch, Num_Nodes] binary mask indicating which nodes are activated for each patient
        - direct_features: [Batch, 11] standardized numeric features
        """
        
        # --- Step 1: Global graph reasoning ---
        # This step is independent of individual patients; it updates the representations of medical concepts
        # e.g., letting 'Obesity' absorb information from 'Sedentary'
        x = self.node_embedding.weight # Get current vectors for all nodes
        x = F.relu(self.rgcn1(x, edge_index, edge_types))
        x = F.dropout(x, p=0.2, training=self.training)
        node_embeds = self.rgcn2(x, edge_index, edge_types) # Output: [Num_Nodes, Graph_Dim]

        # --- Step 2: Patient-Specific Pooling ---
        # Goal: generate a graph_vector for each patient
        
        # Expand dimensions to match batch
        batch_size = patient_masks.size(0)
        # [Batch, Num_Nodes, Graph_Dim]
        batch_nodes = node_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Compute raw attention scores
        attn_logits = self.attention_net(batch_nodes).squeeze(-1) # [Batch, Num_Nodes]
        
        # Masking: crucial step!
        # If a patient has not activated a node (mask=0), set its attention score to -inf
        # So after softmax the weight becomes 0
        # 1e-9 prevents NaN; -1e9 represents negative infinity
        mask_value = -1e9
        attn_logits = attn_logits.masked_fill(patient_masks == 0, mask_value)
        
        # Softmax normalize to obtain final weights alpha 
        attn_weights = F.softmax(attn_logits, dim=1) # [Batch, Num_Nodes]
        
        # Weighted sum to obtain patient-specific graph vector g_i
        # [Batch, Num_Nodes, 1] * [Batch, Num_Nodes, Dim] -> Sum over nodes
        patient_graph_vec = torch.sum(attn_weights.unsqueeze(-1) * batch_nodes, dim=1)
        
        # --- Step 3: Direct feature encoding ---
        direct_vec = self.direct_encoder(direct_features)
        
        # --- Step 4: Fusion (cite: 186) ---
        combined = torch.cat([patient_graph_vec, direct_vec], dim=1)
        
        # --- Step 5: Classification ---
        logits = self.classifier(combined)
        
        # Return prediction logits and attention weights (for visualization/interpretability)
        return logits, attn_weights
    
from data_preprocessing import process_data_for_model
from knowledge_graph import build_graph_tensors
import pandas as pd
if __name__ == "__main__":
    # 1. Simulate data (Batch Size = 4)
    BATCH_SIZE = 4
    NUM_NODES = 16
    NUM_DIRECT = 11
    
    # Simulate static graph (normally obtained from knowledge_graph.py)
    # Randomly generate here as a substitute
    dummy_edge_index = torch.randint(0, NUM_NODES, (2, 40))
    dummy_edge_types = torch.randint(0, 3, (40,))
    
    # Simulate patient data
    # Masks: nodes activated by patient (0/1)
    dummy_masks = torch.randint(0, 2, (BATCH_SIZE, NUM_NODES)).float()
    # Direct: numeric features (standardized)
    dummy_direct = torch.randn(BATCH_SIZE, NUM_DIRECT)
    
    # 2. Initialize model
    model = KGNN_Diabetes(
        num_nodes=NUM_NODES,
        num_relations=3,
        num_direct_features=NUM_DIRECT
    )
    
    # 3. Forward pass
    print("Running model forward pass...")
    logits, alpha = model(dummy_edge_index, dummy_edge_types, dummy_masks, dummy_direct)
    
    # 4. Check outputs
    print(f"\nLogits Shape: {logits.shape} (Should be [4, 3])")
    print(f"Attention Weights Shape: {alpha.shape} (Should be [4, 16])")
    
    # 5. Interpretability demo
    # Inspect attention distribution for the first patient
    patient_id = 0
    weights = alpha[patient_id].detach().numpy()
    print(f"Patient {patient_id} Attention Weights sum: {weights.sum():.4f}")
    
    # Assume node 2 is Obesity
    print(f"Weight assigned to Obesity (Node 2): {weights[2]:.4f}")
