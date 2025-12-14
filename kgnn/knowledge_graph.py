import torch
import pandas as pd
import numpy as np

# ==========================================
# 1. Knowledge graph definition (static structure)
# ==========================================

# Node name mapping (for debugging and visualization)
NODE_MAP = {
    0: 'High_BP',           # Hypertension
    1: 'High_Chol',         # High cholesterol
    2: 'Obesity',           # Obesity (BMI>=30)
    3: 'Overweight',        # Overweight (25<=BMI<30)
    4: 'Smoker',            # Smoker
    5: 'Heavy_Drinker',     # Heavy drinker
    6: 'Sedentary',         # Sedentary / Lack of exercise
    7: 'Poor_Diet',         # Poor diet (lack of fruits/vegetables)
    8: 'Stroke_History',    # Stroke history
    9: 'Heart_Disease',     # Heart disease history
    10: 'Mobility_Issue',   # Mobility issues
    11: 'Poor_GenHlth',     # Poor general health
    # --- Target concepts (endpoints for message-passing; not used as direct inputs) ---
    12: 'Diabetes_Concept',     # Diabetes concept
    13: 'Pre_Diabetes_Concept',  # Pre-diabetes concept
    14: 'Mental_Stress',
    15: 'Advanced_Age'
}

NUM_NODES = len(NODE_MAP)

# Relation type definitions (used as edge_type for R-GCN)
RELATION_TYPES = {
    'CAUSES': 0,        # Causes / increases risk (directed)
    'ASSOCIATED': 1,    # Associated / comorbid (bidirectional)
    'IS_A': 2           # Is-a (auxiliary relation)
}

# Edge list (Source, Target, Relation_Type)
# These are "knowledge" edges constructed based on medical common sense
# Edge list (Source, Target, Relation_Type)
knowledge_triplets = [
    # --- A. Lifestyle causes risk factors (Lifestyle -> Risk) ---
    (6, 2, 0), # Sedentary -> Causes -> Obesity
    (6, 0, 0), # Sedentary -> Causes -> High_BP
    (7, 2, 0), # Poor_Diet -> Causes -> Obesity
    (7, 1, 0), # Poor_Diet -> Causes -> High_Chol
    (4, 9, 0), # Smoker -> Causes -> Heart_Disease
    (4, 0, 0), # Smoker -> Causes -> High_BP
    (5, 0, 0), # Heavy_Drinker -> Causes -> High_BP

    # --- B. Risk factors lead to diseases/outcomes (Risk -> Disease) ---
    (2, 0, 0), # Obesity -> Causes -> High_BP
    (2, 1, 0), # Obesity -> Causes -> High_Chol
    (2, 12, 0), # Obesity -> Causes -> Diabetes (critical pathway!)
    (2, 13, 0), # Obesity -> Causes -> Pre_Diabetes
    
    (3, 13, 0), # Overweight -> Causes -> Pre_Diabetes
    
    (0, 8, 0),  # High_BP -> Causes -> Stroke
    (0, 9, 0),  # High_BP -> Causes -> Heart_Disease
    (0, 12, 0), # High_BP -> Causes -> Diabetes
    
    (1, 9, 0),  # High_Chol -> Causes -> Heart_Disease
    (1, 12, 0), # High_Chol -> Causes -> Diabetes
    
    (13, 12, 0), # Pre_Diabetes -> Causes -> Diabetes

    # --- C. Symptom and comorbidity associations (Symptom Associations) ---
    (8, 10, 1),  # Stroke <-> Mobility_Issue
    (10, 8, 1),
    
    (9, 10, 1),  # Heart_Disease <-> Mobility_Issue
    (10, 9, 1),
    
    (12, 11, 1), # Diabetes <-> Poor_GenHlth
    (11, 12, 1),
    
    (10, 11, 1), # Mobility_Issue <-> Poor_GenHlth
    (11, 10, 1),

    # --- D. Supplementary relations for newly added nodes (New Additions) ---
    # Mental Stress (Node 14)
    (14, 11, 1), # Mental_Stress <-> Poor_GenHlth (mental issues often accompany poor general health)
    (11, 14, 1),
    
    # Advanced Age (Node 15) - this is a very strong risk factor
    (15, 12, 0), # Advanced_Age -> Increases_Risk -> Diabetes
    (15, 0, 0),  # Advanced_Age -> Increases_Risk -> High_BP
    (15, 9, 0)   # Advanced_Age -> Increases_Risk -> Heart_Disease
]

# Convert to Tensor format (PyTorch Geometric standard)
def build_graph_tensors():
    edge_index = []
    edge_types = []
    
    for src, dst, rel in knowledge_triplets:
        edge_index.append([src, dst])
        edge_types.append(rel)
    
    # Transpose to [2, num_edges]
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_types = torch.tensor(edge_types, dtype=torch.long)
    
    return edge_index, edge_types

# ==========================================
# 2. Patient data mapping
# ==========================================

def get_patient_active_nodes(row):
    """
    Input: a row (Series) from a DataFrame
    Output: list of active node indices for that patient
    Note: This strictly masks the target label (Diabetes_012)
    """
    active_nodes = []
    
    # 1. Basic binary features
    if row['HighBP'] == 1: active_nodes.append(0)
    if row['HighChol'] == 1: active_nodes.append(1)
    if row['Smoker'] == 1: active_nodes.append(4)
    if row['HvyAlcoholConsump'] == 1: active_nodes.append(5)
    if row['Stroke'] == 1: active_nodes.append(8)
    if row['HeartDiseaseorAttack'] == 1: active_nodes.append(9)
    if row['DiffWalk'] == 1: active_nodes.append(10)
    
    # 2. Derived features / logical transformations
    # BMI handling
    bmi = row['BMI']
    if bmi >= 30:
        active_nodes.append(2) # Obesity
    elif 25 <= bmi < 30:
        active_nodes.append(3) # Overweight
        
    # Physical activity (0 = No Activity -> activate Sedentary node)
    if row['PhysActivity'] == 0:
        active_nodes.append(6)
        
    # Diet (if both Fruits & Veggies are not eaten -> activate Poor_Diet)
    if row['Fruits'] == 0 and row['Veggies'] == 0:
        active_nodes.append(7)
        
    # Self-reported general health (1-5, 4 and 5 are poor -> activate Poor_GenHlth)
    if row['GenHlth'] >= 4:
        active_nodes.append(11)
        
    # *** Crucial: never add nodes 12 and 13 (the targets) ***
    
    # Assume mental health >= 14 days of poor mental health maps to Mental_Stress node
    if row['MentHlth'] >= 14:
        active_nodes.append(14)

    # In the dataset Age is coded 1-13, level 10 corresponds to ages 65-69
    if row['Age'] >= 10: 
        active_nodes.append(15)


    return active_nodes

# ==========================================
# 3. Usage example
# ==========================================
if __name__ == "__main__":
    # 1. Get graph structure
    edge_index, edge_types = build_graph_tensors()
    print(f"Knowledge Graph Created: {NUM_NODES} Nodes, {edge_index.shape[1]} Edges")

    df = pd.read_csv("diabetes_012_health_indicators_BRFSS2015.csv")
    
    # 2. Simulate a Kaggle sample (hypertension, obesity, has diabetes)
    sample_patient = pd.Series({
        'HighBP': 1, 'HighChol': 0, 'BMI': 33, 
        'Smoker': 0, 'Stroke': 0, 'HeartDiseaseorAttack': 0, 
        'PhysActivity': 0, 'Fruits': 0, 'Veggies': 1, 
        'HvyAlcoholConsump': 0, 'GenHlth': 3, 'DiffWalk': 0,
        'Diabetes_012': 2 # This is the label and will be ignored by the mapping function
    })
    sample_patient = df.iloc[0,:]
    
    # 3. Get this patient's input nodes
    active_indices = get_patient_active_nodes(sample_patient)
    print(f"Patient Active Node Indices: {active_indices}")
    print(f"Active Concepts: {[NODE_MAP[i] for i in active_indices]}")
    
    # Expected output should include High_BP(0), Obesity(2), Sedentary(6)
    # But must never include Diabetes_Concept(12)