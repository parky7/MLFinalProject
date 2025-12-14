import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

# ==========================================
# Feature extraction configuration
# ==========================================

# 1. Column names to feed directly into the MLP (includes pure direct features + dual-use features)
DIRECT_FEATURE_COLS = [
    # --- Pure Direct Features ---
    'Sex', 
    'CholCheck', 
    'AnyHealthcare', 
    'NoDocbcCost', 
    'Education', 
    'Income', 
    'PhysHlth',
    
    # --- Dual-Use Features (keep the original numeric values) ---
    'Age',      # Keep gradient 1-13
    'BMI',      # Keep actual value
    'MentHlth', # Keep exact days 0-30
    'GenHlth'   # Keep exact rating 1-5
]

def process_data_for_model(df, sample_size=None):
    """
    Input: raw Pandas DataFrame
    Output:
      1. patient_masks: [N, Num_Nodes] (graph input)
      2. direct_features: [N, Num_Direct_Features] (MLP input, normalized)
      3. labels: [N] (labels)
    """
    
    # Debug mode: to quickly test the code, you can sample a subset of the data
    if sample_size:
        print(f"Sampling {sample_size} rows for testing...")
        df = df.sample(n=sample_size, random_state=42)

    # ------------------------------------
    # Part 1: Build graph inputs (Masks)
    # ------------------------------------
    # Depends on get_patient_active_nodes function defined elsewhere
    # Here we process rows in a loop (slower but clear; for large datasets consider vectorized operations)
    print("Building Graph Masks...")
    # Note: need to import the mapping function from previous code
    from knowledge_graph import get_patient_active_nodes, NUM_NODES
    
    masks_list = []
    for _, row in df.iterrows():
        active_indices = get_patient_active_nodes(row)
        # Create an all-zero vector
        mask = torch.zeros(NUM_NODES)
        # Set active node positions to 1
        mask[active_indices] = 1.0
        masks_list.append(mask)
    
    patient_masks = torch.stack(masks_list) # Shape: [N, 16]

    # ------------------------------------
    # Part 2: Construct direct features
    # ------------------------------------
    print("Processing Direct Features...")
    # Extract the required columns
    features_raw = df[DIRECT_FEATURE_COLS].values
    
    # Standardization: (x - mean) / std
    # Neural networks are sensitive to input scales; scale inputs to be near N(0,1)
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features_raw)
    
    direct_features = torch.tensor(features_normalized, dtype=torch.float32) # Shape: [N, 11]

    # ------------------------------------
    # Part 3: Labels
    # ------------------------------------
    labels = torch.tensor(df.iloc[:,0].values, dtype=torch.long)
    
    return patient_masks, direct_features, labels

# ==========================================
# Test code
# ==========================================
if __name__ == "__main__":
    # Read data
    # df = pd.read_csv("diabetes_012_health_indicators_BRFSS2015.csv") 
    
    # For demonstration, create some dummy data
    data = {
        'HighBP': [1, 0], 'HighChol': [1, 0], 'BMI': [35, 22], 'Smoker': [1, 0],
        'Stroke': [0, 0], 'HeartDiseaseorAttack': [0, 0], 'PhysActivity': [0, 1],
        'Fruits': [0, 1], 'Veggies': [0, 1], 'HvyAlcoholConsump': [0, 0],
        'GenHlth': [4, 1], 'MentHlth': [15, 0], 'PhysHlth': [20, 0], 'DiffWalk': [1, 0],
        'Sex': [1, 0], 'Age': [10, 3], 'Education': [4, 6], 'Income': [3, 8],
        'CholCheck': [1, 1], 'AnyHealthcare': [1, 1], 'NoDocbcCost': [0, 0],
        'Diabetes_012': [2, 0]
    }
    df_dummy = pd.DataFrame(data)

    df_dummy = pd.read_csv("diabetes_012_health_indicators_BRFSS2015.csv")
    
    masks, directs, ys = process_data_for_model(df_dummy)
    
    print(f"\nGraph Masks Shape: {masks.shape}")   # Expect: [2, 16]
    print(f"Direct Features Shape: {directs.shape}") # Expect: [2, 11]
    print(f"Labels Shape: {ys.shape}")
    
    print("\nExample Direct Features (Normalized):")
    print(directs[0])