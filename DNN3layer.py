import kagglehub
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class SimpleDNN(nn.Module):
    """Simple 3-layer network: input -> hidden -> output (logits).

    Note: CrossEntropyLoss expects raw logits (no softmax on output).
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleDNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.activation = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)  # logits for CrossEntropyLoss
        return x


class ImprovedDNN(nn.Module):
    """Improved deeper network with regularization: input -> hidden1 -> hidden2 -> output.
    
    Features:
    - Multiple hidden layers
    - Batch normalization
    - Dropout for regularization
    - ReLU activations (generally better than Sigmoid)
    """
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, dropout_rate=0.3):
        super(ImprovedDNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)  # logits for CrossEntropyLoss
        return x


def train_model(hidden1, lr, num_epochs, batch_size, verbose=False, dataloader=None, input_size=None, output_size=None, **kwargs):
    """Train a model with given hyperparameters and return training accuracy."""
    if dataloader is None or input_size is None or output_size is None:
        raise ValueError("dataloader, input_size, and output_size are required")
    
    model = SimpleDNN(input_size, hidden1, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_inputs, batch_labels in dataloader:
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_inputs.size(0)
        if verbose:
            avg_loss = epoch_loss / len(dataloader.dataset)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    # Evaluate on training set
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for batch_inputs, batch_labels in dataloader:
            outputs = model(batch_inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
        accuracy = correct / total if total > 0 else 0.0
    return accuracy

def sequential_feature_selection(hidden1, lr, num_epochs, batch_size, X_train, y_train, input_size=None, output_size=None):
    """Perform sequential feature selection to identify best features.

    Returns list of selected feature indices.
    """
    num_features = input_size
    selected_features = []
    remaining_features = list(range(num_features))
    best_accuracy = 0.0

    while remaining_features:
        feature_to_add = None
        best_feature_accuracy = best_accuracy
        
        for feature in remaining_features:
            current_features = selected_features + [feature]
            # Create a new dataloader with only current features
            X_subset = X_train[:, current_features]
            X_subset_tensor = torch.tensor(X_subset, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.long)
            train_dataset = TensorDataset(X_subset_tensor, y_train_tensor)
            subset_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

            accuracy = train_model(hidden1, lr, num_epochs, batch_size, verbose=False, dataloader=subset_dataloader, input_size=len(current_features), output_size=output_size)
            if accuracy > best_feature_accuracy:
                best_feature_accuracy = accuracy
                feature_to_add = feature

        if feature_to_add is not None:
            selected_features.append(feature_to_add)
            remaining_features.remove(feature_to_add)
            best_accuracy = best_feature_accuracy
            print(f'Added feature {feature_to_add}, new accuracy: {best_accuracy:.4f}')
        else:
            break  # No improvement

    return selected_features


def train_model_with_history(hidden1, lr, num_epochs, batch_size, verbose=False,
                             X_val=None, y_val=None, dataloader=None, input_size=None, output_size=None,
                             model_class=SimpleDNN, use_early_stopping=True, patience=5, hidden2=None, class_weights=None):
    """Train and return per-epoch training loss and validation loss history.
    
    Args:
        hidden1: First hidden layer size
        hidden2: Second hidden layer size (only for ImprovedDNN)
        use_early_stopping: Whether to stop early if validation loss plateaus
        patience: Number of epochs with no improvement before stopping
        class_weights: Weights for each class to handle imbalance
    """
    if dataloader is None or input_size is None or output_size is None:
        raise ValueError("dataloader, input_size, and output_size are required")
    
    # Create model based on class
    if model_class == ImprovedDNN:
        if hidden2 is None:
            hidden2 = hidden1  # Use same size if not specified
        model = ImprovedDNN(input_size, hidden1, hidden2, output_size)
    else:
        model = SimpleDNN(input_size, hidden1, output_size)
    
    # Handle class weights for imbalanced data
    if class_weights is not None:
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Use Adam optimizer instead of SGD - generally better
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    # Optional: learning rate scheduler to decay learning rate
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    train_losses = []
    val_losses = []
    val_acc_list = []

    # Convert validation data to tensors once
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    
    # Verify validation data dimensions match input_size
    if X_val_tensor.shape[1] != input_size:
        raise ValueError(f"Validation data has {X_val_tensor.shape[1]} features but model expects {input_size}")

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_inputs, batch_labels in dataloader:
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_inputs.size(0)
        avg_train_loss = epoch_loss / len(dataloader.dataset)
        train_losses.append(avg_train_loss)

        # Validation loss and accuracy
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
            val_losses.append(val_loss)
            # compute val accuracy for this epoch
            _, val_pred = torch.max(val_outputs.data, 1)
            val_acc = (val_pred == y_val_tensor).sum().item() / y_val_tensor.size(0)
            val_acc_list.append(val_acc)

        if verbose:
            print(f'Epoch [{epoch+1}/{num_epochs}] train_loss={avg_train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}')

        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if use_early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f'Early stopping at epoch {epoch+1}')
                    break

    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'val_acc': val_acc_list,
    }
    # return history and the trained model for further evaluation
    return history, model


def find_best_threshold(y_val, probs):
    """Find optimal decision threshold for classification.
    
    Tries different thresholds to maximize F1 score or other metrics.
    """
    from sklearn.metrics import f1_score
    best_threshold = 0.5
    best_f1 = 0
    
    for threshold in np.arange(0.3, 0.8, 0.01):
        preds = (probs >= threshold).astype(int)
        f1 = f1_score(y_val.astype(int), preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold, best_f1


def train_ensemble_models(n_models, hidden1, lr, num_epochs, batch_size, 
                         X_train, y_train, X_val, y_val, input_size, output_size):
    """Train multiple models and return ensemble predictions."""
    models = []
    print(f"\nTraining ensemble of {n_models} models...")
    
    for i in range(n_models):
        print(f"\n  Model {i+1}/{n_models}")
        # Create dataloader with different random seed (shuffle is true)
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        
        history, model = train_model_with_history(
            hidden1=hidden1,
            hidden2=hidden1//2,
            lr=lr,
            num_epochs=50,
            batch_size=batch_size,
            X_val=X_val,
            y_val=y_val,
            verbose=False,
            dataloader=dataloader,
            input_size=input_size,
            output_size=output_size,
            model_class=ImprovedDNN,
            use_early_stopping=True,
            patience=6
        )
        models.append(model)
    
    return models





if __name__ == '__main__':
    # Download latest version
    path = kagglehub.dataset_download("alexteboul/diabetes-health-indicators-dataset")

    print("Path to dataset files:", path)

    # Load dataset
    data = pd.read_csv("/Users/parky/.cache/kagglehub/datasets/alexteboul/diabetes-health-indicators-dataset/versions/1/diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
    print(data.head())
    # Path: /Users/parky/.cache/kagglehub/datasets/alexteboul/diabetes-health-indicators-dataset/versions/1/diabetes_binary_health_indicators_BRFSS2015.csv
    # Note: determine features (X) and target (y) explicitly
    # This dataset's target is likely the first column; adjust if your CSV differs.
    output_size = 2  # binary classification (0/1)
    batch_size = 32
    num_epochs = 10
    # Prepare data for training
    # Use all columns except the first as input features, and the first column as the target
    X = data.iloc[:, 1:].values
    y = data.iloc[:, 0].values
    # Set input_size based on actual feature count to avoid shape mismatch
    input_size = X.shape[1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


    # Example sweep - adjust lists as needed
    # lr_list = [0.01, 0.05, 0.1]
    # hidden1_list = [12, 18, 32, 64]
    # results = sweep_hyperparams(lr_list, hidden1_list, num_epochs_local=5, batch_size_local=32, input_size=input_size, dataloader=dataloader, output_size=output_size)
    # print('\nAll results:')
    # for r in results:
    #     print(r)
    # # Pick best and visualize training
    # best = max(results, key=lambda t: t[2])
    # best_lr, best_h1, best_acc = best
    # print(f"\nBest by accuracy: lr={best_lr}, h1={best_h1}, acc={best_acc*100:.2f}%")

    # # Retrain best config with history and get trained model
    # history, trained_model = train_model_with_history(best_h1, best_lr,
    #                                                  num_epochs=20, batch_size=32,
    #                                                  X_val=X_test, y_val=y_test,
    #                                                  verbose=True, input_size=input_size, dataloader=dataloader, output_size=output_size)

    # # Plot train vs validation loss
    # plt.figure(figsize=(8,5))
    # plt.plot(history['train_loss'], label='train_loss')
    # plt.plot(history['val_loss'], label='val_loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title(f'Training and Validation Loss (lr={best_lr}, h1={best_h1})')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    best_features = list(range(input_size))
    print(f'\nUsing all {len(best_features)} features')

    # Calculate class weights
    unique, counts = pd.Series(y_train).value_counts().sort_index().values, pd.Series(y_train).value_counts().sort_index().values
    class_weights = counts.sum() / (2 * counts)
    class_weights = class_weights / class_weights.sum()

    # Train single improved model
    print("\n=== Training Single Improved Model ===")
    history, trained_model = train_model_with_history(
        hidden1=64, 
        hidden2=32,
        lr=0.001,
        num_epochs=100, 
        batch_size=32,
        X_val=X_test, 
        y_val=y_test,
        verbose=False,
        dataloader=DataLoader(
            dataset=TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.long)
            ),
            batch_size=32,
            shuffle=True
        ),
        input_size=input_size,
        output_size=output_size,
        model_class=ImprovedDNN,
        use_early_stopping=True,
        patience=8,
        class_weights=class_weights
    )
    print("Single model training complete")

    # Train ensemble for comparison
    print("\n=== Training Ensemble (3 models) ===")
    ensemble_models = train_ensemble_models(
        n_models=3,
        hidden1=64,
        lr=0.001,
        num_epochs=50,
        batch_size=32,
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        input_size=input_size,
        output_size=output_size
    )
    
    # Get ensemble predictions
    print("\nGenerating ensemble predictions...")
    ensemble_probs = []
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        for model in ensemble_models:
            outputs = model(X_test_tensor)
            probs = torch.softmax(outputs, dim=1)[:, 1].numpy()
            ensemble_probs.append(probs)
    
    ensemble_avg_probs = np.mean(ensemble_probs, axis=0)
    
    # Find best threshold for ensemble
    best_threshold_ensemble, best_f1_ensemble = find_best_threshold(y_test, ensemble_avg_probs)
    print(f"Best threshold for ensemble: {best_threshold_ensemble:.3f} (F1={best_f1_ensemble:.4f})")
    
    # Get single model predictions
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        outputs = trained_model(X_test_tensor)
        single_probs = torch.softmax(outputs, dim=1)[:, 1].numpy()
    
    best_threshold_single, best_f1_single = find_best_threshold(y_test, single_probs)
    print(f"Best threshold for single model: {best_threshold_single:.3f} (F1={best_f1_single:.4f})")
    
    # Use ensemble predictions with optimized threshold
    preds_single = (single_probs >= best_threshold_single).astype(int)
    preds_ensemble = (ensemble_avg_probs >= best_threshold_ensemble).astype(int)

    # Results for single model
    cm_single = confusion_matrix(y_test.astype(int), preds_single)
    try:
        auc_single = roc_auc_score(y_test.astype(int), single_probs)
    except ValueError:
        auc_single = float('nan')

    # Results for ensemble
    cm_ensemble = confusion_matrix(y_test.astype(int), preds_ensemble)
    try:
        auc_ensemble = roc_auc_score(y_test.astype(int), ensemble_avg_probs)
    except ValueError:
        auc_ensemble = float('nan')

    print('\n' + '='*60)
    print('RESULTS SUMMARY')
    print('='*60)
    
    print('\n--- SINGLE MODEL (with optimized threshold) ---')
    print(f'Confusion Matrix:\n{cm_single}')
    tn, fp, fn, tp = cm_single.ravel()
    acc_single = (tp + tn) / (tp + tn + fp + fn)
    sens_single = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec_single = tn / (tn + fp) if (tn + fp) > 0 else 0
    balanced_acc_single = (sens_single + spec_single) / 2
    precision_single = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_single = sens_single
    f1_single = 2 * (precision_single * recall_single) / (precision_single + recall_single) if (precision_single + recall_single) > 0 else 0
    print(f'Accuracy:   {acc_single:.4f} ({acc_single*100:.2f}%)')
    print(f'Sensitivity: {sens_single:.4f}')
    print(f'Specificity: {spec_single:.4f}')
    print(f'ROC AUC:    {auc_single:.4f}')
    print(f'Precision:  {precision_single:.4f}')
    print(f'Recall:     {recall_single:.4f}')
    print(f'Balanced Accuracy: {balanced_acc_single:.4f}')
    print(f'F1 Score:   {f1_single:.4f}')   

    
    print('\n--- ENSEMBLE (3 models, with optimized threshold) ---')
    print(f'Confusion Matrix:\n{cm_ensemble}')
    tn, fp, fn, tp = cm_ensemble.ravel()
    acc_ensemble = (tp + tn) / (tp + tn + fp + fn)
    sens_ensemble = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec_ensemble = tn / (tn + fp) if (tn + fp) > 0 else 0
    balanced_acc_single = (sens_ensemble + spec_ensemble) / 2
    precision_ensemble = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_ensemble = sens_ensemble
    f1_ensemble = 2 * (precision_ensemble * recall_ensemble) / (precision_ensemble + recall_ensemble) if (precision_ensemble + recall_ensemble) > 0 else 0
    print(f'Accuracy:   {acc_ensemble:.4f} ({acc_ensemble*100:.2f}%)')
    print(f'Sensitivity: {sens_ensemble:.4f}')
    print(f'Specificity: {spec_ensemble:.4f}')
    print(f'ROC AUC:    {auc_ensemble:.4f}')
    print(f'Precision:  {precision_ensemble:.4f}')
    print(f'Recall:     {recall_ensemble:.4f}')
    print(f'Balanced Accuracy: {balanced_acc_single:.4f}')
    print(f'F1 Score:   {f1_ensemble:.4f}')
    
    print('='*60)
