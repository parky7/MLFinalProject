import kagglehub
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Download latest version
path = kagglehub.dataset_download("alexteboul/diabetes-health-indicators-dataset")

print("Path to dataset files:", path)

# Load dataset
data = pd.read_csv("/Users/parky/.cache/kagglehub/datasets/alexteboul/diabetes-health-indicators-dataset/versions/1/diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
print(data.head())
# Path: /Users/parky/.cache/kagglehub/datasets/alexteboul/diabetes-health-indicators-dataset/versions/1/diabetes_binary_health_indicators_BRFSS2015.csv

class SimpleDNN(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size_2, output_size):
        super(SimpleDNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.fc2 = nn.Linear(hidden_size, hidden_size_2)
        self.fc3 = nn.Linear(hidden_size_2, output_size)


    def forward(self, x):
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc2(out)
        out = self.softmax(out)
        out = self.fc3(out)

        return out

# Note: determine features (X) and target (y) explicitly
# This dataset's target is likely the first column; adjust if your CSV differs.
output_size = 3  # Assuming binary classification
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
def train_model(hidden1, hidden2, lr, num_epochs=num_epochs, batch_size=batch_size, verbose=False):
    """Train a model with given hyperparameters and return test accuracy."""
    # Prepare dataloader (recreate to ensure batch_size)
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    model = SimpleDNN(input_size, hidden1, hidden2, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)
        if verbose:
            avg_loss = epoch_loss / len(train_loader.dataset)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    # Evaluate on test set
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs.data, 1)
        total = y_test_tensor.size(0)
        correct = (predicted == y_test_tensor).sum().item()
        accuracy = correct / total
    return accuracy


def train_model_with_history(hidden1, hidden2, lr, num_epochs=num_epochs, batch_size=batch_size, verbose=False,
                             X_tr=None, y_tr=None, X_val=None, y_val=None):
    """Train and return per-epoch training loss and validation loss history.

    If X_tr/y_tr or X_val/y_val are provided they will be used; otherwise global data is used.
    """
    # Use provided data or fall back to globals
    X_tr = X_tr if X_tr is not None else X_train
    y_tr = y_tr if y_tr is not None else y_train
    X_val = X_val if X_val is not None else X_test
    y_val = y_val if y_val is not None else y_test

    # Prepare dataloader
    train_dataset = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long))
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    model = SimpleDNN(input_size, hidden1, hidden2, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)
        avg_train_loss = epoch_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        # Validation loss and accuracy on X_val
        model.eval()
        with torch.no_grad():
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val, dtype=torch.long)
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
            val_losses.append(val_loss)
            # compute val accuracy for this epoch
            _, val_pred = torch.max(val_outputs.data, 1)
            val_acc = (val_pred == y_val_tensor).sum().item() / y_val_tensor.size(0)
            # keep per-epoch val accuracy
            if 'val_acc_list' not in locals():
                val_acc_list = []
            val_acc_list.append(val_acc)

        if verbose:
            print(f'Epoch [{epoch+1}/{num_epochs}] train_loss={avg_train_loss:.4f} val_loss={val_loss:.4f}')

    # final accuracy (optional)
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs.data, 1)
        total = y_test_tensor.size(0)
        correct = (predicted == y_test_tensor).sum().item()
        accuracy = correct / total

    # make sure val_acc_list exists
    try:
        val_acc_list
    except NameError:
        val_acc_list = []

    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'val_acc': val_acc_list,
        'accuracy': accuracy,
    }
    # return history and the trained model for further evaluation
    return history, model


def sweep_hyperparams(lr_list, hidden1_list, hidden2_list, num_epochs_local=10, batch_size_local=32):
    """Try combinations of learning rates and hidden layer sizes and return results.

    Returns a list of tuples: (lr, hidden1, hidden2, accuracy)
    """
    results = []
    # Use local copies for training
    global num_epochs, batch_size
    num_epochs_backup = num_epochs
    batch_size_backup = batch_size
    num_epochs = num_epochs_local
    batch_size = batch_size_local

    for lr in lr_list:
        for h1 in hidden1_list:
            for h2 in hidden2_list:
                acc = train_model(h1, h2, lr, num_epochs=num_epochs, batch_size=batch_size)
                print(f'lr={lr}, h1={h1}, h2={h2} -> accuracy={acc*100:.2f}%')
                results.append((lr, h1, h2, acc))

    # restore globals
    num_epochs = num_epochs_backup
    batch_size = batch_size_backup
    return results


if __name__ == '__main__':
    # Example sweep - adjust lists as needed
    lr_list = [0.01, 0.05, 0.1]
    hidden1_list = [12, 18, 32, 64]
    hidden2_list = [8, 12, 16, 32]
    results = sweep_hyperparams(lr_list, hidden1_list, hidden2_list, num_epochs_local=5, batch_size_local=32)
    print('\nAll results:')
    for r in results:
        print(r)
    # Pick best and visualize training
    best = max(results, key=lambda t: t[3])
    best_lr, best_h1, best_h2, best_acc = best
    print(f"\nBest by accuracy: lr={best_lr}, h1={best_h1}, h2={best_h2}, acc={best_acc*100:.2f}%")

    # Retrain best config with history and get trained model
    history, trained_model = train_model_with_history(best_h1, best_h2, best_lr,
                                                     num_epochs=20, batch_size=32,
                                                     X_tr=X_train, y_tr=y_train,
                                                     X_val=X_test, y_val=y_test,
                                                     verbose=True)

    # Plot train vs validation loss
    plt.figure(figsize=(8,5))
    plt.plot(history['train_loss'], label='train_loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss (lr={best_lr}, h1={best_h1}, h2={best_h2})')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Compute confusion matrix and ROC AUC on test set
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        outputs = trained_model(X_test_tensor)
        probs = torch.softmax(outputs, dim=1)[:, 1].numpy()  # probability of positive class
        preds = (probs >= 0.5).astype(int)

    cm = confusion_matrix(y_test.astype(int), preds)
    try:
        auc = roc_auc_score(y_test.astype(int), probs)
    except ValueError:
        auc = float('nan')

    print('\nConfusion Matrix:')
    print(cm)
    print(f'ROC AUC: {auc:.4f}')
