import os
import sys
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from src.model import DeepSequenceModel
from src.utils import log_model_training

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # inputs are probabilities from sigmoid
        bce_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        # pt is the probability of the true class
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        # Apply Focal Loss formula to down-weight easy examples
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return torch.mean(focal_loss)

def generate_mock_data(num_samples=100, seq_len=6, input_size=19):
    # Generates mock tensor data for training (batch, seq, features)
    X = torch.rand(num_samples, seq_len, input_size)
    # Rare dismissal events: only 5% of samples are 'out' (1)
    y_vals = np.random.choice([0.0, 1.0], size=(num_samples, 1), p=[0.95, 0.05])
    y = torch.tensor(y_vals, dtype=torch.float32)
    return X, y

def train_model():
    print("Initializing DeepSequenceModel Training Pipeline...")
    model = DeepSequenceModel(input_size=19, hidden_size=32, num_layers=2)
    
    # Member 1 Task: Focal Loss
    criterion = FocalLoss(alpha=0.8, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    print("Generating 19-Dimensional Dummy T20I Sequence Data (class imbalance: 95% Safe, 5% Out)...")
    X_train, y_train = generate_mock_data(num_samples=200)
    
    epochs = 10
    final_loss = 0.0
    
    print("Starting Training Loop:")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        loss.backward()
        optimizer.step()
        
        final_loss = loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Focal Loss: {final_loss:.4f}")
        
    print("Training complete.")
    
    # Save Model Weights
    model_version = "v1.0-baseline"
    save_path = os.path.join("models", f"deepsequence_{model_version}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    
    # Member 2 Task: Database Routing
    success = log_model_training(model_version, final_loss, save_path)
    if success:
        print("Successfully routed model parameters to SQLite database (model_registry).")
    else:
        print("Failed to route model parameters.")

if __name__ == "__main__":
    train_model()
