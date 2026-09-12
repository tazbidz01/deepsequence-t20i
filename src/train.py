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

def generate_mock_data(num_samples=100, seq_len=6, input_size=59):
    # Generates mock tensor data for training (batch, seq, features)
    X = torch.rand(num_samples, seq_len, input_size)
    y_vals = []
    
    for i in range(num_samples):
        last_ball = X[i, -1, :]
        bat_avg_vs_style = last_ball[33].item()
        b_career_sr = last_ball[32].item()
        b_wkts_vs_style = last_ball[36].item()
        
        # New 50D Supporting Bowler KPIs
        try:
            sb_career_wkts = last_ball[46].item()
            sb_career_sr = last_ball[49].item()
        except IndexError:
            sb_career_wkts = 0.5
            sb_career_sr = 0.5
            
        # New 52D True Lifetime Batsman KPIs
        try:
            bat_career_avg = last_ball[50].item()
            bat_career_sr = last_ball[51].item()
        except IndexError:
            bat_career_avg = 0.5
            bat_career_sr = 0.5
            
        # New 59D NLP Vulnerability Matrix
        try:
            v_len = last_ball[56].item()
            v_line = last_ball[57].item()
            v_shot = last_ball[58].item()
        except IndexError:
            v_len, v_line, v_shot = 0.5, 0.5, 0.5
        
        # Heavily weight the targets based on career KPIs
        risk_prob = 0.05
        if bat_avg_vs_style < 0.4: risk_prob += 0.10
        if b_career_sr < 0.5: risk_prob += 0.10
        if b_wkts_vs_style > 0.5: risk_prob += 0.10
        
        # Add Partnership pressure
        if sb_career_wkts > 0.6: risk_prob += 0.05
        if sb_career_sr < 0.4: risk_prob += 0.10
        
        # Add True Lifetime Batsman Pressure
        if bat_career_avg < 0.4: risk_prob += 0.10
        if bat_career_sr < 0.5: risk_prob += 0.05
        
        # Fluid Continuous Math for NLP Vulnerability Injection
        # Transforming the 3 continuous mechanic arrays into an exponential growth/decay curve
        nlp_exponent = (v_len - 0.5) + (v_line - 0.5) + (v_shot - 0.5)
        nlp_multiplier = np.exp(nlp_exponent * 1.5)  # Scale multiplier curve
        
        # Smoothly apply the fluid transformation to the baseline risk
        risk_prob = risk_prob * nlp_multiplier
        
        risk_prob = min(max(risk_prob, 0.0), 1.0)
        y_val = 1.0 if np.random.rand() < risk_prob else 0.0
        y_vals.append([y_val])
        
    y = torch.tensor(y_vals, dtype=torch.float32)
    return X, y

def train_model():
    print("Initializing DeepSequenceModel Training Pipeline (59D Matrix Logic)...")
    model = DeepSequenceModel(input_size=59, hidden_size=64, num_layers=2)
    
    # Member 1 Task: Focal Loss
    criterion = FocalLoss(alpha=0.8, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    print("Generating 59-Dimensional Dummy T20I Sequence Data...")
    X_train, y_train = generate_mock_data(num_samples=8000)
    
    epochs = 40
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
    
    # Save model weights to the shared models directory
    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', 'deepsequence_v1.6-baseline.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Member 2 Task: Database Routing
    model_version = "v1.6-baseline"
    success = log_model_training(model_version, final_loss, model_path)
    if success:
        print("Successfully routed model parameters to SQLite database (model_registry).")
    else:
        print("Failed to route model parameters.")

if __name__ == "__main__":
    train_model()
