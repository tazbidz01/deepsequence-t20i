import torch
import torch.nn as nn

class DeepSequenceModel(nn.Module):
    def __init__(self, input_size=6, hidden_size=16, num_layers=1):
        super(DeepSequenceModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # dynamic batch_first=True allows arbitrary sequence lengths
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: (batch, seq_length, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        
        # We take the output of the last time step for prediction
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out

# Singleton mock model for the UI
_mock_model = None

def get_model():
    global _mock_model
    if _mock_model is None:
        _mock_model = DeepSequenceModel(input_size=6, hidden_size=16)
        _mock_model.eval() # Set to evaluation mode for inference
    return _mock_model
