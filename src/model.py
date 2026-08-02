import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except (ImportError, OSError):
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    class DeepSequenceModel(nn.Module):
        def __init__(self, input_size=6, hidden_size=16, num_layers=1):
            super(DeepSequenceModel, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)
            self.sigmoid = nn.Sigmoid()
            
        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            
            out, _ = self.lstm(x, (h0, c0))
            
            out = self.fc(out[:, -1, :])
            out = self.sigmoid(out)
            return out
else:
    # Mock model for systems missing PyTorch C++ Redistributables (WinError 1114 fallback)
    class MockTensor:
        def __init__(self, val, shape):
            self.val = val
            self.shape_attr = shape
        def item(self):
            return self.val
        @property
        def shape(self):
            return self.shape_attr

    class DeepSequenceModel:
        def __init__(self, input_size=6, hidden_size=16, num_layers=1):
            self.input_size = input_size
            
        def eval(self):
            pass
            
        def __call__(self, x):
            # x is a numpy array instead of a torch tensor in fallback mode
            # Mock risk calculation based on sequence length for UI demonstration
            return MockTensor(0.68, x.shape) 

# Singleton mock model for the UI
_mock_model = None

def get_model():
    global _mock_model
    if _mock_model is None:
        _mock_model = DeepSequenceModel(input_size=6, hidden_size=16)
        if hasattr(_mock_model, 'eval'):
            _mock_model.eval() # Set to evaluation mode for inference
    return _mock_model
