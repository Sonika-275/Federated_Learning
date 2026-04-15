# model.py
# PURPOSE: Define the neural network structure
# This file is imported by client.py, server.py and app.py
# Everyone uses the same blueprint

import torch
import torch.nn as nn

class FraudDetectionModel(nn.Module):
    
    def __init__(self):
        super(FraudDetectionModel, self).__init__()
        
        # ── LAYER 1: Input layer ──────────────────────────────
        # 29 input features → 64 neurons
        # Why 64? Enough capacity to learn patterns, not too big
        self.layer1 = nn.Linear(29,64)
        
        # ── LAYER 2: Hidden layer ─────────────────────────────
        # 64 neurons → 32 neurons
        # Getting narrower — forcing model to learn key features
        self.layer2 = nn.Linear(64, 32)
        
        # ── LAYER 3: Output layer ─────────────────────────────
        # 32 neurons → 1 output
        # 1 output because its binary: fraud or not fraud
        self.layer3 = nn.Linear(32, 1)
        
        # ── ACTIVATION FUNCTIONS ──────────────────────────────
        # ReLU: used between layers, helps model learn non-linear patterns
        self.relu = nn.ReLU()
       
        
        # ── DROPOUT ───────────────────────────────────────────
        # Randomly turns off 30% of neurons during training
        # Prevents model from memorizing — forces it to actually learn
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        # This defines how data flows through the network
        
        x = self.relu(self.layer1(x))   # input → layer1 → relu
        x = self.dropout(x)              # dropout for regularization
        x = self.relu(self.layer2(x))   # layer1 → layer2 → relu
        x = self.dropout(x)              # dropout again
        x = self.layer3(x) # layer2 → output 
        
        return x


# ── HELPER FUNCTIONS ──────────────────────────────────────────
# These are used by client.py to extract and load weights

def get_model_weights(model):
    # Extract weights from model as a list of numpy arrays
    # This is what gets sent to the server
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_model_weights(model, weights):
    # Load weights (received from server) back into the model
    params_dict = zip(model.state_dict().keys(), weights)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)