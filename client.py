# client.py
# PURPOSE: Simulates one bank
# Loads local data, trains model, communicates with server

import flwr as fl
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
import joblib
import os  # ✅ added for saving model
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from model import FraudDetectionModel, get_model_weights, set_model_weights

# ── LOAD AND PREPARE DATA ─────────────────────────────────────
def load_data(bank_name):
    print(f"Loading data for Bank {bank_name.upper()}...")
    
    # Load this bank's private CSV
    df = pd.read_csv(f'data/bank_{bank_name}.csv')
    
    # Separate features and labels
    X = df.drop(['Class', 'Time'], axis=1).values
    y = df['Class'].values
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    os.makedirs('model', exist_ok=True)
    joblib.dump(scaler, f'model/scaler_{bank_name}.pkl')

    # Handle class imbalance
    fraud_count = sum(y == 1)
    legit_count = sum(y == 0)
    class_weight = min(legit_count / fraud_count, 10)
    
    # Split into train (80%) and test (20%)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Convert to PyTorch tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test).unsqueeze(1)
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    print(f"Bank {bank_name.upper()} — Train: {len(X_train)} | Test: {len(X_test)} | Fraud cases: {sum(y==1)}")
    
    return train_loader, X_test_t, y_test_t, class_weight


# ── FLOWER CLIENT CLASS ───────────────────────────────────────
class BankClient(fl.client.NumPyClient):
    
    def __init__(self, bank_name):
        self.bank_name = bank_name
        self.model = FraudDetectionModel()
        self.train_loader, self.X_test, self.y_test, self.class_weight = load_data(bank_name)
        
        self.criterion = nn.BCEWithLogitsLoss(
              pos_weight=torch.tensor([self.class_weight])
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
    
    def get_parameters(self, config):
        return get_model_weights(self.model)
    
    def fit(self, parameters, config):
        # Load global weights
        set_model_weights(self.model, parameters)
        
        # Train locally
        self.model.train()
        
        for epoch in range(3):
            total_loss = 0
            
            for batch_X, batch_y in self.train_loader:
                self.optimizer.zero_grad()
                
                predictions = self.model(batch_X)
                loss = self.criterion(predictions, batch_y)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            print(f"  Bank {self.bank_name.upper()} | Epoch {epoch+1}/3 | Loss: {total_loss/len(self.train_loader):.4f}")
        
        # ✅ NEW: Save local model after training
        os.makedirs('model', exist_ok=True)
        torch.save(
            self.model.state_dict(),
            f'model/local_model_{self.bank_name}.pth'
        )
        print(f"  Bank {self.bank_name.upper()} local model saved")
        
        # Return updated weights
        return get_model_weights(self.model), len(self.train_loader.dataset), {}
    
    def evaluate(self, parameters, config):
        set_model_weights(self.model, parameters)
        self.model.eval()
        
        with torch.no_grad():
            predictions = self.model(self.X_test)
            predictions_binary = (predictions > 0.5).float()
            
            loss = self.criterion(predictions, self.y_test)
            accuracy = accuracy_score(
                self.y_test.numpy(),
                predictions_binary.numpy()
            )
        
        print(f"  Bank {self.bank_name.upper()} | Accuracy: {accuracy*100:.2f}%")
        
        return float(loss), len(self.X_test), {'accuracy': float(accuracy)}


# ── START CLIENT ──────────────────────────────────────────────
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--bank', type=str, required=True, choices=['a', 'b', 'c'],
                        help='Which bank: a (SBI), b (HDFC), c (ICICI)')
    args = parser.parse_args()
    
    bank_names = {'a': 'SBI', 'b': 'HDFC', 'c': 'ICICI'}
    print(f"\n── Starting {bank_names[args.bank]} Client ──")
    print("Connecting to central server at localhost:8080...")
    
    client = BankClient(args.bank)
   
    # 🔥 LOCAL TRAINING MODE (for UI demo)
    print("Running LOCAL training (no federation)...")

    # Train using its own weights (no server)
    client.fit(get_model_weights(client.model), {})

    print(f"{bank_names[args.bank]} local training complete")