# client.py
# PURPOSE: Simulates one bank
# Loads local data, trains model, communicates with server

import flwr as fl
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
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
    X = df.drop(['Class', 'Time'], axis=1).values       #everything except Class column
    y = df['Class'].values                  # just the Class column (0 or 1)
    
    # Normalize features
    # Why? Neural networks work better when all numbers are in similar range
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Handle class imbalance
    # Problem: 99.8% legit, 0.2% fraud — model would just predict legit always
    # Solution: give fraud cases higher weight during training
    fraud_count = sum(y == 1)
    legit_count = sum(y == 0)
    class_weight = legit_count / fraud_count  # fraud cases weighted more
    
    # Split into train (80%) and test (20%)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Convert to PyTorch tensors
    # Why tensors? PyTorch works with tensors, not numpy arrays
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)  # add extra dimension
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test).unsqueeze(1)
    
    # Create DataLoaders — batches data for efficient training
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
        
        # Loss function — measures how wrong predictions are
        # pos_weight handles class imbalance
        self.criterion = nn.BCELoss()
        
        # Optimizer — adjusts weights to reduce loss
        # Adam is most popular optimizer, lr=learning rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
    
    def get_parameters(self, config):
        # Server calls this to get current weights
        return get_model_weights(self.model)
    
    def fit(self, parameters, config):
        # Server sends global weights → we load them → train locally → send back
        
        # Load global weights received from server
        set_model_weights(self.model, parameters)
        
        # Train for 3 epochs locally
        self.model.train()  # set model to training mode
        
        for epoch in range(3):
            total_loss = 0
            
            for batch_X, batch_y in self.train_loader:
                # Clear previous gradients
                self.optimizer.zero_grad()
                
                # Forward pass — model makes predictions
                predictions = self.model(batch_X)
                
                # Calculate loss — how wrong were the predictions
                loss = self.criterion(predictions, batch_y)
                
                # Backward pass — calculate gradients
                loss.backward()
                
                # Update weights based on gradients
                self.optimizer.step()
                
                total_loss += loss.item()
            
            print(f"  Bank {self.bank_name.upper()} | Epoch {epoch+1}/3 | Loss: {total_loss/len(self.train_loader):.4f}")
        
        # Return updated weights + number of training samples used
        return get_model_weights(self.model), len(self.train_loader.dataset), {}
    
    def evaluate(self, parameters, config):
        # Server asks us to evaluate the global model on our local test data
        
        set_model_weights(self.model, parameters)
        self.model.eval()  # set model to evaluation mode
        
        with torch.no_grad():  # no gradient calculation needed for evaluation
            predictions = self.model(self.X_test)
            predictions_binary = (predictions > 0.5).float()  # threshold at 0.5
            
            loss = self.criterion(predictions, self.y_test)
            accuracy = accuracy_score(
                self.y_test.numpy(),
                predictions_binary.numpy()
            )
        
        print(f"  Bank {self.bank_name.upper()} | Accuracy: {accuracy*100:.2f}%")
        
        return float(loss), len(self.X_test), {'accuracy': float(accuracy)}


# ── START CLIENT ──────────────────────────────────────────────
if __name__ == "__main__":
    
    # Accept bank name as command line argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--bank', type=str, required=True, choices=['a', 'b', 'c'],
                        help='Which bank: a (SBI), b (HDFC), c (ICICI)')
    args = parser.parse_args()
    
    bank_names = {'a': 'SBI', 'b': 'HDFC', 'c': 'ICICI'}
    print(f"\n── Starting {bank_names[args.bank]} Client ──")
    print("Connecting to central server at localhost:8080...")
    
    client = BankClient(args.bank)
    
    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=client
    )