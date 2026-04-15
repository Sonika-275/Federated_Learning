# server.py
# PURPOSE: Central aggregation server
# Waits for 3 clients, runs FedAvg, saves final model

import flwr as fl
import torch
import numpy as np
from model import FraudDetectionModel, set_model_weights
from typing import List, Tuple, Optional, Dict
import os

# ── TRACKING ACCURACY ACROSS ROUNDS ──────────────────────────
# We store accuracy each round to show improvement in UI later
round_accuracies = []

# ── CUSTOM FEDAVG STRATEGY ────────────────────────────────────
class SaveModelStrategy(fl.server.strategy.FedAvg):
    
    def aggregate_fit(self, server_round, results, failures):
        # This runs after every round when all clients send weights back
        
        # Call parent FedAvg to do the actual averaging
        aggregated_weights, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        if aggregated_weights is not None:
            print(f"\nRound {server_round} — Aggregation complete")
            
            # Save the global model after EVERY round
            # So we have a checkpoint in case something crashes
            model = FraudDetectionModel()
            weights = fl.common.parameters_to_ndarrays(aggregated_weights)
            set_model_weights(model, weights)
            
            os.makedirs('model', exist_ok=True)
            torch.save(model.state_dict(), f'model/round_{server_round}_model.pth')
            
            # Save final model separately after last round
            if server_round == 5:
                torch.save(model.state_dict(), 'model/global_model.pth')
                print("✓ Final global model saved to model/global_model.pth")
        
        return aggregated_weights, aggregated_metrics
    
    def aggregate_evaluate(self, server_round, results, failures):
        # This collects accuracy reported by each client after evaluation
        
        if not results:
            return None, {}
        
        # Average the accuracy across all clients
        accuracies = [r.metrics['accuracy'] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]
        avg_accuracy = sum(accuracies) / sum(examples)
        
        round_accuracies.append({
            'round': server_round,
            'accuracy': avg_accuracy
        })
        
        print(f"Round {server_round} — Average Accuracy: {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)")
        
        # Save accuracy log for UI to read later
        import json
        os.makedirs('model', exist_ok=True)
        with open('model/accuracy_log.json', 'w') as f:
            json.dump(round_accuracies, f)
        
        return avg_accuracy, {'accuracy': avg_accuracy}


# ── START THE SERVER ──────────────────────────────────────────
def start_server():
    
    strategy = SaveModelStrategy(
        fraction_fit=1.0,          # use 100% of available clients each round
        fraction_evaluate=1.0,     # evaluate on 100% of clients
        min_fit_clients=3,         # wait for exactly 3 clients before starting
        min_evaluate_clients=3,    # evaluate on all 3 clients
        min_available_clients=3,   # don't start until 3 clients connected
    )
    
    print("── Federated Learning Server Starting ──")
    print("Waiting for 3 bank clients to connect...")
    print("Run client.py in 3 separate terminals\n")
    
    fl.server.start_server(
        server_address="localhost:8080",  # server runs on port 8080
        config=fl.server.ServerConfig(num_rounds=5),  # 5 rounds of federation
        strategy=strategy,
    )
    
    print("\n── Training Complete ──")
    print("Accuracy per round:")
    for r in round_accuracies:
        print(f"  Round {r['round']}: {r['accuracy']*100:.2f}%")


if __name__ == "__main__":
    start_server()