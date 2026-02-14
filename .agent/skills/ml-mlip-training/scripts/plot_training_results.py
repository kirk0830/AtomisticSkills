import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def plot_training_history(history_json, output_dir):
    """
    Plot training and validation loss/metrics from a training history JSON file.
    """
    print(f"Loading history from {history_json}...")
    with open(history_json, 'r') as f:
        data = json.load(f)
    
    # Check if data is a list of epochs or a dict with lists
    if isinstance(data, list):
        epochs = [d.get('epoch', i) for i, d in enumerate(data)]
        train_loss = [d.get('train_loss') or d.get('loss') for d in data]
        val_loss = [d.get('val_loss') for d in data]
    elif isinstance(data, dict):
        epochs = data.get('epochs', range(len(data.get('train_loss', []))))
        train_loss = data.get('train_loss', [])
        val_loss = data.get('val_loss', [])
    else:
        print("Unknown history format.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Train Loss')
    if val_loss and any(v is not None for v in val_loss):
        plt.plot(epochs, val_loss, label='Val Loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    plot_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(plot_path)
    print(f"Saved history plot to {plot_path}")

def plot_parity(results_json, output_dir):
    """
    Plot parity for energy and forces.
    Expects results_json to contain 'labels' and 'predictions' keys,
    each with 'energy' (eV) and 'forces' (eV/A) entries.
    """
    print(f"Loading results from {results_json}...")
    with open(results_json, 'r') as f:
        data = json.load(f)
    
    labels = data.get('labels') or data.get('reference')
    preds = data.get('predictions') or data.get('predicted')
    
    if not labels or not preds:
        print("Could not find labels or predictions in JSON.")
        return

    # Energy Parity
    true_e = np.array(labels.get('energy', []))
    pred_e = np.array(preds.get('energy', []))
    
    if len(true_e) > 0 and len(pred_e) > 0:
        plt.figure(figsize=(8, 8))
        plt.scatter(true_e, pred_e, alpha=0.5)
        lims = [min(min(true_e), min(pred_e)), max(max(true_e), max(pred_e))]
        plt.plot(lims, lims, 'r--')
        plt.xlabel('DFT Energy (eV)')
        plt.ylabel('MLIP Energy (eV)')
        plt.title('Energy Parity')
        
        mae = np.mean(np.abs(true_e - pred_e))
        plt.text(0.05, 0.95, f'MAE: {mae:.4f} eV', transform=plt.gca().transAxes)
        
        plt.savefig(os.path.join(output_dir, 'parity_energy.png'))
        print(f"Saved energy parity plot.")

    # Force Parity
    true_f = np.array(labels.get('forces', [])).flatten()
    pred_f = np.array(preds.get('forces', [])).flatten()
    
    if len(true_f) > 0 and len(pred_f) > 0:
        plt.figure(figsize=(8, 8))
        plt.scatter(true_f, pred_f, alpha=0.5, s=1)
        lims = [min(min(true_f), min(pred_f)), max(max(true_f), max(pred_f))]
        plt.plot(lims, lims, 'r--')
        plt.xlabel('DFT Force (eV/A)')
        plt.ylabel('MLIP Force (eV/A)')
        plt.title('Force Parity')
        
        mae = np.mean(np.abs(true_f - pred_f))
        plt.text(0.05, 0.95, f'MAE: {mae:.4f} eV/A', transform=plt.gca().transAxes)
        
        plt.savefig(os.path.join(output_dir, 'parity_forces.png'))
        print(f"Saved force parity plot.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot MLIP training results.")
    parser.add_argument("--history", help="Path to training history JSON")
    parser.add_argument("--results", help="Path to prediction results JSON for parity plots")
    parser.add_argument("--output_dir", default=".", help="Output directory")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    if args.history:
        plot_training_history(args.history, args.output_dir)
        
    if args.results:
        plot_parity(args.results, args.output_dir)
