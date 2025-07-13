#!/usr/bin/env python3
"""
Script to run inference on strains using a trained model checkpoint.
Takes a TSV file with strains as rows and outputs predictions.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from smtb_amr.model import MyModel


def load_model_from_checkpoint(checkpoint_path: str, n_feats: int) -> MyModel:
    """Load model from Lightning checkpoint file."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Extract hyperparameters from checkpoint
    hparams = checkpoint.get("hyper_parameters", {})

    # Create model with saved hyperparameters or defaults
    model = MyModel(
        n_feats=n_feats,
        dropout=hparams.get("dropout", 0.5),
        hidden_dim=hparams.get("hidden_dim", 64),
        lr=hparams.get("lr", 1e-3),
    )

    # Load the state dict
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model


def preprocess_strains(input_file: str) -> tuple[pd.DataFrame, list]:
    """
    Load and preprocess strain data from TSV file.
    Returns the feature matrix and strain IDs.
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Load the TSV file
    df = pd.read_csv(input_file, sep="\t", index_col=0)

    # Get strain IDs
    strain_ids = df.index.tolist()

    # Convert to float32 for model input
    features = df.values.astype(np.float32)

    return df, strain_ids, features


def run_inference(model: MyModel, features: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Run inference on the feature matrix."""
    model.to(device)

    # Convert to tensor
    x = torch.tensor(features, dtype=torch.float32, device=device)

    # Run inference
    with torch.no_grad():
        logits = model(x)
        # Apply sigmoid to get probabilities
        probabilities = torch.sigmoid(logits)
        # Convert to binary predictions (threshold at 0.5)
        predictions = (probabilities > 0.5).int()

    return predictions.cpu().numpy().flatten(), probabilities.cpu().numpy().flatten()


def save_results(strain_ids: list, predictions: np.ndarray, probabilities: np.ndarray, output_file: str):
    """Save predictions to TSV file."""
    results_df = pd.DataFrame({"strain_id": strain_ids, "prediction": predictions, "probability": probabilities})

    # Save to TSV
    results_df.to_csv(output_file, sep="\t", index=False)
    print(f"Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on strains using a trained AMR model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--input", type=str, required=True, help="Input TSV file with strains as rows and features as columns"
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint file (.ckpt)")
    parser.add_argument("--output", type=str, required=True, help="Output TSV file for predictions")

    # Optional arguments
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"], help="Device to run inference on"
    )
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for inference (for large datasets)")

    args = parser.parse_args()

    # Validate device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    elif args.device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        args.device = "cpu"

    print(f"Using device: {args.device}")
    print(f"Loading data from: {args.input}")

    # Load and preprocess data
    try:
        df, strain_ids, features = preprocess_strains(args.input)
        print(f"Loaded {len(strain_ids)} strains with {features.shape[1]} features")

        # Load model
        print(f"Loading model from checkpoint: {args.checkpoint}")
        model = load_model_from_checkpoint(args.checkpoint, n_feats=features.shape[1])

        # Run inference
        print("Running inference...")
        if len(strain_ids) > args.batch_size:
            # Process in batches for large datasets
            all_predictions = []
            all_probabilities = []

            for i in range(0, len(strain_ids), args.batch_size):
                batch_features = features[i : i + args.batch_size]
                batch_preds, batch_probs = run_inference(model, batch_features, args.device)
                all_predictions.extend(batch_preds)
                all_probabilities.extend(batch_probs)

                if (i // args.batch_size + 1) % 10 == 0:
                    print(f"Processed {i + len(batch_features)} / {len(strain_ids)} strains")

            predictions = np.array(all_predictions)
            probabilities = np.array(all_probabilities)
        else:
            predictions, probabilities = run_inference(model, features, args.device)

        # Save results
        print("Saving results...")
        save_results(strain_ids, predictions, probabilities, args.output)

        # Print summary
        resistant_count = np.sum(predictions == 1)
        susceptible_count = np.sum(predictions == 0)
        print("\nPrediction Summary:")
        print(f"  Resistant: {resistant_count} ({resistant_count / len(predictions) * 100:.1f}%)")
        print(f"  Susceptible: {susceptible_count} ({susceptible_count / len(predictions) * 100:.1f}%)")
        print(f"  Total: {len(predictions)}")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
