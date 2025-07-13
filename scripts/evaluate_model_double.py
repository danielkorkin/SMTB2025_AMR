#!/usr/bin/env python3
"""
Model Evaluation Script for Double Bacteria AMR Prediction Models

This script evaluates model performance by extracting test MCC scores from
metrics.csv files across different bacterial pairs and creates a heatmap
visualization.

Usage:
    python evaluate_model_double.py /path/to/results/directory

The script expects the following directory structure:
/path/to/results/directory/
├── {bacteria1}_{bacteria2}/
│   └── version_0/
│       └── metrics.csv
└── ...
"""

import argparse
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# Define the bacterial species
SPECIES = [
    "Acinetobacter_baumannii",
    "Campylobacter_jejuni",
    "Escherichia_coli",
    "Klebsiella_pneumoniae",
    "Neisseria_gonorrhoeae",
    "Pseudomonas_aeruginosa",
    "Salmonella_enterica",
    "Staphylococcus_aureus",
    "Streptococcus_pneumoniae",
]


def extract_last_test_mcc(csv_path: Path) -> Optional[float]:
    """
    Extract the test MCC score from the last epoch in a metrics.csv file.

    Args:
        csv_path: Path to the metrics.csv file

    Returns:
        The test MCC score from the last epoch, or None if not found
    """
    try:
        df = pd.read_csv(csv_path)

        # Filter rows that have test_mcc values (not NaN)
        test_mcc_rows = df.dropna(subset=["test_mcc"])

        if test_mcc_rows.empty:
            print(f"Warning: No test_mcc values found in {csv_path}")
            return None

        # Get the last row with test_mcc
        last_test_mcc = test_mcc_rows.iloc[-1]["test_mcc"]
        return float(last_test_mcc)

    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None


def collect_double_bacteria_results(results_dir: Path) -> Dict[str, Dict[str, float]]:
    """
    Collect test MCC results from all bacterial pairs.

    Args:
        results_dir: Path to the results directory

    Returns:
        Nested dictionary: {bacteria1: {bacteria2: mcc_score}}
    """
    results = {}

    # Initialize the results dictionary
    for bacteria1 in SPECIES:
        results[bacteria1] = {}

    # Look for all bacterial pair directories
    for bacteria1 in SPECIES:
        for bacteria2 in SPECIES:
            folder_name = f"{bacteria1}_{bacteria2}"
            model_dir = results_dir / folder_name

            if not model_dir.exists():
                print(f"Warning: Directory {model_dir} not found")
                continue

            # Look for version_0/metrics.csv
            version_dir = model_dir / "version_0"
            metrics_file = version_dir / "metrics.csv"

            if not metrics_file.exists():
                print(f"Warning: Metrics file {metrics_file} not found")
                continue

            mcc_score = extract_last_test_mcc(metrics_file)
            if mcc_score is not None:
                results[bacteria1][bacteria2] = mcc_score
            else:
                print(f"Warning: Could not extract MCC score from {metrics_file}")

    return results


def create_heatmap_dataframe(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Create a DataFrame suitable for heatmap visualization.

    Args:
        results: Nested dictionary with MCC scores

    Returns:
        DataFrame with bacteria as both rows and columns, MCC scores as values
    """
    # Create a matrix with NaN values
    heatmap_data = pd.DataFrame(index=SPECIES, columns=SPECIES, dtype=float)

    # Fill in the MCC scores
    for bacteria1 in SPECIES:
        for bacteria2 in SPECIES:
            if bacteria1 in results and bacteria2 in results[bacteria1]:
                heatmap_data.loc[bacteria1, bacteria2] = results[bacteria1][bacteria2]

    return heatmap_data


def create_heatmap(heatmap_data: pd.DataFrame, output_dir: Path) -> None:
    """
    Create and save a heatmap visualization.

    Args:
        heatmap_data: DataFrame with MCC scores
        output_dir: Directory to save the heatmap
    """
    # Create figure and axis
    plt.figure(figsize=(14, 12))

    # Create heatmap
    mask = heatmap_data.isnull()

    # Check if entire rows or columns are missing
    missing_rows = heatmap_data.isnull().all(axis=1)
    missing_cols = heatmap_data.isnull().all(axis=0)

    if missing_rows.any():
        print(f"\nWarning: Completely missing data for training bacteria: {list(heatmap_data.index[missing_rows])}")
    if missing_cols.any():
        print(f"Warning: Completely missing data for test bacteria: {list(heatmap_data.columns[missing_cols])}")

    # Create the heatmap with better formatting for missing data
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".3f",
        cmap="RdYlBu_r",
        center=0,
        mask=mask,
        square=True,
        cbar_kws={"shrink": 0.8, "label": "Test MCC Score"},
        xticklabels=[species.replace("_", " ") for species in SPECIES],
        yticklabels=[species.replace("_", " ") for species in SPECIES],
        linewidths=0.5,
        linecolor="white",
    )

    plt.title("Test MCC Scores for Bacterial Pair Models\n(Gray cells indicate missing data)", fontsize=16, pad=20)
    plt.xlabel("Test Bacteria", fontsize=12)
    plt.ylabel("Training Bacteria", fontsize=12)

    # Rotate labels for better readability
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    output_file = output_dir / "bacterial_pairs_mcc_heatmap.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Heatmap saved to: {output_file}")

    # Also save as PDF
    output_file_pdf = output_dir / "bacterial_pairs_mcc_heatmap.pdf"
    plt.savefig(output_file_pdf, bbox_inches="tight")
    print(f"Heatmap saved to: {output_file_pdf}")

    plt.show()

    # Create a second heatmap excluding completely missing rows/columns
    if missing_rows.any() or missing_cols.any():
        create_filtered_heatmap(heatmap_data, output_dir, missing_rows, missing_cols)


def create_filtered_heatmap(
    heatmap_data: pd.DataFrame, output_dir: Path, missing_rows: pd.Series, missing_cols: pd.Series
) -> None:
    """
    Create a filtered heatmap excluding completely missing rows/columns.

    Args:
        heatmap_data: DataFrame with MCC scores
        output_dir: Directory to save the heatmap
        missing_rows: Boolean series indicating missing rows
        missing_cols: Boolean series indicating missing columns
    """
    # Filter out completely missing rows and columns
    filtered_data = heatmap_data.loc[~missing_rows, ~missing_cols]

    if filtered_data.empty:
        print("Warning: No data available after filtering missing rows/columns")
        return

    # Create figure and axis
    plt.figure(figsize=(12, 10))

    # Create heatmap
    mask = filtered_data.isnull()
    sns.heatmap(
        filtered_data,
        annot=True,
        fmt=".3f",
        cmap="RdYlBu_r",
        center=0,
        mask=mask,
        square=True,
        cbar_kws={"shrink": 0.8, "label": "Test MCC Score"},
        xticklabels=[species.replace("_", " ") for species in filtered_data.columns],
        yticklabels=[species.replace("_", " ") for species in filtered_data.index],
        linewidths=0.5,
        linecolor="white",
    )

    plt.title("Test MCC Scores for Bacterial Pair Models\n(Filtered - Missing data excluded)", fontsize=16, pad=20)
    plt.xlabel("Test Bacteria", fontsize=12)
    plt.ylabel("Training Bacteria", fontsize=12)

    # Rotate labels for better readability
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    output_file = output_dir / "bacterial_pairs_mcc_heatmap_filtered.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Filtered heatmap saved to: {output_file}")

    # Also save as PDF
    output_file_pdf = output_dir / "bacterial_pairs_mcc_heatmap_filtered.pdf"
    plt.savefig(output_file_pdf, bbox_inches="tight")
    print(f"Filtered heatmap saved to: {output_file_pdf}")

    plt.show()


def save_summary_csv(heatmap_data: pd.DataFrame, output_dir: Path) -> None:
    """
    Save the MCC scores as a CSV file.

    Args:
        heatmap_data: DataFrame with MCC scores
        output_dir: Directory to save the CSV
    """
    output_file = output_dir / "bacterial_pairs_mcc_scores.csv"
    heatmap_data.to_csv(output_file)
    print(f"MCC scores saved to: {output_file}")


def print_summary_statistics(heatmap_data: pd.DataFrame) -> None:
    """
    Print summary statistics of the MCC scores.

    Args:
        heatmap_data: DataFrame with MCC scores
    """
    # Get all non-null values
    all_scores = heatmap_data.values.flatten()
    all_scores = all_scores[~np.isnan(all_scores)]

    if len(all_scores) == 0:
        print("No MCC scores found!")
        return

    print("\n" + "=" * 50)
    print("SUMMARY STATISTICS")
    print("=" * 50)
    print(f"Total number of bacterial pairs evaluated: {len(all_scores)}")
    print(f"Mean MCC score: {np.mean(all_scores):.4f}")
    print(f"Median MCC score: {np.median(all_scores):.4f}")
    print(f"Standard deviation: {np.std(all_scores):.4f}")
    print(f"Min MCC score: {np.min(all_scores):.4f}")
    print(f"Max MCC score: {np.max(all_scores):.4f}")

    # Find best and worst performing pairs
    max_idx = np.unravel_index(np.nanargmax(heatmap_data.values), heatmap_data.shape)
    min_idx = np.unravel_index(np.nanargmin(heatmap_data.values), heatmap_data.shape)

    best_train = heatmap_data.index[max_idx[0]]
    best_test = heatmap_data.columns[max_idx[1]]
    best_score = heatmap_data.iloc[max_idx[0], max_idx[1]]

    worst_train = heatmap_data.index[min_idx[0]]
    worst_test = heatmap_data.columns[min_idx[1]]
    worst_score = heatmap_data.iloc[min_idx[0], min_idx[1]]

    print("\nBest performing pair:")
    print(f"  Train: {best_train}, Test: {best_test}, MCC: {best_score:.4f}")
    print("Worst performing pair:")
    print(f"  Train: {worst_train}, Test: {worst_test}, MCC: {worst_score:.4f}")


def analyze_missing_data(results: Dict[str, Dict[str, float]]) -> None:
    """
    Analyze and report missing data patterns.

    Args:
        results: Nested dictionary with MCC scores
    """
    print("\n" + "=" * 50)
    print("MISSING DATA ANALYSIS")
    print("=" * 50)

    missing_pairs = []
    total_pairs = len(SPECIES) * len(SPECIES)
    found_pairs = 0

    for bacteria1 in SPECIES:
        for bacteria2 in SPECIES:
            if bacteria1 in results and bacteria2 in results[bacteria1]:
                found_pairs += 1
            else:
                missing_pairs.append(f"{bacteria1} -> {bacteria2}")

    print(f"Total possible bacterial pairs: {total_pairs}")
    print(f"Found pairs with data: {found_pairs}")
    print(f"Missing pairs: {len(missing_pairs)}")
    print(f"Data coverage: {found_pairs / total_pairs * 100:.1f}%")

    # Analyze missing patterns by bacteria
    missing_as_train = {}
    missing_as_test = {}

    for bacteria in SPECIES:
        missing_as_train[bacteria] = 0
        missing_as_test[bacteria] = 0

    for pair in missing_pairs:
        train_bacteria, test_bacteria = pair.split(" -> ")
        missing_as_train[train_bacteria] += 1
        missing_as_test[test_bacteria] += 1

    print("\nMissing data by training bacteria:")
    for bacteria, count in missing_as_train.items():
        if count > 0:
            print(f"  {bacteria}: {count}/{len(SPECIES)} missing ({count / len(SPECIES) * 100:.1f}%)")

    print("\nMissing data by test bacteria:")
    for bacteria, count in missing_as_test.items():
        if count > 0:
            print(f"  {bacteria}: {count}/{len(SPECIES)} missing ({count / len(SPECIES) * 100:.1f}%)")


def main():
    """Main function to run the evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate double bacteria AMR prediction models and create heatmap")
    parser.add_argument("results_dir", type=str, help="Path to the directory containing the bacterial pair results")
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Directory to save output files (default: same as results_dir)"
    )

    args = parser.parse_args()

    # Convert to Path objects
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir

    # Validate input directory
    if not results_dir.exists():
        print(f"Error: Results directory {results_dir} does not exist")
        return

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Collecting results from: {results_dir}")
    print(f"Output will be saved to: {output_dir}")
    # Collect results
    results = collect_double_bacteria_results(results_dir)

    # Analyze missing data patterns
    analyze_missing_data(results)

    # Create heatmap dataframe
    heatmap_data = create_heatmap_dataframe(results)

    # Print summary statistics
    print_summary_statistics(heatmap_data)

    # Analyze missing data patterns
    analyze_missing_data(results)

    # Create and save heatmap
    create_heatmap(heatmap_data, output_dir)

    # Save summary CSV
    save_summary_csv(heatmap_data, output_dir)

    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()
