#!/usr/bin/env python3
"""
Model Evaluation Script for AMR Prediction Models

This script evaluates model performance by extracting test MCC scores from
metrics.csv files across different model types (esm, feature_importance,
genomeocean, variance) and versions (0-3) for various bacterial species.

Usage:
    python evaluate_model.py /path/to/results/directory

The script expects the following directory structure:
/path/to/results/directory/
├── {model_type}_{species}/
│   ├── version_0/
│   │   └── metrics.csv
│   ├── version_1/
│   │   └── metrics.csv
│   ├── version_2/
│   │   └── metrics.csv
│   └── version_3/
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

# Define the species and model types
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

MODEL_TYPES = ["esm", "feature_importance", "genomeocean", "variance"]

VERSIONS = ["version_0", "version_1", "version_2", "version_3"]


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


def collect_results(results_dir: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Collect test MCC results from all models, species, and versions.

    Args:
        results_dir: Path to the results directory

    Returns:
        Nested dictionary: {model_type: {species: {version: mcc_score}}}
    """
    results = {}

    for model_type in MODEL_TYPES:
        results[model_type] = {}

        for species in SPECIES:
            folder_name = f"{model_type}_{species}"
            model_dir = results_dir / folder_name

            if not model_dir.exists():
                print(f"Warning: Directory {model_dir} not found")
                continue

            results[model_type][species] = {}

            for version in VERSIONS:
                version_dir = model_dir / version
                metrics_file = version_dir / "metrics.csv"

                if not metrics_file.exists():
                    print(f"Warning: Metrics file {metrics_file} not found")
                    continue

                mcc_score = extract_last_test_mcc(metrics_file)
                if mcc_score is not None:
                    results[model_type][species][version] = mcc_score

    return results


def create_summary_dataframe(results: Dict) -> pd.DataFrame:
    """
    Create a summary DataFrame from the results dictionary.

    Args:
        results: Nested dictionary with MCC scores

    Returns:
        DataFrame with columns: Model, Species, Version, Test_MCC
    """
    summary_data = []

    for model_type, species_data in results.items():
        for species, version_data in species_data.items():
            for version, mcc_score in version_data.items():
                summary_data.append(
                    {"Model": model_type, "Species": species, "Version": version, "Test_MCC": mcc_score}
                )

    return pd.DataFrame(summary_data)


def create_pivot_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a pivot table for easier comparison across models and versions.

    Args:
        df: Summary DataFrame

    Returns:
        Pivot table with Models as index and Species_Version as columns
    """
    # Create a combined column for species and version
    df["Species_Version"] = df["Species"] + "_" + df["Version"]

    pivot = df.pivot_table(index="Model", columns="Species_Version", values="Test_MCC", aggfunc="mean")

    return pivot


def generate_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary statistics for each model across all species and versions.

    Args:
        df: Summary DataFrame

    Returns:
        DataFrame with statistics for each model
    """
    stats = df.groupby("Model")["Test_MCC"].agg(["count", "mean", "std", "min", "max", "median"]).round(4)

    stats.columns = ["N_Experiments", "Mean_MCC", "Std_MCC", "Min_MCC", "Max_MCC", "Median_MCC"]
    return stats


def plot_model_comparison(df: pd.DataFrame, output_dir: Path):
    """
    Create visualizations comparing model performance.

    Args:
        df: Summary DataFrame
        output_dir: Directory to save plots
    """
    # Set up the plotting style
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")

    # 1. Box plot comparing models
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="Model", y="Test_MCC")
    plt.title("Test MCC Distribution by Model Type")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison_boxplot.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Heatmap of average performance by species
    plt.figure(figsize=(14, 8))
    species_model_avg = df.groupby(["Species", "Model"])["Test_MCC"].mean().unstack()
    sns.heatmap(species_model_avg, annot=True, fmt=".3f", cmap="RdYlBu_r", center=0)
    plt.title("Average Test MCC by Species and Model")
    plt.xlabel("Model Type")
    plt.ylabel("Species")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / "species_model_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2b. Heatmap of best (maximum) performance by species
    plt.figure(figsize=(14, 8))
    species_model_best = df.groupby(["Species", "Model"])["Test_MCC"].max().unstack()
    sns.heatmap(species_model_best, annot=True, fmt=".3f", cmap="RdYlBu_r", center=0)
    plt.title("Best Test MCC by Species and Model")
    plt.xlabel("Model Type")
    plt.ylabel("Species")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / "species_model_best_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()

    return species_model_best  # Return the best performance matrix for saving

    # 3. Performance by version
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="Version", y="Test_MCC", hue="Model")
    plt.title("Test MCC Distribution by Version and Model")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "version_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 4. Species-specific performance
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()

    for i, species in enumerate(SPECIES):
        species_data = df[df["Species"] == species]
        if not species_data.empty:
            sns.barplot(data=species_data, x="Model", y="Test_MCC", ax=axes[i])
            axes[i].set_title(f"{species.replace('_', ' ')}")
            axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45)
            axes[i].set_ylim(-1, 1)  # MCC ranges from -1 to 1

    plt.tight_layout()
    plt.savefig(output_dir / "species_specific_performance.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_results(
    df: pd.DataFrame, pivot: pd.DataFrame, stats: pd.DataFrame, best_performance: pd.DataFrame, output_dir: Path
):
    """
    Save results to CSV files.

    Args:
        df: Summary DataFrame
        pivot: Pivot table
        stats: Statistics DataFrame
        best_performance: Best performance matrix by species and model
        output_dir: Directory to save files
    """
    # Save detailed results
    df.to_csv(output_dir / "detailed_results.csv", index=False)

    # Save pivot table
    pivot.to_csv(output_dir / "pivot_table.csv")

    # Save statistics
    stats.to_csv(output_dir / "model_statistics.csv")

    # Save best performance matrix
    best_performance.to_csv(output_dir / "best_performance_by_species_model.csv")
    # Also save as TSV for convenience
    best_performance.to_csv(output_dir / "best_performance_by_species_model.tsv", sep="\t")

    # Save best performers
    best_overall = df.loc[df["Test_MCC"].idxmax()]
    best_by_model = df.groupby("Model")["Test_MCC"].max().sort_values(ascending=False)
    best_by_species = df.groupby("Species")["Test_MCC"].max().sort_values(ascending=False)

    with open(output_dir / "best_performers.txt", "w") as f:
        f.write("BEST PERFORMERS SUMMARY\n")
        f.write("=" * 50 + "\n\n")

        f.write("Overall Best Performance:\n")
        f.write(f"Model: {best_overall['Model']}\n")
        f.write(f"Species: {best_overall['Species']}\n")
        f.write(f"Version: {best_overall['Version']}\n")
        f.write(f"Test MCC: {best_overall['Test_MCC']:.4f}\n\n")

        f.write("Best Performance by Model:\n")
        for model, mcc in best_by_model.items():
            f.write(f"{model}: {mcc:.4f}\n")

        f.write("\nBest Performance by Species:\n")
        for species, mcc in best_by_species.items():
            f.write(f"{species}: {mcc:.4f}\n")


def main():
    """
    Main function to parse arguments and run the evaluation.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate AMR prediction model performance across versions and model types",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("results_dir", type=str, help="Path to the results directory containing model outputs")

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./evaluation_output",
        help="Directory to save evaluation results (default: ./evaluation_output)",
    )

    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots")

    args = parser.parse_args()

    # Convert paths to Path objects
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    # Validate input directory
    if not results_dir.exists():
        print(f"Error: Results directory {results_dir} does not exist")
        return 1

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Collecting results from: {results_dir}")
    print(f"Output will be saved to: {output_dir}")

    # Collect results
    results = collect_results(results_dir)

    if not any(results.values()):
        print("Error: No results found. Please check the directory structure and file paths.")
        return 1

    # Create summary DataFrame
    df = create_summary_dataframe(results)

    if df.empty:
        print("Error: No valid MCC scores found.")
        return 1

    print(f"\nFound {len(df)} valid experiments")
    print(f"Models: {df['Model'].unique()}")
    print(f"Species: {df['Species'].unique()}")
    print(f"Versions: {df['Version'].unique()}")

    # Generate pivot table and statistics
    pivot = create_pivot_table(df)
    stats = generate_statistics(df)

    # Display summary statistics
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE SUMMARY")
    print("=" * 60)
    print(stats)

    # Generate best performance matrix
    best_performance = df.groupby(["Species", "Model"])["Test_MCC"].max().unstack()

    # Save results
    save_results(df, pivot, stats, best_performance, output_dir)

    # Generate plots if requested
    if not args.no_plots:
        print("\nGenerating visualizations...")
        plot_model_comparison(df, output_dir)
        print("Plots saved successfully!")

    print(f"\nEvaluation complete! Results saved to: {output_dir}")
    print("\nFiles generated:")
    print("- detailed_results.csv: All MCC scores with metadata")
    print("- pivot_table.csv: Matrix view of results")
    print("- model_statistics.csv: Summary statistics by model")
    print("- best_performance_by_species_model.csv/.tsv: Best MCC by species and model")
    print("- best_performers.txt: Best performing configurations")
    if not args.no_plots:
        print("- *.png: Various visualization plots")

    return 0


if __name__ == "__main__":
    exit(main())
