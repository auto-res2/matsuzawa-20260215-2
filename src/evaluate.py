"""Evaluation script for comparing multiple runs."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
import wandb
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


# [VALIDATOR FIX - Attempt 1]
# [PROBLEM]: evaluate.py was being called with key=value syntax but expected --key value
# [CAUSE]: Workflow passes results_dir="..." and run_ids="..." but argparse expects --results_dir ... --run_ids ...
# [FIX]: Modified argument parser to accept both formats by preprocessing sys.argv
#
# [OLD CODE]:
# def parse_args():
#     """Parse command line arguments."""
#     parser = argparse.ArgumentParser(description="Evaluate and compare experiment runs")
#     parser.add_argument("--results_dir", type=str, required=True, help="Results directory path")
#     parser.add_argument("--run_ids", type=str, required=True, help="JSON string list of run IDs")
#     parser.add_argument("--wandb_entity", type=str, default="airas", help="WandB entity")
#     parser.add_argument("--wandb_project", type=str, default="2026-02-15-2", help="WandB project")
#     return parser.parse_args()
#
# [NEW CODE]:
def parse_args():
    """Parse command line arguments.
    
    Supports both formats:
    - --key value (standard argparse)
    - key=value (Hydra-style, used by workflow)
    """
    # Preprocess sys.argv to convert key=value to --key value
    processed_argv = []
    for arg in sys.argv[1:]:
        if '=' in arg and not arg.startswith('--'):
            # Split key=value into --key value
            key, value = arg.split('=', 1)
            processed_argv.append(f'--{key}')
            processed_argv.append(value)
        else:
            processed_argv.append(arg)
    
    parser = argparse.ArgumentParser(description="Evaluate and compare experiment runs")
    parser.add_argument("--results_dir", type=str, required=True, help="Results directory path")
    parser.add_argument("--run_ids", type=str, required=True, help="JSON string list of run IDs")
    parser.add_argument("--wandb_entity", type=str, default="airas", help="WandB entity")
    parser.add_argument("--wandb_project", type=str, default="2026-02-15-2", help="WandB project")
    
    # Parse the preprocessed arguments
    return parser.parse_args(processed_argv)


def fetch_wandb_run_data(entity: str, project: str, run_id: str) -> Dict[str, Any]:
    """Fetch run data from WandB API.
    
    Args:
        entity: WandB entity
        project: WandB project
        run_id: Run ID
    
    Returns:
        Dictionary with run history, summary, and config
    """
    api = wandb.Api()
    
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
        
        # Fetch history (time series data)
        history = run.history()
        history_dict = history.to_dict(orient='list') if not history.empty else {}
        
        # [VALIDATOR FIX - Attempt 2]
        # [PROBLEM]: TypeError: Object of type SummarySubDict is not JSON serializable
        # [CAUSE]: run.summary returns a SummarySubDict object which cannot be directly serialized to JSON;
        #          dict(run.summary) creates a shallow copy that may still contain nested SummarySubDict objects
        # [FIX]: Use json.loads(json.dumps(..., default=str)) to recursively convert all values to JSON-serializable types
        #
        # [OLD CODE]:
        # summary = dict(run.summary)
        #
        # [NEW CODE]:
        # Convert summary to JSON-serializable dict by first converting to JSON string and back
        # Use default=str as fallback for any non-serializable objects
        summary_raw = dict(run.summary)
        summary = json.loads(json.dumps(summary_raw, default=str))
        
        # Fetch config
        config = dict(run.config)
        
        return {
            "history": history_dict,
            "summary": summary,
            "config": config
        }
    except Exception as e:
        print(f"Warning: Could not fetch WandB data for {run_id}: {e}")
        return {"history": {}, "summary": {}, "config": {}}


def load_local_metrics(results_dir: Path, run_id: str) -> Dict[str, Any]:
    """Load metrics from local results directory.
    
    Args:
        results_dir: Results directory
        run_id: Run ID
    
    Returns:
        Dictionary with metrics
    """
    metrics_file = results_dir / run_id / "metrics.json"
    
    if metrics_file.exists():
        with open(metrics_file, "r") as f:
            return json.load(f)
    
    # Fallback: try to load from results.json
    results_file = results_dir / run_id / "results.json"
    if results_file.exists():
        with open(results_file, "r") as f:
            data = json.load(f)
            return data.get("metrics", {})
    
    return {}


def create_comparison_plots(
    results_dir: Path,
    run_data: Dict[str, Dict[str, Any]],
    run_ids: List[str]
) -> List[str]:
    """Create comparison plots for all runs.
    
    Args:
        results_dir: Results directory
        run_data: Dictionary mapping run_id to data
        run_ids: List of run IDs
    
    Returns:
        List of generated file paths
    """
    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    generated_files = []
    
    # Define colors for each run
    colors = plt.cm.tab10(np.linspace(0, 1, len(run_ids)))
    
    # Plot 1: Accuracy comparison (from history if available)
    plt.figure(figsize=(10, 6))
    has_accuracy_history = False
    
    for idx, run_id in enumerate(run_ids):
        history = run_data[run_id].get("history", {})
        if "accuracy" in history and history["accuracy"]:
            accuracy_values = history["accuracy"]
            sample_indices = history.get("sample_idx", list(range(len(accuracy_values))))
            plt.plot(sample_indices, accuracy_values, label=run_id, color=colors[idx], linewidth=2)
            has_accuracy_history = True
    
    if has_accuracy_history:
        plt.xlabel("Sample Index", fontsize=12)
        plt.ylabel("Accuracy", fontsize=12)
        plt.title("Accuracy Over Time (Cumulative)", fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        accuracy_plot = comparison_dir / "comparison_accuracy.pdf"
        plt.savefig(accuracy_plot, format='pdf', bbox_inches='tight')
        print(f"Generated: {accuracy_plot}")
        generated_files.append(str(accuracy_plot))
        plt.close()
    
    # Plot 2: Final accuracy bar chart
    plt.figure(figsize=(10, 6))
    final_accuracies = []
    labels = []
    
    for run_id in run_ids:
        summary = run_data[run_id].get("summary", {})
        local_metrics = run_data[run_id].get("local_metrics", {})
        
        accuracy = summary.get("accuracy", local_metrics.get("accuracy", 0.0))
        final_accuracies.append(accuracy)
        labels.append(run_id)
    
    bars = plt.bar(range(len(labels)), final_accuracies, color=colors[:len(labels)])
    plt.xlabel("Run ID", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Final Accuracy Comparison", fontsize=14, fontweight='bold')
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    accuracy_bar_plot = comparison_dir / "comparison_accuracy_bar.pdf"
    plt.savefig(accuracy_bar_plot, format='pdf', bbox_inches='tight')
    print(f"Generated: {accuracy_bar_plot}")
    generated_files.append(str(accuracy_bar_plot))
    plt.close()
    
    # Plot 3: Average calls per problem (if available)
    plt.figure(figsize=(10, 6))
    avg_calls = []
    call_labels = []
    
    for run_id in run_ids:
        summary = run_data[run_id].get("summary", {})
        local_metrics = run_data[run_id].get("local_metrics", {})
        
        calls = summary.get("avg_calls_per_problem", local_metrics.get("avg_calls_per_problem", 0.0))
        if calls > 0:
            avg_calls.append(calls)
            call_labels.append(run_id)
    
    if avg_calls:
        bars = plt.bar(range(len(call_labels)), avg_calls, color=colors[:len(call_labels)])
        plt.xlabel("Run ID", fontsize=12)
        plt.ylabel("Average Calls per Problem", fontsize=12)
        plt.title("Inference Cost Comparison", fontsize=14, fontweight='bold')
        plt.xticks(range(len(call_labels)), call_labels, rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        calls_plot = comparison_dir / "comparison_calls.pdf"
        plt.savefig(calls_plot, format='pdf', bbox_inches='tight')
        print(f"Generated: {calls_plot}")
        generated_files.append(str(calls_plot))
        plt.close()
    
    return generated_files


def create_per_run_plots(
    results_dir: Path,
    run_id: str,
    run_data: Dict[str, Any]
) -> List[str]:
    """Create per-run plots.
    
    Args:
        results_dir: Results directory
        run_id: Run ID
        run_data: Run data
    
    Returns:
        List of generated file paths
    """
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    generated_files = []
    history = run_data.get("history", {})
    
    # Plot accuracy over time
    if "accuracy" in history and history["accuracy"]:
        plt.figure(figsize=(10, 6))
        accuracy_values = history["accuracy"]
        sample_indices = history.get("sample_idx", list(range(len(accuracy_values))))
        plt.plot(sample_indices, accuracy_values, linewidth=2, color='blue')
        plt.xlabel("Sample Index", fontsize=12)
        plt.ylabel("Cumulative Accuracy", fontsize=12)
        plt.title(f"Accuracy Over Time - {run_id}", fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        accuracy_plot = run_dir / f"{run_id}_accuracy.pdf"
        plt.savefig(accuracy_plot, format='pdf', bbox_inches='tight')
        print(f"Generated: {accuracy_plot}")
        generated_files.append(str(accuracy_plot))
        plt.close()
    
    return generated_files


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Parse run_ids from JSON string
    try:
        run_ids = json.loads(args.run_ids)
    except json.JSONDecodeError as e:
        print(f"Error parsing run_ids JSON: {e}")
        sys.exit(1)
    
    if not isinstance(run_ids, list) or len(run_ids) == 0:
        print("Error: run_ids must be a non-empty list")
        sys.exit(1)
    
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Evaluating runs: {run_ids}")
    print(f"Results directory: {results_dir}")
    
    # [VALIDATOR FIX - Attempt 1]
    # [PROBLEM]: Visualization stage may be called before any runs complete
    # [CAUSE]: Workflow can trigger visualization even when no run data exists
    # [FIX]: Check if any run directories exist and warn if data is missing
    #
    # Check if any run directories exist
    has_any_data = False
    for run_id in run_ids:
        run_dir = results_dir / run_id
        metrics_file = results_dir / run_id / "metrics.json"
        results_file = results_dir / run_id / "results.json"
        if run_dir.exists() and (metrics_file.exists() or results_file.exists()):
            has_any_data = True
            break
    
    if not has_any_data:
        print("\n" + "=" * 80)
        print("WARNING: No run data found in results directory.")
        print("This may be because:")
        print("  1. The main experiment runs haven't completed yet")
        print("  2. Run directories don't exist yet")
        print("  3. Runs failed before generating any output")
        print("\nThe script will still generate placeholder outputs.")
        print("=" * 80 + "\n")
    
    # Fetch data for all runs
    run_data = {}
    for run_id in run_ids:
        print(f"\nProcessing run: {run_id}")
        
        # Try WandB first
        wandb_data = fetch_wandb_run_data(args.wandb_entity, args.wandb_project, run_id)
        
        # Load local metrics
        local_metrics = load_local_metrics(results_dir, run_id)
        
        # Combine data
        run_data[run_id] = {
            "history": wandb_data.get("history", {}),
            "summary": wandb_data.get("summary", {}),
            "config": wandb_data.get("config", {}),
            "local_metrics": local_metrics
        }
        
        # Export per-run metrics
        metrics_output = results_dir / run_id / "metrics.json"
        metrics_output.parent.mkdir(parents=True, exist_ok=True)
        
        # Merge summary and local metrics
        all_metrics = {**local_metrics, **run_data[run_id]["summary"]}
        
        with open(metrics_output, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"Exported metrics to: {metrics_output}")
        
        # Create per-run plots
        per_run_plots = create_per_run_plots(results_dir, run_id, run_data[run_id])
    
    # Aggregate metrics
    aggregated = {
        "primary_metric": "accuracy",
        "metrics": {}
    }
    
    proposed_accuracies = []
    baseline_accuracies = []
    
    for run_id in run_ids:
        summary = run_data[run_id].get("summary", {})
        local_metrics = run_data[run_id].get("local_metrics", {})
        
        # Get key metrics
        accuracy = summary.get("accuracy", local_metrics.get("accuracy", 0.0))
        avg_calls = summary.get("avg_calls_per_problem", local_metrics.get("avg_calls_per_problem", 0.0))
        
        aggregated["metrics"][run_id] = {
            "accuracy": accuracy,
            "avg_calls_per_problem": avg_calls
        }
        
        # Categorize runs
        if "proposed" in run_id:
            proposed_accuracies.append(accuracy)
        else:
            baseline_accuracies.append(accuracy)
    
    # Calculate best proposed and baseline
    if proposed_accuracies:
        aggregated["best_proposed"] = max(proposed_accuracies)
    if baseline_accuracies:
        aggregated["best_baseline"] = max(baseline_accuracies)
    
    # Calculate gap
    if proposed_accuracies and baseline_accuracies:
        aggregated["gap"] = aggregated["best_proposed"] - aggregated["best_baseline"]
    
    # Save aggregated metrics
    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    aggregated_file = comparison_dir / "aggregated_metrics.json"
    
    with open(aggregated_file, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"\nAggregated metrics saved to: {aggregated_file}")
    
    # Create comparison plots
    print("\nGenerating comparison plots...")
    comparison_plots = create_comparison_plots(results_dir, run_data, run_ids)
    
    # Summary
    print("\n" + "=" * 80)
    print("Evaluation Summary:")
    print("=" * 80)
    print(f"Runs evaluated: {len(run_ids)}")
    print(f"Primary metric: {aggregated['primary_metric']}")
    
    if "best_proposed" in aggregated:
        print(f"Best proposed: {aggregated['best_proposed']:.4f}")
    if "best_baseline" in aggregated:
        print(f"Best baseline: {aggregated['best_baseline']:.4f}")
    if "gap" in aggregated:
        print(f"Gap (proposed - baseline): {aggregated['gap']:.4f}")
    
    print("\nGenerated files:")
    print(f"  - {aggregated_file}")
    for plot_file in comparison_plots:
        print(f"  - {plot_file}")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
