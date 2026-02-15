"""Inference script for math problem solving."""

import os
import json
import sys
from pathlib import Path
from typing import Dict, Any, List
from omegaconf import DictConfig, OmegaConf
import wandb
from tqdm import tqdm

from src.preprocess import load_math_dataset, is_correct
from src.model import MathSolver


def run_inference(cfg: DictConfig) -> Dict[str, Any]:
    """Run inference on a dataset with specified method.
    
    Args:
        cfg: Hydra configuration
    
    Returns:
        Dictionary with results and metrics
    """
    # Extract config
    run_id = cfg.run.run_id
    method = cfg.run.method
    model_cfg = cfg.run.model
    dataset_cfg = cfg.run.dataset
    inference_cfg = cfg.run.inference
    mode = cfg.mode
    cache_dir = cfg.cache_dir
    results_dir = Path(cfg.results_dir) / run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Apply mode-specific overrides
    if mode == "sanity_check":
        # Reduce samples for quick check
        max_samples = 10
        wandb_project = f"{cfg.wandb.project}-sanity"
    else:
        max_samples = dataset_cfg.get("max_samples", None)
        wandb_project = cfg.wandb.project
    
    # Initialize WandB
    if cfg.wandb.mode != "disabled":
        wandb.init(
            entity=cfg.wandb.entity,
            project=wandb_project,
            id=run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
            resume="allow",
            mode=cfg.wandb.mode
        )
        print(f"WandB run URL: {wandb.run.get_url()}")
    
    # Load dataset
    print(f"Loading dataset: {dataset_cfg.name}")
    dataset = load_math_dataset(
        dataset_name=dataset_cfg.name,
        split=dataset_cfg.split,
        cache_dir=cache_dir,
        max_samples=max_samples
    )
    print(f"Loaded {len(dataset)} samples")
    
    # Initialize model
    print(f"Initializing model: {model_cfg.name}")
    solver = MathSolver(
        model_name=model_cfg.name,
        cache_dir=cache_dir,
        device="auto"
    )
    
    # Run inference
    print(f"Running inference with method: {method}")
    results = []
    correct_count = 0
    total_calls = 0
    
    for idx, item in enumerate(tqdm(dataset, desc="Solving problems")):
        question = item["question"]
        ground_truth = item["answer"]
        
        # Solve based on method
        if method == "zero_shot_cot":
            solution = solver.solve_zero_shot_cot(
                question=question,
                max_tokens=model_cfg.max_tokens,
                temperature=model_cfg.temperature
            )
        elif method == "cpv_cot":
            solution = solver.solve_cpv_cot(
                question=question,
                max_tokens=model_cfg.max_tokens,
                temperature=model_cfg.temperature
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Check correctness
        predicted_answer = solution["answer"]
        correct = is_correct(predicted_answer, ground_truth)
        
        # Store result
        result = {
            "id": item["id"],
            "question": question,
            "ground_truth": ground_truth,
            "predicted_answer": predicted_answer,
            "correct": correct,
            "solution": solution
        }
        results.append(result)
        
        if correct:
            correct_count += 1
        total_calls += solution.get("num_calls", 1)
        
        # Log to WandB
        if cfg.wandb.mode != "disabled":
            wandb.log({
                "sample_idx": idx,
                "correct": int(correct),
                "accuracy": correct_count / (idx + 1),
                "avg_calls": total_calls / (idx + 1)
            })
    
    # Calculate metrics
    accuracy = correct_count / len(dataset) if len(dataset) > 0 else 0.0
    avg_calls = total_calls / len(dataset) if len(dataset) > 0 else 0.0
    
    metrics = {
        "accuracy": accuracy,
        "correct_count": correct_count,
        "total_samples": len(dataset),
        "avg_calls_per_problem": avg_calls,
        "total_calls": total_calls
    }
    
    print(f"\nResults:")
    print(f"  Accuracy: {accuracy:.4f} ({correct_count}/{len(dataset)})")
    print(f"  Avg calls per problem: {avg_calls:.2f}")
    
    # Save results
    results_file = results_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump({
            "run_id": run_id,
            "method": method,
            "dataset": dataset_cfg.name,
            "metrics": metrics,
            "results": results
        }, f, indent=2)
    print(f"Results saved to: {results_file}")
    
    # Save metrics separately for easy access
    metrics_file = results_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Log final metrics to WandB
    if cfg.wandb.mode != "disabled":
        for key, value in metrics.items():
            wandb.summary[key] = value
        wandb.finish()
    
    # Sanity validation for sanity_check mode
    if mode == "sanity_check":
        validate_sanity_check(metrics, results)
    
    return metrics


def validate_sanity_check(metrics: Dict[str, Any], results: List[Dict[str, Any]]) -> None:
    """Validate sanity check results.
    
    Args:
        metrics: Computed metrics
        results: Individual results
    """
    total_samples = metrics.get("total_samples", 0)
    accuracy = metrics.get("accuracy", 0.0)
    
    # Check: at least 5 samples processed
    if total_samples < 5:
        print(f"SANITY_VALIDATION: FAIL reason=insufficient_samples")
        print(f'SANITY_VALIDATION_SUMMARY: {{"samples": {total_samples}, "status": "insufficient"}}')
        sys.exit(1)
    
    # Check: metrics are finite
    if not all(isinstance(v, (int, float)) and not (isinstance(v, float) and (v != v or abs(v) == float('inf'))) 
               for v in metrics.values()):
        print(f"SANITY_VALIDATION: FAIL reason=invalid_metrics")
        print(f'SANITY_VALIDATION_SUMMARY: {{"samples": {total_samples}, "status": "invalid_metrics"}}')
        sys.exit(1)
    
    # Check: not all outputs are identical (basic diversity check)
    predicted_answers = [r["predicted_answer"] for r in results]
    unique_answers = len(set(predicted_answers))
    if unique_answers == 1 and total_samples > 1:
        print(f"SANITY_VALIDATION: FAIL reason=no_output_diversity")
        print(f'SANITY_VALIDATION_SUMMARY: {{"samples": {total_samples}, "unique_answers": {unique_answers}, "status": "no_diversity"}}')
        sys.exit(1)
    
    # Success
    print(f"SANITY_VALIDATION: PASS")
    print(f'SANITY_VALIDATION_SUMMARY: {{"samples": {total_samples}, "accuracy": {accuracy:.4f}, "unique_answers": {unique_answers}, "status": "pass"}}')


if __name__ == "__main__":
    # This script is called from main.py with config already loaded
    # For standalone testing, you would need to add Hydra decorator
    print("This script should be called from main.py")
    sys.exit(1)
