"""Main entry point for CPV-CoT experiments."""

import sys
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from src.inference import run_inference


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main orchestrator for CPV-CoT experiments.
    
    This script:
    1. Loads Hydra configuration
    2. Applies mode-specific overrides
    3. Executes inference
    
    Args:
        cfg: Hydra configuration object
    """
    # Print config for debugging
    print("=" * 80)
    print("Configuration:")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)
    
    # Validate required fields
    if cfg.run.run_id is None:
        raise ValueError("run_id is required. Use run=<run_config> to specify.")
    
    # Create cache directory
    Path(cfg.cache_dir).mkdir(parents=True, exist_ok=True)
    
    # Run inference
    print(f"\nStarting inference for run_id: {cfg.run.run_id}")
    print(f"Mode: {cfg.mode}")
    print(f"Method: {cfg.run.method}")
    print(f"Dataset: {cfg.run.dataset.name}")
    print(f"Model: {cfg.run.model.name}")
    
    try:
        metrics = run_inference(cfg)
        print("\n" + "=" * 80)
        print("Inference completed successfully!")
        print(f"Final metrics: {metrics}")
        print("=" * 80)
    except Exception as e:
        print(f"\nError during inference: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
