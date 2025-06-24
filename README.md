# adapters-vs-lora-cifar10

This repository provides a modular, reproducible, and enterprise-grade framework for benchmarking Adapter and LoRA architectures against a ViT-Tiny baseline on the CIFAR-10 dataset. Designed for research and production, it features robust logging, error handling, and statistical reporting.

## Features
- Modular codebase with clear separation of concerns
- Baseline, Adapter, and LoRA model variants
- Sophisticated logging and exception handling
- Reproducible experiments (seed management, config)
- Comprehensive statistics and visualization utilities
- Extensible for new models, datasets, or experiment types
- Ready for CI/CD and large-scale benchmarking

## Project Structure
```
adapters_vs_lora_cifar10/
  config.py        # Experiment configuration and seed management
  data.py          # Data loading, transformation, and validation
  visualize.py     # Visualization utilities
  model/
    model_utils.py # Model architectures (ViT, Adapters, LoRA integration)
    train_utils.py # Training and evaluation logic
  logger.py        # Logging and error handling
  stats.py         # Statistics and metrics
scripts/
  run_experiment.py # Entry point for running all experiments
tests/             # Unit and integration tests
requirements.txt   # Python dependencies
README.md          # Project documentation
```

## Getting Started
1. Clone the repository
2. Create and activate a virtual environment:
   - Windows: `python -m venv .venv && .venv\Scripts\activate`
   - Linux/Mac: `python3 -m venv .venv && source .venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Run all experiments (from project root):
   - `python -m scripts.run_experiment`

## Experiment Variants
- **Baseline**: ViT-Tiny + classification head
- **Adapter**: ViT-Tiny with integrated Adapter layers (train only adapters and head)
- **LoRA**: ViT-Tiny with LoRA applied to QKV projections (requires `peft`)

Results are saved in `results/all_results.json`.

## Extending the Framework
- Add new model variants in `adapters_vs_lora_cifar10/model/model_utils.py`
- Add new training/evaluation logic in `train_utils.py`
- Add new datasets in `data.py`
- Add new experiment scripts in `scripts/`

## Troubleshooting
- Always run scripts as modules from the project root (e.g., `python -m scripts.run_experiment`)
- If you see `ModuleNotFoundError`, check your working directory and PYTHONPATH
- For LoRA experiments, ensure `peft` is installed (`pip install peft`)

## License
MIT
