"""
Entry point for running Adapter vs LoRA experiments on CIFAR-10.
"""
import logging
from adapters_vs_lora_cifar10 import config, logger, data, visualize, stats
from adapters_vs_lora_cifar10.model.model_utils import load_vit_tiny, ClassificationHead, integrate_adaptors, integrate_lora
from adapters_vs_lora_cifar10.model.train_utils import train_model, evaluate_model, count_trainable_parameters
import torch
import torch.nn as nn
import torch.optim as optim
import os
import json

# LoRA imports
try:
    from peft import LoraConfig, get_peft_model
    peft_available = True
except ImportError:
    peft_available = False

def run_experiment(
    variant: str,
    device: torch.device,
    train_loader,
    val_loader,
    test_loader,
    bottleneck_dim: int = 32,
    lora_rank: int = 8,
    epochs: int = 5
) -> dict:
    """
    Run a single experiment variant (baseline, adapter, lora).
    Returns a dict of results.
    """
    # --- Model setup ---
    model = load_vit_tiny(device)
    classification_head = ClassificationHead(int(model.embed_dim), 10).to(device)
    if variant == "adapter":
        model = integrate_adaptors(model, bottleneck_dim, device)
        params = list(classification_head.parameters())
        # Only train adaptors and head
        for block in model.blocks:
            attn = block.attn
            if hasattr(attn, "adaptor_q"):
                params += list(attn.adaptor_q.parameters())
            if hasattr(attn, "adaptor_v"):
                params += list(attn.adaptor_v.parameters())
    elif variant == "lora":
        if not peft_available:
            raise ImportError("peft is not installed. Please install with 'pip install peft'.")
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=32,
            target_modules=["qkv"],
            lora_dropout=0.1,
            bias="none"
        )
        model = integrate_lora(model, lora_config, get_peft_model, device)
        params = list(model.parameters()) + list(classification_head.parameters())
    else:
        params = list(model.parameters()) + list(classification_head.parameters())

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(params, lr=1e-4)

    # --- Training ---
    logging.info(f"Starting {variant} training...")
    train_metrics = train_model(
        model, classification_head, train_loader, val_loader, criterion, optimizer, device, epochs=epochs
    )
    logging.info(f"Training time (s): {train_metrics['training_time']:.2f}")

    # --- Evaluation ---
    logging.info(f"Evaluating {variant} model...")
    eval_metrics = evaluate_model(
        model, classification_head, test_loader, criterion, device
    )
    logging.info(f"Test accuracy: {eval_metrics['accuracy']:.2f}% | Test loss: {eval_metrics['avg_loss']:.4f} | Avg inference time per batch (ms): {eval_metrics['avg_inference_time_ms']:.2f}")

    # --- Parameter Count ---
    total_params = count_trainable_parameters(model) + count_trainable_parameters(classification_head)
    logging.info(f"Total trainable parameters ({variant}): {total_params}")

    return {
        "accuracy": eval_metrics['accuracy'],
        "loss": eval_metrics['avg_loss'],
        "training_time": train_metrics['training_time'],
        "inference_time_ms": eval_metrics['avg_inference_time_ms'],
        "trainable_params": total_params
    }

def main():
    """
    Run all experiment variants (baseline, adapter, lora) and aggregate results.
    """
    logger.setup_logging(log_file="logs/experiment.log")
    with logger.ExceptionHandler("Experiment Setup"):
        config.set_global_seed(42)
        device = config.get_device()
        data_path = config.ensure_data_dir()
        train_loader, val_loader, test_loader = data.load_cifar10(data_path)
        logging.info(f"Device in use: {device}")

        results = {}
        for variant in ["baseline", "adapter", "lora"]:
            try:
                res = run_experiment(
                    variant, device, train_loader, val_loader, test_loader,
                    bottleneck_dim=32, lora_rank=8, epochs=5
                )
                results[variant] = res
            except Exception as e:
                logging.error(f"{variant} experiment failed: {e}")
                results[variant] = {"error": str(e)}

        os.makedirs("results", exist_ok=True)
        with open("results/all_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print("\n--- All Experiment Results ---")
        print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
