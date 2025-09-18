#!/usr/bin/env python3
"""
Example script showing how to use the training script.

This script demonstrates how to train a GPT model on the TinyStories dataset.
"""

import subprocess
import sys
import os

def main():
    # Example training command for TinyStories dataset
    cmd = [
        sys.executable, "scripts/training_together.py",
        "--train_dataset_file", "data/tinystories_train.npy",
        "--val_dataset_file", "data/tinystories_val.npy",
        "--batch_size", "32",
        "--context_length", "256",
        "--d_model", "384",
        "--num_layers", "6",
        "--num_heads", "6",
        "--d_ff", "1536",
        "--learning_rate", "1e-4",
        "--max_iters", "1000",
        "--eval_interval", "50",
        "--save_interval", "100",
        "--log_interval", "10",
        "--warmup_iters", "50",
        "--cosine_decay_iters", "500",
        "--checkpoint_dir", "./checkpoints/tinystories",
        "--wandb_project", "cs336-tinystories",
        "--wandb_name", "gpt-tinystories-small",
        "--seed", "42"
    ]
    
    print("Running training command:")
    print(" ".join(cmd))
    print()
    
    # Check if data files exist
    train_file = "data/tinystories_train.npy"
    val_file = "data/tinystories_val.npy"
    
    if not os.path.exists(train_file):
        print(f"Error: Training file {train_file} not found!")
        print("Please make sure you have prepared the dataset files.")
        print("You may need to run the data preparation scripts first.")
        return 1
    
    if not os.path.exists(val_file):
        print(f"Error: Validation file {val_file} not found!")
        print("Please make sure you have prepared the dataset files.")
        print("You may need to run the data preparation scripts first.")
        return 1
    
    # Run the training
    try:
        subprocess.run(cmd, check=True)
        print("Training completed successfully!")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error code {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return 1

if __name__ == "__main__":
    exit(main()) 