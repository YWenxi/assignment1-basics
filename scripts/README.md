# Training Scripts

This directory contains scripts for training GPT models on various datasets.

## Files

- `training_together.py`: Main training script with comprehensive functionality
- `example_training.py`: Example script showing how to use the training script
- `README.md`: This file

## Usage

### Basic Training

To train a model, you need to have prepared dataset files in `.npy` format. The training script supports various hyperparameters:

```bash
python scripts/training_together.py \
    --train_dataset_file data/train.npy \
    --val_dataset_file data/val.npy \
    --batch_size 32 \
    --context_length 256 \
    --d_model 384 \
    --num_layers 6 \
    --num_heads 6 \
    --learning_rate 1e-4 \
    --max_iters 1000
```

### Key Parameters

#### Dataset Parameters
- `--train_dataset_file`: Path to training dataset (.npy file)
- `--val_dataset_file`: Path to validation dataset (.npy file)
- `--batch_size`: Batch size for training (default: 128)
- `--context_length`: Context length for language modeling (default: 1024)
- `--vocab_size`: Vocabulary size (default: 50257)

#### Model Parameters
- `--d_model`: Model dimension (default: 768)
- `--num_layers`: Number of transformer layers (default: 12)
- `--num_heads`: Number of attention heads (default: 12)
- `--d_ff`: Feed-forward dimension (default: 3072)
- `--rope_theta`: RoPE theta parameter (default: 10000.0)

#### Training Parameters
- `--learning_rate`: Learning rate (default: 1e-4)
- `--weight_decay`: Weight decay (default: 0.01)
- `--max_iters`: Maximum training iterations (default: 10000)
- `--eval_interval`: Evaluation interval (default: 100)
- `--save_interval`: Checkpoint saving interval (default: 100)
- `--log_interval`: Logging interval (default: 10)
- `--warmup_iters`: Warmup iterations (default: 100)
- `--cosine_decay_iters`: Cosine decay iterations (default: 1000)

#### Logging and Checkpoints
- `--wandb_project`: Weights & Biases project name (default: "cs336-lm-training")
- `--wandb_name`: Weights & Biases run name (default: "gpt2")
- `--checkpoint_dir`: Directory to save checkpoints (default: "./checkpoints")
- `--resume_from`: Path to checkpoint to resume from (optional)

### Example Usage

#### TinyStories Training
```bash
python scripts/training_together.py \
    --train_dataset_file data/tinystories_train.npy \
    --val_dataset_file data/tinystories_val.npy \
    --batch_size 32 \
    --context_length 256 \
    --d_model 384 \
    --num_layers 6 \
    --num_heads 6 \
    --d_ff 1536 \
    --learning_rate 1e-4 \
    --max_iters 1000 \
    --eval_interval 50 \
    --save_interval 100 \
    --checkpoint_dir ./checkpoints/tinystories \
    --wandb_project cs336-tinystories \
    --wandb_name gpt-tinystories-small
```

#### Resume Training
```bash
python scripts/training_together.py \
    --train_dataset_file data/train.npy \
    --val_dataset_file data/val.npy \
    --resume_from ./checkpoints/checkpoint_500.pt \
    # ... other parameters
```

### Features

The training script includes:

1. **Memory-efficient data loading**: Uses `np.memmap` for large datasets
2. **Comprehensive logging**: Console output and Weights & Biases integration
3. **Checkpointing**: Automatic saving and resuming from checkpoints
4. **Learning rate scheduling**: Cosine annealing with warmup
5. **Evaluation**: Periodic validation on a subset of validation data
6. **Hyperparameter control**: All major model and training parameters configurable

### Requirements

- PyTorch
- NumPy
- Weights & Biases (wandb)
- The GPT implementation from the `gpt` module

### Output

The script will:
- Print training progress to console
- Log metrics to Weights & Biases (if configured)
- Save checkpoints to the specified directory
- Save a final checkpoint when training completes

### Tips

1. Start with smaller models and datasets to test your setup
2. Use appropriate batch sizes for your GPU memory
3. Monitor validation loss to avoid overfitting
4. Use Weights & Biases to track experiments
5. Save checkpoints frequently to avoid losing progress 