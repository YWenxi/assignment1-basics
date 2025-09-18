"""  script that runs a training loop to train your model on user-provided input. 

- Ability to configure and control the various model and optimizer hyperparameters.
- Memory-efficient loading of training and validation large datasets with np.memmap.
- Serializing checkpoints to a user-provided path.
- Periodically logging training and validation performance (e.g., to console and/or an external
  service like Weights and Biases)

"""

import argparse
import torch
import numpy as np
import random
import wandb
import os
from typing import Callable 

from gpt import (
    CausalLM,
    LazyDataset,
    AdamW,
    cosine_annealing_scheduler,
    cross_entropy,
    save_checkpoint,
    load_checkpoint,
    Tokenizer,
)


def seed_everything(seed: int):
    """
    Set the random seed for all relevant libraries.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    
def train_step(
    model: torch.nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    train_dataset: LazyDataset,
    args: argparse.Namespace,
    loss_fn: Callable = cross_entropy,
):
    """
    Train the model.
    """
    samples, targets = batch
    
    # forward pass
    logits = model(samples)
    loss = loss_fn(logits, targets)
    
    # backward pass
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    
    return loss.item()


def evaluate(
    model: torch.nn.Module,
    val_dataset: LazyDataset,
    args: argparse.Namespace,
    loss_fn: Callable = cross_entropy,
):
    """
    Evaluate the model on validation data.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for _ in range(args.eval_interval // 10):  # Evaluate on a subset of validation data
            batch = val_dataset.get_batch(
                batch_size=args.batch_size,
                block_size=args.context_length,
                device=model.device if hasattr(model, 'device') else next(model.parameters()).device
            )
            samples, targets = batch
            
            logits = model(samples)
            loss = loss_fn(logits, targets)
            total_loss += loss.item()
            num_batches += 1
    
    model.train()
    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser()
    
    # tokenizer hyperparameters
    parser.add_argument("--vocab_path", type=str, default="./vocab.json")
    parser.add_argument("--merges_path", type=str, default="./merges.txt")
    
    # dataset hyperparameters
    parser.add_argument("--train_dataset_file", type=str, required=True)
    parser.add_argument("--val_dataset_file", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--context_length", type=int, default=1024)
    
    
    # model hyperparameters
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--d_ff", type=int, default=3072,
                        help="d_ff = 4 * d_model(768) for gpt2")
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    
    # optimizer hyperparameters
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--betas", type=tuple, default=(0.9, 0.95))
    parser.add_argument("--eps", type=float, default=1e-8)
    
    # training hyperparameters
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--warmup_iters", type=int, default=100)
    parser.add_argument("--cosine_decay_iters", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--max_iters", type=int, default=10000)
    
    # logging hyperparameters for wandb
    parser.add_argument("--wandb_project", type=str, default="cs336-lm-training")
    parser.add_argument("--wandb_name", type=str, default="gpt2")
    
    # checkpoint hyperparameters
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--max_checkpoint_to_keep", type=int, default=3)
    parser.add_argument("--resume_from", type=str, default=None)
    
    # random seed
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # set random seed
    seed_everything(args.seed)
    
    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # load tokenizer
    tokenizer = Tokenizer.from_files(
        vocab_filepath=args.vocab_path,
        merges_filepath=args.merges_path,
        special_tokens=["<|endoftext|>"],
    )
    
    # load datasets
    train_dataset = LazyDataset(args.train_dataset_file, device=device, tokenizer=tokenizer)
    val_dataset = LazyDataset(args.val_dataset_file, device=device, tokenizer=tokenizer)
    
    # load model
    model = CausalLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
    )
    
    # load optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=args.betas,
        eps=args.eps,
    )
    
    # load scheduler
    scheduler = cosine_annealing_scheduler(
        optimizer,
        warmup_iters=args.warmup_iters,
        cosine_decay_iters=args.cosine_decay_iters,
    )
    
    # resume from checkpoint if specified
    start_iter = 0
    if args.resume_from is not None:
        print(f"Resuming from checkpoint: {args.resume_from}")
        start_iter = load_checkpoint(args.resume_from, model, optimizer)
        print(f"Resumed from iteration {start_iter}")
    
    # create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # initialize wandb
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        config=vars(args),
    )
    
    # training loop
    print(f"Starting training on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for iter_num in range(start_iter, args.max_iters):
        # get batch
        batch = train_dataset.get_batch(
            batch_size=args.batch_size,
            block_size=args.context_length,
            device=device
        )
        
        # train step
        loss = train_step(
            model=model,
            batch=batch,
            optimizer=optimizer,
            scheduler=scheduler,
            train_dataset=train_dataset,
            args=args,
        )
        
        # logging
        if iter_num % args.log_interval == 0:
            print(f"iter {iter_num}: loss {loss:.4f}, lr {scheduler.get_last_lr()[0]:.2e}")
            wandb.log({
                "train/loss": loss,
                "train/learning_rate": scheduler.get_last_lr()[0],
                "train/iteration": iter_num,
            })
        
        # evaluation
        if iter_num % args.eval_interval == 0:
            val_loss = evaluate(model, val_dataset, args)
            print(f"iter {iter_num}: val_loss {val_loss:.4f}")
            wandb.log({
                "val/loss": val_loss,
                "val/iteration": iter_num,
            })
        
        # save checkpoint
        if iter_num % args.save_interval == 0:
            # delete old checkpoints if we have more than max_checkpoint_to_keep
            if args.max_checkpoint_to_keep > 0:
                checkpoint_files = [f for f in os.listdir(args.checkpoint_dir) if f.endswith(".pt")]
                checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(args.checkpoint_dir, x)))
                for file in checkpoint_files[:-args.max_checkpoint_to_keep]:
                    os.remove(os.path.join(args.checkpoint_dir, file))
            
            checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_{iter_num}.pt")
            save_checkpoint(model, optimizer, iter_num, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # save final checkpoint
    final_checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint_final.pt")
    save_checkpoint(model, optimizer, args.max_iters, final_checkpoint_path)
    print(f"Training completed. Final checkpoint saved to {final_checkpoint_path}")
    
    wandb.finish()


if __name__ == "__main__":
    main()
    