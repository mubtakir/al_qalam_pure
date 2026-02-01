# -*- coding: utf-8 -*-
"""
Dynamic Transformer Trainer
مُدرّب الـ Transformer الديناميكي الحقيقي
"""

import os
import sys
import time
import math
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.dynamic_transformer.transformer_torch import DynamicTransformerLM
from core.dynamic_transformer.tokenizer import ArabicTokenizer, create_dataloader


class Trainer:
    """
    مُدرّب الـ Transformer الديناميكي
    
    Features:
    - Gradient accumulation
    - Mixed precision (optional)
    - Checkpointing
    - Logging
    """
    
    def __init__(self,
                 model: DynamicTransformerLM,
                 tokenizer: ArabicTokenizer,
                 save_dir: str = "checkpoints",
                 device: str = None):
        
        self.model = model
        self.tokenizer = tokenizer
        self.save_dir = save_dir
        
        # Device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model.to(self.device)
        print(f"[TRAINER] Device: {self.device}")
        print(f"[TRAINER] Parameters: {model.count_parameters():,}")
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
    
    def train(self,
              train_file: str,
              epochs: int = 3,
              batch_size: int = 32,
              seq_len: int = 128,
              learning_rate: float = 3e-4,
              warmup_steps: int = 1000,
              grad_accum_steps: int = 1,
              log_every: int = 100,
              save_every: int = 1000,
              max_samples: int = None):
        """
        Train the model.
        
        Args:
            train_file: Path to training text file
            epochs: Number of epochs
            batch_size: Batch size
            seq_len: Sequence length
            learning_rate: Learning rate
            warmup_steps: Warmup steps
            grad_accum_steps: Gradient accumulation steps
            log_every: Log every N steps
            save_every: Save checkpoint every N steps
            max_samples: Max samples (for testing)
        """
        print(f"\n{'='*60}")
        print(f"TRAINING DYNAMIC TRANSFORMER")
        print(f"{'='*60}")
        print(f"File: {train_file}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Sequence length: {seq_len}")
        print(f"Learning rate: {learning_rate}")
        print()
        
        # Create dataloader
        dataloader = create_dataloader(
            train_file,
            self.tokenizer,
            batch_size=batch_size,
            seq_len=seq_len,
            max_samples=max_samples
        )
        
        # Optimizer
        optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.1
        )
        
        # Scheduler
        total_steps = len(dataloader) * epochs // grad_accum_steps
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
        
        # Training loop
        self.model.train()
        global_step = 0
        total_loss = 0
        start_time = time.time()
        
        for epoch in range(epochs):
            print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
            epoch_loss = 0
            epoch_steps = 0
            
            for batch_idx, (input_ids, labels) in enumerate(dataloader):
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                
                # Forward
                _, loss = self.model(input_ids, labels)
                loss = loss / grad_accum_steps
                
                # Backward
                loss.backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % grad_accum_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    global_step += 1
                    self.model.train_steps = global_step
                
                total_loss += loss.item() * grad_accum_steps
                epoch_loss += loss.item() * grad_accum_steps
                epoch_steps += 1
                
                # Logging
                if (batch_idx + 1) % log_every == 0:
                    avg_loss = total_loss / (global_step + 1)
                    elapsed = time.time() - start_time
                    samples_per_sec = (global_step + 1) * batch_size / elapsed
                    
                    print(f"  Step {global_step:5d} | "
                          f"Loss: {avg_loss:.4f} | "
                          f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                          f"Speed: {samples_per_sec:.1f} samples/s")
                
                # Save checkpoint
                if global_step > 0 and global_step % save_every == 0:
                    self.save_checkpoint(global_step)
            
            # Epoch summary
            epoch_avg_loss = epoch_loss / epoch_steps
            print(f"Epoch {epoch + 1} - Average Loss: {epoch_avg_loss:.4f}")
            
            # Save epoch checkpoint
            self.save_checkpoint(global_step, f"epoch_{epoch + 1}")
        
        # Final save
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE!")
        print(f"Total steps: {global_step}")
        print(f"Final loss: {total_loss / global_step:.4f}")
        print(f"Time: {(time.time() - start_time) / 60:.1f} minutes")
        print(f"{'='*60}")
        
        self.save_checkpoint(global_step, "final")
        self.model.persist_as_code(os.path.join(self.save_dir, "code_export"))
    
    def save_checkpoint(self, step: int, suffix: str = None):
        """Save a checkpoint."""
        name = f"checkpoint_{step}" if suffix is None else f"checkpoint_{suffix}"
        path = os.path.join(self.save_dir, f"{name}.pt")
        
        torch.save({
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "vocab_size": self.model.vocab_size,
            "dim": self.model.dim,
            "num_heads": self.model.num_heads,
            "num_layers": self.model.num_layers,
            "tokenizer_vocab": self.tokenizer.word2idx,
        }, path)
        
        print(f"  [SAVE] {path}")
    
    def load_checkpoint(self, path: str):
        """Load a checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.tokenizer.word2idx = checkpoint["tokenizer_vocab"]
        self.tokenizer.idx2word = {v: k for k, v in self.tokenizer.word2idx.items()}
        print(f"[LOAD] Loaded checkpoint from {path}")
        return checkpoint["step"]


def main():
    """Main training script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Dynamic Transformer")
    parser.add_argument("--data", type=str, required=True, help="Training data file")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length")
    parser.add_argument("--dim", type=int, default=256, help="Model dimension")
    parser.add_argument("--heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--layers", type=int, default=6, help="Number of layers")
    parser.add_argument("--vocab-size", type=int, default=32000, help="Vocabulary size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples (for testing)")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Save directory")
    
    args = parser.parse_args()
    
    # Build tokenizer
    print("[MAIN] Building tokenizer...")
    tokenizer = ArabicTokenizer(vocab_size=args.vocab_size)
    
    def line_iterator(path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                yield line.strip()
    
    tokenizer.build_vocab(line_iterator(args.data), min_freq=2)
    
    # Create model
    print("[MAIN] Creating model...")
    model = DynamicTransformerLM(
        vocab_size=len(tokenizer.word2idx),
        dim=args.dim,
        num_heads=args.heads,
        num_layers=args.layers,
        max_seq_len=args.seq_len
    )
    print(model)
    
    # Create trainer
    trainer = Trainer(model, tokenizer, save_dir=args.save_dir)
    
    # Train
    trainer.train(
        train_file=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        learning_rate=args.lr,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()
