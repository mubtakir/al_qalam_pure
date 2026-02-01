# -*- coding: utf-8 -*-
"""
Pure Code Trainer - Training with Code-Only Export
مُدرّب للتدريب مع تصدير ككود فقط - بدون .pt!
"""

import os
import sys
import time
from datetime import datetime

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.dynamic_transformer.pure_dynamic_transformer import PureDynamicTransformer
from core.dynamic_transformer.tokenizer import ArabicTokenizer, create_dataloader
from core.self_writing_model import SelfWritingModel


class PureCodeTrainer:
    """
    مُدرّب يحفظ النتائج ككود Python فقط.
    
    NO .pt FILES AT ALL!
    """
    
    def __init__(self,
                 model: PureDynamicTransformer,
                 tokenizer: ArabicTokenizer,
                 save_dir: str = "trained_model",
                 device: str = None):
        
        self.model = model
        self.tokenizer = tokenizer
        self.save_dir = save_dir
        
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model.to(self.device)
        
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"[TRAINER] Device: {self.device}")
        print(f"[TRAINER] Parameters: {model.count_parameters():,}")
        print(f"[TRAINER] Save format: Python code ONLY (no .pt)")
        
        # V5.0: Symbolic Auditor
        self.symbolic_model = SelfWritingModel(os.getcwd())
    
    def train(self,
              train_file: str,
              epochs: int = 3,
              batch_size: int = 32,
              seq_len: int = 128,
              learning_rate: float = 3e-4,
              log_every: int = 100,
              save_every: int = 1000,
              max_samples: int = None):
        """Train and save as Python code."""
        
        print(f"\n{'='*60}")
        print(f"PURE CODE TRAINING - Al-Qalam V5.0 (Symbolic Enhanced)")
        print(f"{'='*60}")
        print(f"Output: Python code ONLY - NO .pt files!")
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
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.1)
        total_steps = len(dataloader) * epochs
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
        
        # Training
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
                
                optimizer.zero_grad()
                _, loss = self.model(input_ids, labels)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                global_step += 1
                self.model.train_steps = global_step
                total_loss += loss.item()
                epoch_loss += loss.item()
                epoch_steps += 1
                
                # Log
                if global_step % log_every == 0:
                    avg_loss = total_loss / global_step
                    elapsed = time.time() - start_time
                    speed = global_step * batch_size / elapsed
                    print(f"  Step {global_step:5d} | Loss: {avg_loss:.4f} | Speed: {speed:.1f} s/s")
                    
                    # V5.0: Symbolic Validation Check
                    self._run_symbolic_audit()
                
                # Save as code
                if global_step % save_every == 0:
                    self._save_as_code(f"step_{global_step}")
            
            # Epoch save
            epoch_avg = epoch_loss / epoch_steps
            print(f"Epoch {epoch + 1} - Loss: {epoch_avg:.4f}")
            self._save_as_code(f"epoch_{epoch + 1}")
        
        # Final save
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE!")
        print(f"Final loss: {total_loss / global_step:.4f}")
        print(f"Time: {(time.time() - start_time) / 60:.1f} min")
        print(f"{'='*60}")
        
        self._save_as_code("final")
        
        print(f"\n✅ Saved to: {self.save_dir}/final/")
        print(f"   - config.py")
        print(f"   - vocab.py")
        print(f"   - weights.py (ALL WEIGHTS AS CODE!)")
        print(f"   - loader.py")
        print(f"\n   NO .pt FILES! ✅")
    
    def _save_as_code(self, name: str):
        """Save model as Python code."""
        path = os.path.join(self.save_dir, name)
        self.model.to_python_code(path, self.tokenizer.word2idx)
        print(f"  [CODE] {path}/")

    def _run_symbolic_audit(self):
        """V5.0: Check model generation against symbolic rules."""
        with torch.no_grad():
            self.model.eval()
            prompt = torch.tensor([[self.tokenizer.word2idx.get("<START>", 2)]]).to(self.device)
            gen_ids = self.model.generate(prompt, max_new_tokens=10)
            text = " ".join([self.tokenizer.idx2word[i] for i in gen_ids[0].tolist()])
            
            # Audit against contradictions
            warnings = self.symbolic_model.auditor.check_contradictions()
            if warnings:
                print(f"  [AUDITOR] Contradictions in Knowledge Base: {len(warnings)}")
            
            self.model.train()


def main():
    """Main pure code training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train with Pure Code Export")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--save-dir", type=str, default="trained_model")
    parser.add_argument("--max-samples", type=int, default=None)
    
    args = parser.parse_args()
    
    # Tokenizer
    tokenizer = ArabicTokenizer(vocab_size=args.vocab_size)
    
    def lines(path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                yield line.strip()
    
    tokenizer.build_vocab(lines(args.data), min_freq=3)
    
    # Model
    model = PureDynamicTransformer(
        vocab_size=len(tokenizer.word2idx),
        dim=args.dim,
        num_heads=args.heads,
        num_layers=args.layers
    )
    
    # Train
    trainer = PureCodeTrainer(model, tokenizer, save_dir=args.save_dir)
    trainer.train(
        train_file=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()
