# -*- coding: utf-8 -*-
"""
Full Training Script for Dynamic Transformer
سكربت التدريب الكامل على البيانات العربية
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')

import torch
from core.dynamic_transformer.transformer_torch import DynamicTransformerLM
from core.dynamic_transformer.tokenizer import ArabicTokenizer
from core.dynamic_transformer.trainer import Trainer


def main():
    """Full training on linguistic corpus."""
    
    print("=" * 70)
    print("FULL TRAINING - Al-Qalam Dynamic Transformer V4.1")
    print("=" * 70)
    print()
    
    # Configuration
    config = {
        # Data
        "data_file": "training/linguistic/linguistic_corpus.txt",
        "save_dir": "checkpoints/v4_full",
        
        # Model
        "vocab_size": 32000,
        "dim": 256,
        "num_heads": 8,
        "num_layers": 6,
        "max_seq_len": 256,
        
        # Training
        "epochs": 3,
        "batch_size": 32,
        "seq_len": 128,
        "learning_rate": 3e-4,
        
        # Logging
        "log_every": 100,
        "save_every": 1000,
    }
    
    print("Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print()
    
    # Build tokenizer
    print("[1/4] Building tokenizer...")
    tokenizer = ArabicTokenizer(vocab_size=config["vocab_size"])
    
    def line_iterator(path, max_lines=500000):
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                yield line.strip()
    
    tokenizer.build_vocab(line_iterator(config["data_file"]), min_freq=3)
    
    # Save tokenizer
    os.makedirs(config["save_dir"], exist_ok=True)
    tokenizer.save(os.path.join(config["save_dir"], "tokenizer.json"))
    
    # Create model
    print("\n[2/4] Creating model...")
    model = DynamicTransformerLM(
        vocab_size=len(tokenizer.word2idx),
        dim=config["dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        max_seq_len=config["max_seq_len"]
    )
    print(f"Model: {model}")
    print(f"Parameters: {model.count_parameters():,}")
    
    # Create trainer
    print("\n[3/4] Starting training...")
    trainer = Trainer(model, tokenizer, save_dir=config["save_dir"])
    
    # Train
    trainer.train(
        train_file=config["data_file"],
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        seq_len=config["seq_len"],
        learning_rate=config["learning_rate"],
        log_every=config["log_every"],
        save_every=config["save_every"]
    )
    
    # Test generation
    print("\n[4/4] Testing generation...")
    model.eval()
    
    test_prompts = [
        "القط",
        "الشمس",
        "العلم",
        "الإنسان",
        "المعرفة",
    ]
    
    print("\nGeneration samples:")
    for prompt in test_prompts:
        ids = tokenizer.encode(prompt)
        if not ids:
            continue
        prompt_tensor = torch.tensor([ids]).to(trainer.device)
        
        with torch.no_grad():
            generated = model.generate(prompt_tensor, max_new_tokens=20, temperature=0.7)
        
        output = tokenizer.decode(generated[0].tolist())
        print(f"  {prompt} → {output}")
    
    print()
    print("=" * 70)
    print("TRAINING COMPLETE!")
    print(f"Checkpoints saved to: {config['save_dir']}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
