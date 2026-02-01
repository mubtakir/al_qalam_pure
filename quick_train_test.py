# -*- coding: utf-8 -*-
"""
Quick Training Test
اختبار تدريب سريع للتحقق من عمل النظام
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')

import torch
from core.dynamic_transformer.transformer_torch import DynamicTransformerLM
from core.dynamic_transformer.tokenizer import ArabicTokenizer, create_dataloader
from core.dynamic_transformer.trainer import Trainer


def quick_train_test():
    """Quick training test on sample data."""
    
    print("=" * 60)
    print("QUICK TRAINING TEST - Dynamic Transformer")
    print("=" * 60)
    print()
    
    # Paths
    data_file = "training/data/arabic_sample.txt"
    save_dir = "checkpoints/quick_test"
    
    # Build tokenizer on sample
    print("[1/4] Building tokenizer...")
    tokenizer = ArabicTokenizer(vocab_size=5000)
    
    def line_iterator(path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                yield line.strip()
    
    tokenizer.build_vocab(line_iterator(data_file), min_freq=1)
    
    # Create small model for testing
    print("\n[2/4] Creating model...")
    model = DynamicTransformerLM(
        vocab_size=len(tokenizer.word2idx),
        dim=128,       # Small for testing
        num_heads=4,
        num_layers=2,
        max_seq_len=64
    )
    print(f"Model: {model}")
    print(f"Parameters: {model.count_parameters():,}")
    
    # Create trainer
    print("\n[3/4] Training...")
    trainer = Trainer(model, tokenizer, save_dir=save_dir)
    
    # Quick training
    trainer.train(
        train_file=data_file,
        epochs=2,
        batch_size=4,
        seq_len=32,
        learning_rate=1e-3,
        log_every=10,
        save_every=50
    )
    
    # Test generation
    print("\n[4/4] Testing generation...")
    model.eval()
    
    prompts = ["القط", "الشمس", "العلم"]
    for prompt in prompts:
        ids = tokenizer.encode(prompt)
        if not ids:
            continue
        prompt_tensor = torch.tensor([ids]).to(trainer.device)
        
        with torch.no_grad():
            generated = model.generate(prompt_tensor, max_new_tokens=10, temperature=0.8)
        
        output = tokenizer.decode(generated[0].tolist())
        print(f"  {prompt} -> {output}")
    
    print()
    print("=" * 60)
    print("QUICK TEST COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    quick_train_test()
