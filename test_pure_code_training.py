# -*- coding: utf-8 -*-
"""
Quick Test - Pure Code Training
اختبار التدريب بكود فقط
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

import torch
from core.dynamic_transformer.pure_dynamic_transformer import PureDynamicTransformer
from core.dynamic_transformer.tokenizer import ArabicTokenizer, create_dataloader
from core.dynamic_transformer.pure_code_trainer import PureCodeTrainer


def test_pure_code_training():
    """Test training with code-only export."""
    
    print("=" * 60)
    print("PURE CODE TRAINING TEST")
    print("NO .pt FILES - ONLY Python CODE!")
    print("=" * 60)
    print()
    
    data_file = "training/data/arabic_sample.txt"
    save_dir = "trained_model_pure"
    
    # Tokenizer
    print("[1/4] Tokenizer...")
    tokenizer = ArabicTokenizer(vocab_size=2000)
    
    def lines(path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                yield line.strip()
    
    tokenizer.build_vocab(lines(data_file), min_freq=1)
    
    # Model
    print("\n[2/4] Model...")
    model = PureDynamicTransformer(
        vocab_size=len(tokenizer.word2idx),
        dim=128,
        num_heads=4,
        num_layers=2,
        max_seq_len=64
    )
    print(f"Parameters: {model.count_parameters():,}")
    
    # Train
    print("\n[3/4] Training...")
    trainer = PureCodeTrainer(model, tokenizer, save_dir=save_dir)
    trainer.train(
        train_file=data_file,
        epochs=2,
        batch_size=4,
        seq_len=32,
        log_every=5,
        save_every=20
    )
    
    # Verify no .pt files
    print("\n[4/4] Verifying no .pt files...")
    has_pt = False
    for root, dirs, files in os.walk(save_dir):
        for f in files:
            if f.endswith('.pt'):
                print(f"  ❌ Found .pt file: {f}")
                has_pt = True
            else:
                path = os.path.join(root, f)
                size = os.path.getsize(path)
                print(f"  ✅ {f}: {size:,} bytes")
    
    if not has_pt:
        print("\n✅ SUCCESS! NO .pt FILES - PURE PYTHON CODE ONLY!")
    
    # Test generation
    print("\n=== Generation Test ===")
    model.eval()
    for prompt in ["القط", "الشمس"]:
        ids = tokenizer.encode(prompt)
        if ids:
            with torch.no_grad():
                gen = model.generate(torch.tensor([ids]).to(trainer.device), max_new_tokens=8)
            output = tokenizer.decode(gen[0].tolist())
            print(f"  {prompt} → {output}")
    
    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    test_pure_code_training()
