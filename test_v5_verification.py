# test_v5_verification.py
import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.getcwd())

from core.dynamic_transformer.pure_dynamic_transformer import PureDynamicTransformer
from core.dynamic_transformer.tokenizer import ArabicTokenizer
from core.dynamic_transformer.pure_code_trainer import PureCodeTrainer

def test_v5_features():
    print("=== Testing Al-Qalam Pure V5.0 Features ===")
    
    # 1. Setup Model
    vocab_size = 100
    dim = 64
    model = PureDynamicTransformer(vocab_size=vocab_size, dim=dim, num_layers=2)
    word2idx = {f"word_{i}": i for i in range(vocab_size)}
    
    print(f"\n[1] Testing Adaptive Embedding...")
    old_params = model.count_parameters()
    model.update_vocab(150)
    new_params = model.count_parameters()
    print(f"    - Vocab expanded: 100 -> 150")
    print(f"    - Parameters: {old_params:,} -> {new_params:,}")
    assert model.vocab_size == 150
    assert model.embedding.num_embeddings == 150
    
    print(f"\n[2] Testing Semantic Weight Mapping...")
    # Add some dummy training to create variance
    with torch.no_grad():
        model.embedding.weight[10] *= 5.0 # Mark word_10 as important
        
    word2idx_expanded = {f"word_{i}": i for i in range(150)}
    knowledge = model._analyze_semantic_relations(word2idx_expanded)
    print(f"    - Semantic Hubs detected: {len(knowledge['semantic_hubs'])}")
    print(f"    - Top Hub: {knowledge['semantic_hubs'][-1]}")
    assert len(knowledge['semantic_hubs']) == 50
    
    print(f"\n[3] Testing V5.0 Export (Knowledge Map)...")
    export_dir = "test_v5_export"
    model.to_python_code(export_dir, word2idx_expanded)
    
    kmap_path = os.path.join(export_dir, "knowledge_map.py")
    assert os.path.exists(kmap_path)
    print(f"    - knowledge_map.py created successfully.")
    
    with open(kmap_path, 'r', encoding='utf-8') as f:
        content = f.read()
        assert "SEMANTIC_METADATA" in content
        print(f"    - Semantic metadata verified in export.")

    print(f"\n[4] Testing Symbolic Auditor Integration...")
    # Create dummy data for trainer
    with open("dummy.txt", "w", encoding="utf-8") as f:
        f.write("test sentence for symbolic audit\n") * 10
        
    tokenizer = ArabicTokenizer(vocab_size=150)
    tokenizer.word2idx = word2idx_expanded
    tokenizer.idx2word = {v: k for k, v in word2idx_expanded.items()}
    
    trainer = PureCodeTrainer(model, tokenizer, save_dir="test_v5_trainer")
    # Manually trigger auditor
    trainer._run_symbolic_audit()
    print(f"    - Symbolic auditor triggered successfully.")

    print("\n✅ ALL V5.0 FEATURES VERIFIED!")

if __name__ == "__main__":
    test_v5_features()
