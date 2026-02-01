# -*- coding: utf-8 -*-
"""
Dynamic Transformer Demo
اختبار الـ Transformer الديناميكي
"""

import sys
sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding='utf-8')

from core.dynamic_transformer.weight_cell import WeightCell
from core.dynamic_transformer.dynamic_embedding import DynamicEmbedding
from core.dynamic_transformer.dynamic_attention import DynamicAttention
from core.dynamic_transformer.dynamic_ffn import DynamicFFN, DynamicLayerNorm
from core.dynamic_transformer.dynamic_transformer import DynamicTransformer


def main():
    print("=" * 50)
    print("Dynamic Transformer V4.0 - Al-Qalam")
    print("=" * 50)
    print()
    
    # Create model
    model = DynamicTransformer(
        dim=64,
        num_heads=4,
        num_layers=2,
        base_dir="."
    )
    print(model)
    print()
    
    # Learn from sentences
    sentences = [
        ["القط", "يأكل", "السمك"],
        ["الكلب", "يشرب", "الماء"],
        ["الطفل", "يقرأ", "الكتاب"],
        ["الفتاة", "تكتب", "الرسالة"],
        ["المعلم", "يشرح", "الدرس"],
    ]
    
    print("=== Learning ===")
    for s in sentences:
        model.learn_from_sentence(s)
        print(f"  Learned: {' '.join(s)}")
    
    print()
    print(f"Vocabulary: {len(model.embedding)} words")
    print(f"Train steps: {model.train_steps}")
    print()
    
    # Generate
    print("=== Generating ===")
    for prompt in [["القط"], ["الطفل"], ["المعلم"]]:
        gen = model.generate(prompt, max_tokens=3)
        print(f"  {prompt[0]} -> {' '.join(gen)}")
    
    # Similarity
    print()
    print("=== Word Similarity ===")
    print(f"  (القط, الكلب): {model.embedding.similarity('القط', 'الكلب'):.3f}")
    print(f"  (يأكل, يشرب): {model.embedding.similarity('يأكل', 'يشرب'):.3f}")
    
    # Save
    print()
    model.persist()
    
    print()
    print("=" * 50)
    print("SUCCESS! Dynamic Transformer is working!")
    print("=" * 50)


if __name__ == "__main__":
    main()
