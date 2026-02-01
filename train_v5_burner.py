# train_v5_burner.py
import sys
import os
import torch

sys.path.insert(0, os.getcwd())

from core.dynamic_transformer.pure_dynamic_transformer import PureDynamicTransformer
from core.dynamic_transformer.tokenizer import ArabicTokenizer
from core.dynamic_transformer.pure_code_trainer import PureCodeTrainer

def run_burner_session():
    print("🔥 [BURNER SESSION] Al-Qalam Pure V5.0 Initiated 🔥")
    
    # 1. Config
    DATA_PATH = "training/linguistic/linguistic_corpus.txt" # Using real path found earlier
    SAVE_DIR = "trained_model_v5_burner"
    VOCAB_SIZE = 1000 # Small for quick test
    DIM = 128
    LAYERS = 2
    
    if not os.path.exists(DATA_PATH):
        # Fallback to creating a small sample if corpus not found
        print("⚠️ Corpus not found at expected path, creating dummy training data...")
        with open("burner_sample.txt", "w", encoding="utf-8") as f:
            f.write("العلم نور والجهل ظلام\n" * 100)
            f.write("القلم يكتب المستقبل بالمنطق والرياضيات\n" * 100)
        DATA_PATH = "burner_sample.txt"

    # 2. Tokenizer
    print("[1] Building Tokenizer...")
    tokenizer = ArabicTokenizer(vocab_size=VOCAB_SIZE)
    def lines(path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f: yield line.strip()
    tokenizer.build_vocab(lines(DATA_PATH), min_freq=2)
    
    # 3. Model
    print("[2] Creating V5.0 Transformer...")
    model = PureDynamicTransformer(
        vocab_size=len(tokenizer.word2idx),
        dim=DIM,
        num_heads=4,
        num_layers=LAYERS
    )
    
    # 4. Training
    print("[3] Starting Training with Symbolic Audit...")
    trainer = PureCodeTrainer(model, tokenizer, save_dir=SAVE_DIR)
    trainer.train(
        train_file=DATA_PATH,
        epochs=1,
        batch_size=16,
        seq_len=32,
        log_every=10,
        save_every=50,
        max_samples=200 # Very small session
    )
    
    # 5. Reviewing Knowledge Map
    print("\n[4] Reviewing Generated Knowledge Map...")
    kmap_path = os.path.join(SAVE_DIR, "final", "knowledge_map.py")
    if os.path.exists(kmap_path):
        with open(kmap_path, 'r', encoding='utf-8') as f:
            print("\n--- knowledge_map.py CONTENT ---")
            print(f.read())
            print("--------------------------------")
    
    print("\n🔥 BURNER SESSION COMPLETE! Check results in trained_model_v5_burner/final/ 🔥")

if __name__ == "__main__":
    run_burner_session()
