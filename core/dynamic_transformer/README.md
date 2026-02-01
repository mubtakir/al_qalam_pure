# 🚀 Dynamic Transformer - دليل شامل

## نظرة عامة

**Al-Qalam Dynamic Transformer V4.1** - أول Transformer يخزن أوزانه ككود Python بدلاً من ملفات ثنائية.

```
الفلسفة: الكود = الأوزان
```

---

## الفرق عن النماذج التقليدية

| الجانب | GPT/Llama/Qwen | Al-Qalam |
|--------|----------------|----------|
| **الأوزان** | `.pt/.safetensors` (ثنائي) | `.py` (كود Python) |
| **مقروء؟** | ❌ لا | ✅ نعم |
| **قابل للتعديل؟** | ❌ لا | ✅ نعم |
| **التدريب** | Backprop | Backprop ✅ |
| **الحفظ** | `torch.save()` | `to_python_code()` |

---

## البنية المعمارية

```
┌─────────────────────────────────────────────────┐
│          PureDynamicTransformer                  │
├─────────────────────────────────────────────────┤
│  Embedding     → (vocab_size, dim)               │
│  PosEncoding   → (max_seq_len, dim)              │
│  ┌─────────────────────────────────────────┐    │
│  │ TransformerBlock × num_layers           │    │
│  │   - LayerNorm1                          │    │
│  │   - Multi-Head Attention (QKV)          │    │
│  │   - LayerNorm2                          │    │
│  │   - FFN (Linear → GELU → Linear)        │    │
│  └─────────────────────────────────────────┘    │
│  LayerNorm (output)                              │
│  LM_Head → (dim, vocab_size)                     │
└─────────────────────────────────────────────────┘
```

---

## الملفات

```
core/dynamic_transformer/
├── pure_dynamic_transformer.py  # النموذج الأساسي
├── pure_code_trainer.py         # المُدرّب
├── tokenizer.py                 # Tokenizer عربي
├── transformer_torch.py         # نسخة PyTorch عادية
├── trainer.py                   # مُدرّب عادي
└── README.md                    # هذا الملف
```

---

## الاستخدام السريع

### 1. إنشاء نموذج

```python
from core.dynamic_transformer.pure_dynamic_transformer import PureDynamicTransformer

model = PureDynamicTransformer(
    vocab_size=32000,    # حجم المفردات
    dim=256,             # البعد
    num_heads=8,         # رؤوس الانتباه
    num_layers=6,        # الطبقات
    max_seq_len=256      # أقصى طول
)

print(f"Parameters: {model.count_parameters():,}")
```

### 2. التدريب

```python
from core.dynamic_transformer.pure_code_trainer import PureCodeTrainer
from core.dynamic_transformer.tokenizer import ArabicTokenizer

# Tokenizer
tokenizer = ArabicTokenizer(vocab_size=32000)
tokenizer.build_vocab(line_iterator("data.txt"), min_freq=3)

# Training
trainer = PureCodeTrainer(model, tokenizer, save_dir="output")
trainer.train(
    train_file="data.txt",
    epochs=3,
    batch_size=32,
    seq_len=128,
    learning_rate=3e-4
)
```

### 3. الحفظ ككود

```python
model.to_python_code("output_model", tokenizer.word2idx)
```

**النتيجة:**
```
output_model/
├── config.py      # الإعدادات
├── vocab.py       # المفردات
├── weights.py     # الأوزان ككود!
└── loader.py      # للتحميل
```

### 4. التحميل

```python
from output_model import load_model

model, word2idx, idx2word = load_model()
```

### 5. التوليد

```python
import torch

prompt = "القط"
ids = [word2idx.get(w, 1) for w in tokenizer.tokenize(prompt)]
generated = model.generate(torch.tensor([ids]), max_new_tokens=20)
output = " ".join([idx2word[i] for i in generated[0].tolist()])
print(output)
```

---

## نموذج تدريب كامل

```python
# train_full_pure.py
import sys
sys.path.insert(0, ".")

from core.dynamic_transformer.pure_dynamic_transformer import PureDynamicTransformer
from core.dynamic_transformer.tokenizer import ArabicTokenizer
from core.dynamic_transformer.pure_code_trainer import PureCodeTrainer

# Config
DATA = "training/linguistic/linguistic_corpus.txt"
SAVE_DIR = "trained_model"
VOCAB_SIZE = 32000
DIM = 256
HEADS = 8
LAYERS = 6

# Tokenizer
print("Building tokenizer...")
tokenizer = ArabicTokenizer(vocab_size=VOCAB_SIZE)

def lines(path, max_lines=500000):
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_lines: break
            yield line.strip()

tokenizer.build_vocab(lines(DATA), min_freq=3)
tokenizer.save(f"{SAVE_DIR}/tokenizer.json")

# Model
print("Creating model...")
model = PureDynamicTransformer(
    vocab_size=len(tokenizer.word2idx),
    dim=DIM,
    num_heads=HEADS,
    num_layers=LAYERS
)
print(f"Parameters: {model.count_parameters():,}")

# Train
trainer = PureCodeTrainer(model, tokenizer, save_dir=SAVE_DIR)
trainer.train(
    train_file=DATA,
    epochs=3,
    batch_size=32,
    seq_len=128,
    learning_rate=3e-4,
    log_every=100,
    save_every=1000
)

print("Done! Model saved as Python code.")
```

---

## كيف تُحفظ الأوزان ككود؟

```python
# weights.py (مثال مبسط)
import base64
import zlib
import numpy as np

def decompress_array(encoded, shape):
    compressed = base64.b64decode(encoded)
    data = zlib.decompress(compressed)
    return np.frombuffer(data, dtype=np.float32).reshape(shape)

WEIGHTS = {
    "embedding.weight": ("eJzrDPBz5+WS4mJg...", (32000, 256)),
    "blocks.0.qkv.weight": ("eJzrDPBz5+WS...", (768, 256)),
    # ... جميع الأوزان مضغوطة
}

def load_weights():
    return {name: decompress_array(data, shape) 
            for name, (data, shape) in WEIGHTS.items()}
```

---

## المواصفات الموصى بها

| الحجم | dim | heads | layers | params | GPU Memory |
|-------|-----|-------|--------|--------|------------|
| صغير | 128 | 4 | 2 | ~400K | < 1 GB |
| متوسط | 256 | 8 | 6 | ~20M | 2-4 GB |
| كبير | 512 | 8 | 12 | ~80M | 8-16 GB |

---

## الأمر السريع

```bash
# اختبار سريع
python test_pure_code_training.py

# تدريب كامل على البيانات العربية
python -c "
from core.dynamic_transformer.pure_code_trainer import main
main()
" --data training/linguistic/linguistic_corpus.txt --epochs 3
```
