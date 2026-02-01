# 📖 دليل التدريب - Al-Qalam V4.1

## نظرة عامة

هذا الدليل يشرح كيفية تدريب **Pure Dynamic Transformer** - نموذج توليدي يحفظ أوزانه ككود Python فقط.

---

## البيانات المتاحة

```
training/
├── linguistic/
│   ├── linguistic_corpus.txt    # 910 MB (عربي)
│   └── english_corpus.txt       # 232 MB
├── data/
│   └── arabic_sample.txt        # 3 KB (للاختبار)
└── code/
```

---

## المتطلبات

```
- Python 3.8+
- PyTorch 2.0+
- GPU (موصى به): RTX 3060+ / 8GB VRAM
- RAM: 16GB+
```

---

## الاستخدام السريع

### 1. اختبار سريع (دقيقة واحدة)

```bash
python test_pure_code_training.py
```

### 2. تدريب كامل

```python
from core.dynamic_transformer.pure_dynamic_transformer import PureDynamicTransformer
from core.dynamic_transformer.pure_code_trainer import PureCodeTrainer
from core.dynamic_transformer.tokenizer import ArabicTokenizer

# 1. Tokenizer
tokenizer = ArabicTokenizer(vocab_size=32000)

def lines(path, max_lines=500000):
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_lines: break
            yield line.strip()

tokenizer.build_vocab(
    lines("training/linguistic/linguistic_corpus.txt"),
    min_freq=3
)

# 2. Model
model = PureDynamicTransformer(
    vocab_size=len(tokenizer.word2idx),
    dim=256,
    num_heads=8,
    num_layers=6
)

# 3. Train
trainer = PureCodeTrainer(model, tokenizer, save_dir="my_model")
trainer.train(
    train_file="training/linguistic/linguistic_corpus.txt",
    epochs=3,
    batch_size=32,
    seq_len=128,
    learning_rate=3e-4
)
```

---

## المعاملات

| المعامل | الوصف | القيمة الافتراضية |
|---------|-------|-------------------|
| `dim` | بعد النموذج | 256 |
| `num_heads` | رؤوس الانتباه | 8 |
| `num_layers` | عدد الطبقات | 6 |
| `vocab_size` | حجم المفردات | 32000 |
| `batch_size` | حجم الدفعة | 32 |
| `seq_len` | طول السلسلة | 128 |
| `learning_rate` | معدل التعلم | 3e-4 |
| `epochs` | عدد الحقب | 3 |

---

## الوقت المتوقع

| البيانات | GPU | الوقت |
|----------|-----|-------|
| 3 KB (sample) | RTX 3060 | < 1 min |
| 100 MB | RTX 3060 | ~1 hour |
| 910 MB | RTX 3060 | ~6-12 hours |
| 910 MB | RTX 4090 | ~2-4 hours |

---

## النتيجة

```
my_model/final/
├── config.py      # الإعدادات
│   → VOCAB_SIZE, DIM, NUM_LAYERS...
│
├── vocab.py       # المفردات
│   → WORD2IDX = {"القط": 4, ...}
│
├── weights.py     # الأوزان ككود!
│   → WEIGHTS = {"embedding.weight": ("eJz...", (32000, 256)), ...}
│
└── loader.py      # للتحميل
    → load_model() → (model, word2idx, idx2word)
```

**لا `.pt` على الإطلاق!** ✅

---

## التحميل والاستخدام

```python
# من المجلد الناتج
from my_model.final import load_model

model, word2idx, idx2word = load_model()

# التوليد
import torch
prompt = [word2idx.get("القط", 1)]
generated = model.generate(torch.tensor([prompt]), max_new_tokens=20)
output = " ".join([idx2word.get(i, "?") for i in generated[0].tolist()])
print(output)
```

---

## الفرق عن التدريب التقليدي

| الجانب | التقليدي | Al-Qalam |
|--------|----------|----------|
| **التدريب** | Backprop | Backprop ✅ |
| **الحفظ** | `torch.save()` → `.pt` | `to_python_code()` → `.py` |
| **الملف** | ثنائي (غير مقروء) | كود Python (مقروء) |
| **الحجم** | ~X MB | ~X MB (مضغوط) |

---

## نصائح

1. **ابدأ صغيراً**: اختبر على `arabic_sample.txt` أولاً
2. **راقب Loss**: يجب أن ينخفض تدريجياً
3. **احفظ بانتظام**: `save_every=1000` يحفظ كل 1000 خطوة
4. **استخدم GPU**: التدريب على CPU بطيء جداً

---

## استكشاف الأخطاء

| المشكلة | الحل |
|---------|------|
| `CUDA out of memory` | قلل `batch_size` أو `seq_len` |
| `Loss is NaN` | قلل `learning_rate` |
| التوليد متكرر | زد `temperature` (مثلاً 1.0) |
