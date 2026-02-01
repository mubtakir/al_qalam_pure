# 📋 Changelog

## V5.0 - The Living Knowledge Transformer (2026-02-01)

### 🧠 ذكاء معماري شفاف
- **Adaptive Embedding**: التوسع الديناميكي للمفردات دون إعادة تدريب.
- **Semantic Weight Mapping**: تصدير العلاقات الدلالية المكتشفة ككود (`knowledge_map.py`).
- **Symbolic Training Audit**: الرقابة المنطقية التلقائية أثناء التدريب عبر محرك "البيان".

### 📁 ملفات مطورة
- `core/dynamic_transformer/pure_dynamic_transformer.py` (تم تحديثه)
- `core/dynamic_transformer/pure_code_trainer.py` (تم تحديثه)
- `test_v5_verification.py` (جديد)
- `trained_model_pure/knowledge_map.py` (تنسيق مخرجات جديد)

---

## V4.1 - Pure Code Dynamic Transformer (2026-02-01)

### 🚀 الابتكار الجذري
- **الأوزان ككود Python فقط** - لا `.pt` على الإطلاق!
- **تدريب حقيقي** بـ backpropagation
- **تصدير ككود مضغوط** (base64 + zlib)
- **قابل للتحميل من Python فقط**

### 📁 ملفات جديدة
- `core/dynamic_transformer/pure_dynamic_transformer.py`
- `core/dynamic_transformer/pure_code_trainer.py`
- `test_pure_code_training.py`
- `trained_model_pure/` (النموذج ككود)

### 📤 Output
```
trained_model_pure/final/
├── config.py      # الإعدادات
├── vocab.py       # المفردات
├── weights.py     # الأوزان ككود!
└── loader.py      # للتحميل
```

---

## V4.0 - Dynamic Transformer (2026-02-01)

### 🚀 Innovation
- **Dynamic Transformer**: First Transformer storing weights as Python code
- **WeightCell**: Neural network weights as executable code
- **DynamicEmbedding**: Add new words without retraining
- **DynamicAttention**: Learn and persist attention patterns
- **DynamicFFN**: Dynamically expanding feed-forward network

### 📁 New Files
- `core/dynamic_transformer/` (5 files)
- `vault/transformer/embeddings.py`
- `vault/transformer/config.py`

---

## V3.0 - Dynamic Language Engine (2026-02-01)

### ✨ New Features
- **Dynamic Language Engine** - generates sentences without external LLM
- **WordCell** - word with meaning stored as code
- **PatternCell** - sentence pattern stored as code
- **DynamicVocab** - self-writing vocabulary manager
- **DynamicGrammar** - self-writing grammar manager
- **GenerationEngine** - combines them for generation
- **speak()** method in SelfWritingModel
- **Arabic POS Tagger** - custom Arabic morphological analyzer
- **Language Trainer** - automated training system

### 📁 New Files
- `core/language_engine/` (5 files)
- `training/` (3 files)
- `vault/language/vocab.py`
- `vault/language/patterns.py`

---

## V2.0 - Liquid Engine Integration (2026-01-31)

### ✨ Features
- LiquidEngine for dynamic code generation
- Enhanced self-writing capabilities

---

## V1.0 - Foundation (2026-01-31)

### 🏗️ Core
- DynamicCell - the fundamental unit
- SelfWritingModel - the self-writing model
- Auditor - code validation
- ImmuneSystem - safety checks
- LogicalBridge - Bayan logic integration
