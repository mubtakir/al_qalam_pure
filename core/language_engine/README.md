# 🗣️ Dynamic Language Engine V3.0

المحرك اللغوي الديناميكي - قلب القلم الناطق

## 📁 الملفات

| الملف | الوظيفة |
|-------|---------|
| `word_cell.py` | خلية تُخزن كلمة ومعناها ككود |
| `pattern_cell.py` | خلية تُخزن نمط جملة |
| `dynamic_vocab.py` | مدير مفردات يكتب نفسه |
| `dynamic_grammar.py` | مدير أنماط يكتب نفسه |
| `generation_engine.py` | محرك التوليد |

## 🚀 الاستخدام

### مباشرة
```python
from core.language_engine import GenerationEngine

engine = GenerationEngine()
engine.vocab.add_word("القط", pos="NOUN")
engine.vocab.add_word("يأكل", pos="VERB")
engine.grammar.learn_from_tagged(
    ["القط", "يأكل", "السمك"],
    ["NOUN", "VERB", "NOUN"],
    "القط يأكل السمك"
)

# توليد
print(engine.generate_simple("الكلب", "يشرب", "الماء"))
```

### عبر SelfWritingModel
```python
from core.self_writing_model import SelfWritingModel

model = SelfWritingModel(".")
model.learn_language("القط يأكل السمك")
print(model.speak("الكلب", "يحب", "اللعب"))
model.persist_language()
```

## 📝 الملفات المُولّدة

```
vault/language/
├── vocab.py      # المفردات ككود Python
└── patterns.py   # الأنماط ككود Python
```

### مثال vocab.py
```python
word_القط = WordCell(
    word="القط",
    pos="NOUN",
    examples=["القط يأكل السمك"]
)
word_القط.frequency = 5
```

## 🔄 حلقة التعلم

```
نص → learn_from_text() → WordCell + PatternCell
                              ↓
                    persist() → vocab.py + patterns.py
                              ↓
                    generate_from_pattern() → جملة جديدة
```

## ✅ المميزات

- ✅ لا يحتاج LLM خارجي
- ✅ يتعلم من النصوص العربية
- ✅ يولّد جمل صحيحة نحوياً
- ✅ يحفظ المفردات ككود مقروء
- ✅ يدعم أنماط متعددة
