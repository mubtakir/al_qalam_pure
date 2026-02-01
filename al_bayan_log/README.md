# 📜 Al-Bayan Log (أرشيف لغة البيان)

تم نقل هذه المكونات من `baserah_ai` بتاريخ 2026-02-01.

## 📂 المجلدات

| المجلد | الملفات | الوظيفة |
|--------|---------|---------|
| `compiler/` | 12 | Lexer, Parser, LogicalEngine, Compiler |
| `inference/` | 5 | NeuroSymbolicBridge (Syllogism, Induction) |
| `nlp/` | 4 | Arabic NLP (Segment, POS, NER) |
| `safety/` | 4 | ImmuneSystem (Sandbox IDE) |
| `physics/` | 4 | SI Dimensions, PhysicalQuantity |
| `learning/` | 4 | KnowledgeDigester, Dreamer, Librarian |
| `llm/` | 2 | LLM_BRIDGE (Qwen/Llama) |
| `thinking/` | 7 | ThinkingCore (Nodes, Relations) |
| `domains/` | 8 | ملفات المجالات العلمية (.by) |
| `mind/` | 4 | ExistentialCore, UnifiedMind |

## 🚀 الاستخدام

### ImmuneSystem
```python
from al_bayan_log.safety import ImmuneSystem
ims = ImmuneSystem(".")
result = ims.sandbox_test("x = 1 + 1")
# → {"valid": True}
```

### Physics
```python
from al_bayan_log.physics import Kg, Acceleration
f = Kg(10) * Acceleration(5)
# → 50.0 (kg*m/s^2)
```

### NLP
```python
from al_bayan_log.nlp import extract_entities
extract_entities("المهندس أحمد يسكن في الرياض")
# → [{"text": "المهندس أحمد", "type": "PERSON"}]
```

### LLM Bridge
```python
from al_bayan_log.llm import LLMBridge
llm = LLMBridge()  # يحتاج ملف GGUF
response = llm.think("ما هي الجاذبية؟")
```

## 📖 ملفات البيان (.by)

```
domains/
├── al_qalam_concepts.by   # مفاهيم القلم
├── al_qalam_rules.by      # قواعد الاستدلال
├── physics_domain.by      # الفيزياء (760 سطر)
├── chemistry_domain.by    # الكيمياء
├── mathematics_domain.by  # الرياضيات
├── electronics_domain.by  # الإلكترونيات
└── life_domain.by         # علوم الحياة
```

## 🔌 تفعيل LLM

```
al_bayan_log/llm/model/external/
└── qwen2.5-7b-instruct-q4_k_m.gguf
```
