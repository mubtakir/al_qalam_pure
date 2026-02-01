# 🔧 Technical Guide: Al-Qalam V2.3

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    SelfWritingModel                          │
│  ┌─────────┐ ┌──────────┐ ┌─────────┐ ┌─────────────────┐  │
│  │ Cells   │ │ Rules    │ │ Auditor │ │ LogicalBridge   │  │
│  │(Dict)   │ │(List)    │ │         │ │ (Bayan Engine)  │  │
│  └────┬────┘ └────┬─────┘ └────┬────┘ └────────┬────────┘  │
│       │           │            │               │            │
│  ┌────▼───────────▼────────────▼───────────────▼────────┐  │
│  │                    Core Methods                       │  │
│  │  learn_concept() | add_fact() | infer() | query()    │  │
│  └──────────────────────────────────────────────────────┘  │
│                              │                              │
│  ┌───────────────────────────▼──────────────────────────┐  │
│  │               Bayan Integration Layer                 │  │
│  │  ┌──────────────┐  ┌───────────┐  ┌───────────────┐  │  │
│  │  │ ImmuneSystem │  │ LLMBridge │  │ KnowledgeDigester │ │
│  │  │ (Safety)     │  │ (Donor)   │  │ (Learning)    │  │  │
│  │  └──────────────┘  └───────────┘  └───────────────┘  │  │
│  └──────────────────────────────────────────────────────┘  │
│                              │                              │
│  ┌───────────────────────────▼──────────────────────────┐  │
│  │                 Persistence Layer                     │  │
│  │        vault/auto_generated/cells.py                  │  │
│  │        vault/auto_generated/rules.py                  │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

### Learning Flow
```
User Input → learn_concept()
    │
    ▼
Create DynamicCell → Connect Cells
    │
    ▼
sync_cell() → LogicalBridge (Bayan Engine)
    │
    ▼
persist_cells() → ImmuneSystem.sandbox_test()
    │
    ├─ Valid → Write to vault/cells.py
    └─ Invalid → Reject (no write)
```

### Inference Flow
```
Context → infer()
    │
    ▼
For each rule in self.rules:
    │
    ├─ Check condition(model, context)
    │   │
    │   ├─ True → Execute action(model, context)
    │   │           └─ May create new cells/facts
    │   │
    │   └─ False → Skip
    │
    ▼
Return triggered_rules[]
```

### LLM Generation Flow
```
Task Description → generate_code_with_llm()
    │
    ▼
LLMBridge.think_and_validate()
    │
    ▼
ImmuneSystem.sandbox_test()
    │
    ├─ Valid → Return code
    │
    └─ Invalid → Add error to prompt
                  │
                  └─ Retry (max 3 times)
```

## Component Details

### DynamicCell
- **Purpose**: Stores knowledge as executable code
- **Key Methods**:
  - `activate(input)`: Sigmoid activation
  - `connect_to(target, weight)`: Create weighted connection
  - `to_source_code()`: Serialize to Python

### ImmuneSystem
- **Purpose**: Prevent broken code from being saved
- **Key Methods**:
  - `validate_code(code)`: Static syntax check
  - `sandbox_test(code)`: Runtime execution test
  - `safe_write(file, old, new)`: Atomic safe update

### LLMBridge
- **Purpose**: Connect to external LLM for intelligent generation
- **Backends**: llama-cpp-python, ctransformers, binary
- **Key Methods**:
  - `think(prompt)`: Simple query
  - `think_and_validate(prompt)`: Query + code extraction + validation

### LogicalBridge
- **Purpose**: Interface between cells and Bayan LogicalEngine
- **Key Methods**:
  - `sync_cell(cell)`: Convert cell to logical facts
  - `add_rule(rule)`: Add logical rule
  - `query(question)`: Natural language query

## File Formats

### Bayan (.by) Format
```bayan
مجال "القلم":
{
    "كائن_أساسي": "خلية",
    "معانٍ_أساسية": ["مفهوم", "مثال", "قاعدة"]
}

كائن_وجودي "خلية" من_نوع "خلية" في_مجال "القلم":
{
    "خصائص_ذاتية": {...},
    "علاقات": {...}
}
```

### Generated Code (cells.py)
```python
# Al-Qalam Pure: Dynamic Cells Registry
from core.dynamic_cell import DynamicCell, AdaptiveParameter

cell_concept_fruit = DynamicCell(
    cell_id="concept_fruit",
    cell_type="concept",
    metadata={"name": "فاكهة"}
)
cell_concept_fruit.connect_to('inst_apple', weight=0.9)
```

## Configuration

### Environment Variables
- None required (all paths are relative)

### Model Configuration
Place GGUF files in: `al_bayan_log/llm/model/external/`

## Testing

```bash
# Full integration test
python test_v23_integration.py

# Specific component tests
python -c "from al_bayan_log.safety import ImmuneSystem; ..."
python -c "from al_bayan_log.physics import Kg; ..."
```
