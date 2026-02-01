# Al-Qalam Pure: Dynamic Cells Registry
from core.dynamic_cell import DynamicCell, AdaptiveParameter


# --- DYNAMIC CELL: concept_human ---
cell_concept_human = DynamicCell(
    cell_id="concept_human",
    cell_type="concept",
    initial_value=1.0,
    metadata={
  "name": "Human",
  "created": "2026-02-01T06:58:44.350387",
  "last_modified": "2026-02-01T09:31:53.684383"
}
)
cell_concept_human.stats = {
  "activation_count": 0,
  "total_activation": 0.0
}
cell_concept_human.memory = []
cell_concept_human.connect_to('inst_human_0', weight=0.9, delta=0.0) # adaptive_slot
cell_concept_human.connect_to('inst_human_1', weight=0.9, delta=0.0) # adaptive_slot
cell_concept_human.connect_to('inst_human_2', weight=0.9, delta=0.0) # adaptive_slot


# --- DYNAMIC CELL: inst_human_0 ---
cell_inst_human_0 = DynamicCell(
    cell_id="inst_human_0",
    cell_type="instance",
    initial_value=0.5,
    metadata={
  "value": "Khalid",
  "created": "2026-02-01T06:58:44.350387",
  "last_modified": "2026-02-01T09:31:53.684383",
  "facts": [
    "likes Apple"
  ]
}
)
cell_inst_human_0.stats = {
  "activation_count": 0,
  "total_activation": 0.0
}
cell_inst_human_0.memory = []
cell_inst_human_0.connect_to('concept_human', weight=0.9, delta=0.0) # adaptive_slot
cell_inst_human_0.connect_to('item_apple', weight=1.0, delta=0.0) # adaptive_slot


# --- DYNAMIC CELL: inst_human_1 ---
cell_inst_human_1 = DynamicCell(
    cell_id="inst_human_1",
    cell_type="instance",
    initial_value=0.5,
    metadata={
  "value": "Zaid",
  "created": "2026-02-01T06:58:44.350387",
  "last_modified": "2026-02-01T09:31:53.684383",
  "facts": [
    "likes Apple"
  ]
}
)
cell_inst_human_1.stats = {
  "activation_count": 0,
  "total_activation": 0.0
}
cell_inst_human_1.memory = []
cell_inst_human_1.connect_to('concept_human', weight=0.9, delta=0.0) # adaptive_slot
cell_inst_human_1.connect_to('item_apple', weight=1.0, delta=0.0) # adaptive_slot


# --- DYNAMIC CELL: inst_human_2 ---
cell_inst_human_2 = DynamicCell(
    cell_id="inst_human_2",
    cell_type="instance",
    initial_value=0.5,
    metadata={
  "value": "Ahmed",
  "created": "2026-02-01T06:58:44.350387",
  "last_modified": "2026-02-01T09:31:53.684383"
}
)
cell_inst_human_2.stats = {
  "activation_count": 0,
  "total_activation": 0.0
}
cell_inst_human_2.memory = []
cell_inst_human_2.connect_to('concept_human', weight=0.9, delta=0.0) # adaptive_slot
cell_inst_human_2.connect_to('item_apple', weight=0.8, delta=0.0) # adaptive_slot


# --- DYNAMIC CELL: item_apple ---
cell_item_apple = DynamicCell(
    cell_id="item_apple",
    cell_type="item",
    initial_value=0.0,
    metadata={
  "name": "Apple",
  "created": "2026-02-01T06:58:44.350837",
  "last_modified": "2026-02-01T09:31:53.684383"
}
)
cell_item_apple.stats = {
  "activation_count": 0,
  "total_activation": 0.0
}
cell_item_apple.memory = []


# --- DYNAMIC CELL: concept_fruit ---
cell_concept_fruit = DynamicCell(
    cell_id="concept_fruit",
    cell_type="concept",
    initial_value=1.0,
    metadata={
  "name": "fruit",
  "created": "2026-02-01T08:54:40.932086",
  "last_modified": "2026-02-01T09:31:53.684383"
}
)
cell_concept_fruit.stats = {
  "activation_count": 0,
  "total_activation": 0.0
}
cell_concept_fruit.memory = []
cell_concept_fruit.connect_to('inst_fruit_0', weight=0.9, delta=0.0) # adaptive_slot
cell_concept_fruit.connect_to('inst_fruit_1', weight=0.9, delta=0.0) # adaptive_slot
cell_concept_fruit.connect_to('inst_fruit_2', weight=0.9, delta=0.0) # adaptive_slot


# --- DYNAMIC CELL: inst_fruit_0 ---
cell_inst_fruit_0 = DynamicCell(
    cell_id="inst_fruit_0",
    cell_type="instance",
    initial_value=0.5,
    metadata={
  "value": "apple",
  "created": "2026-02-01T08:54:40.932086",
  "last_modified": "2026-02-01T09:31:53.684383"
}
)
cell_inst_fruit_0.stats = {
  "activation_count": 0,
  "total_activation": 0.0
}
cell_inst_fruit_0.memory = []
cell_inst_fruit_0.connect_to('concept_fruit', weight=0.9, delta=0.0) # adaptive_slot


# --- DYNAMIC CELL: inst_fruit_1 ---
cell_inst_fruit_1 = DynamicCell(
    cell_id="inst_fruit_1",
    cell_type="instance",
    initial_value=0.5,
    metadata={
  "value": "orange",
  "created": "2026-02-01T08:54:40.932086",
  "last_modified": "2026-02-01T09:31:53.684383"
}
)
cell_inst_fruit_1.stats = {
  "activation_count": 0,
  "total_activation": 0.0
}
cell_inst_fruit_1.memory = []
cell_inst_fruit_1.connect_to('concept_fruit', weight=0.9, delta=0.0) # adaptive_slot


# --- DYNAMIC CELL: inst_fruit_2 ---
cell_inst_fruit_2 = DynamicCell(
    cell_id="inst_fruit_2",
    cell_type="instance",
    initial_value=0.5,
    metadata={
  "value": "banana",
  "created": "2026-02-01T08:54:40.932086",
  "last_modified": "2026-02-01T09:31:53.684383"
}
)
cell_inst_fruit_2.stats = {
  "activation_count": 0,
  "total_activation": 0.0
}
cell_inst_fruit_2.memory = []
cell_inst_fruit_2.connect_to('concept_fruit', weight=0.9, delta=0.0) # adaptive_slot


# --- DYNAMIC CELL: logic_testhello ---
cell_logic_testhello = DynamicCell(
    cell_id="logic_testhello",
    cell_type="logic_concept",
    initial_value=0.0,
    metadata={
  "name": "TestHello",
  "methods": [
    "hello_func"
  ],
  "code": "def hello():\n    return 'Hello World'\n",
  "created": "2026-02-01T09:31:20.881891",
  "last_modified": "2026-02-01T09:31:53.684383"
}
)
cell_logic_testhello.stats = {
  "activation_count": 0,
  "total_activation": 0.0
}
cell_logic_testhello.memory = []


# --- DYNAMIC CELL: logic_greeting ---
cell_logic_greeting = DynamicCell(
    cell_id="logic_greeting",
    cell_type="logic_concept",
    initial_value=0.0,
    metadata={
  "name": "Greeting",
  "methods": [
    "greet"
  ],
  "code": "def greet(name: str) -> str:\n    return f\"Hello, {name}!\"\n",
  "created": "2026-02-01T09:31:53.685014",
  "last_modified": "2026-02-01T09:31:53.685014"
}
)
cell_logic_greeting.stats = {
  "activation_count": 0,
  "total_activation": 0.0
}
cell_logic_greeting.memory = []

