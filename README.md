# Ai-assistant-execl-Controller
Here is a production-ready, professionally structured `README.md` fully tailored to your final architecture. It includes clean system topology diagrams, architectural explanations of your 3-tier routing cascade, transactional guards, and a localized offline-first design philosophy.

---

# Sara — Voice-Driven Enterprise Excel Automation Agent

Sara is an intelligent, production-ready, voice-controlled workflow assistant engineered to automate complex Microsoft Excel and structured data operations using natural language. Built with an enterprise-grade hybrid instruction-parsing pipeline, Sara bridges the gap between flexible human speech and deterministic system controls, guaranteeing absolute execution safety, memory persistence, and anti-hallucination runtime guards during data mutations.

---

## 🏗️ Architectural Overview

Sara is decoupled cleanly across a **Workflow-Agent-Tools (WAT)** design pattern. The system is built with a highly resilient, local-first operational philosophy paired with deterministic fallback channels to maintain structural reliability in production environments.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE                                 │
│                        (Browser / app.py Flask)                             │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │  raw audio / text
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SPEECH-TO-TEXT LAYER                                 │
│                                                                             │
│  ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────────┐    │
│  │  faster-whisper  │   │  Google Cloud STT│   │  Groq Whisper API    │    │
│  │  (offline, fast) │   │  (cloud fallback)│   │  (cloud, high-qual.) │    │
│  └──────────────────┘   └──────────────────┘   └──────────────────────┘    │
│         ▲ primary              ▲ fallback 1            ▲ fallback 2        │
│         └──────────────────────┴────────────────────────┘                  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  openwakeword  — always-on wake-word detection ("Hey Sara")         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │  transcript (str)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PIPELINE SANITIZER                                 │
│                       (pipeline_sanitizer.py)                               │
│                                                                             │
│  • Escape prompt-injection attempts in transcript                           │
│  • Sanitize memory values before insertion into LLM prompt                  │
│  • Validate cell values / formulas at system boundary                       │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │  sanitized text
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LLM ROUTING LAYER                                 │
│                            (llm_router.py)                                  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Tier 1 — Heuristic Parser  (llm_parser._heuristic_*)               │   │
│  │  • 30+ regex rules, zero latency, confidence = 0.95                 │   │
│  │  • Handles: create, add_row, delete, undo, calculate, etc.          │   │
│  └─────────────────────┬───────────────────────────────────────────────┘   │
│            hit (fast)  │  miss                                              │
│                        ▼                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Tier 2 — Ollama  (local, qwen3:8b)  ← GROQ_FIRST=false (default) │   │
│  │  • Offline-capable, streaming NDJSON, circuit-breaker (3 fails)     │   │
│  │  • ConversationBuffer deque(maxlen=5) for multi-turn context        │   │
│  │  • RobustJsonExtractor: 4 extraction strategies                     │   │
│  │  • confidence = 0.82                                                │   │
│  └─────────────────────┬───────────────────────────────────────────────┘   │
│            ok          │  fail / GROQ_FIRST=true                            │
│                        ▼                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Tier 3 — Groq Cloud API  (llama-3 / mixtral)                       │   │
│  │  • 8s timeout, fail-fast, strips <think> tags before JSON parse     │   │
│  │  • GROQ_FIRST=true: tried BEFORE Ollama when online                 │   │
│  │  • confidence = 0.75                                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Pydantic v2 Validation  (validate_command in llm_parser.py)        │   │
│  │  • Type coercion, field normalisation, formal schema enforcement     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │  action dict  {type, params…}
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AGENT LAYER                                    │
│                               (agent.py)                                    │
│                                                                             │
│  handle_text()  →  handle_action()                                          │
│                                                                             │
│  ┌────────────────────┐  ┌────────────────────┐  ┌──────────────────────┐  │
│  │  WorkflowRouter    │  │  Planner           │  │  ErrorHandler        │  │
│  │  • route(action)   │  │  • validate refs   │  │  • unknown_command   │  │
│  │  • preconditions   │  │  • expand steps    │  │  • execution_error   │  │
│  │  • destructive?    │  │  • undo routing    │  │  • system_error      │  │
│  │  • is_destructive  │  │  • destructive_cnt │  │  • classify_exc()    │  │
│  └────────────────────┘  └────────────────────┘  └──────────────────────┘  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Confirmation Gate  — requires_confirmation=True stops execution    │   │
│  │  until caller explicitly sets confirmed=True                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Transactional Multi-step Execution  (_execute)                     │   │
│  │  • Backup file before step 1 → rollback on any step failure         │   │
│  │  • Each step dispatched through ToolRegistry                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                 ┌──────────────────┼──────────────────┐
                 ▼                  ▼                  ▼
┌──────────────────────┐  ┌──────────────────┐  ┌──────────────────────────┐
│   ToolRegistry       │  │  WorkflowEngine  │  │  StateManager            │
│  (tool_registry.py)  │  │ (workflow_engine) │  │  (state_manager.py)      │
│                      │  │                  │  │                          │
│  • register tools    │  │  • YAML workflows│  │  • update_after_action() │
│  • per-tool timing   │  │  • multi-step    │  │  • active_file/sheet     │
│  • BaseTool ABC      │  │  • JSON results  │  │  • active_headers        │
│  • assert_registered │  │                  │  │  • enrich_text()         │
└──────────┬───────────┘  └──────────────────┘  └──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         EXCEL CONTROLLER                                    │
│                        (excel_controller.py)                                │
│                                                                             │
│  All operations use atomic save (tempfile → os.replace)                     │
│  Undo stack: _UNDO_STACK[abs_path] = deque(maxlen=10)                       │
│                                                                             │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌──────────────────────┐ │
│  │  create    │  │  read      │  │  modify    │  │  destructive         │ │
│  │  create_   │  │  read_cell │  │  write_    │  │  delete_row ──┐      │ │
│  │  excel     │  │  read_     │  │  cell      │  │  delete_col ──┼──► _undo_snapshot() │
│  │  create_   │  │  range     │  │  add_row   │  │  find_replace ┘      │ │
│  │  sheet     │  │  list_     │  │  add_col   │  │                      │ │
│  │            │  │  sheets    │  │  write_    │  │  undo_last():        │ │
│  │            │  │            │  │  formula   │  │  ← deque.pop()       │ │
│  │            │  │            │  │  format_   │  │  → atomic restore    │ │
│  │            │  │            │  │  cells     │  │                      │ │
│  └────────────┘  └────────────┘  └────────────┘  └──────────────────────┘ │
│                                                                             │
│  Formula injection guard: =, +, -, @ prefixed with tab-char                │
│  Formula whitelist: 37 allowed functions (SUM, AVERAGE, IF, VLOOKUP, …)    │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │  result dict  {message, speech_text, …}
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MEMORY LAYER                                      │
│                             (memory.py)                                     │
│                                                                             │
│  • SQLite-backed persistent memory (active_file, active_sheet, headers)     │
│  • ConversationBuffer: deque(maxlen=5) multi-turn history                   │
│  • add_command(), get_active_file(), snapshot()                             │
│  • Non-fatal: SQLite failure wrapped in try/except — never aborts pipeline  │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     TEXT-TO-SPEECH (TTS) LAYER                              │
│                          (tts_engine.py)                                    │
│                                                                             │
│  • pyttsx3 on a daemon thread with queue.Queue                              │
│  • Deadlock-safe: drains queue on init failure, checks worker.is_alive()    │
│  • ENABLE_TTS env var to toggle on/off                                      │
│  • ErrorHandler.safe_speech() → speech_text → spoken to user               │
└─────────────────────────────────────────────────────────────────────────────┘

```

---

## 🛠️ Key Design Decisions & Core Safeguards

| Concern | Operational Solution | Technical Implementation |
| --- | --- | --- |
| **Offline-First Mode** | Local processing infrastructure fallback.

 | Integrates `faster-whisper` and local `Ollama` (`qwen3:8b`) models.

 |
| **High-Speed Execution** | Dynamic backend processing bypass environment flags.

 | Flips the parsing order to run `Groq Cloud API` first via `GROQ_FIRST=true`.

 |
| **Malformed LLM Output** | Structural extraction validation engines.

 | `RobustJsonExtractor` running 4 fallback strategies to bypass unstructured text.

 |
| **Prompt Injection** | String-level security boundary monitors.

 | `pipeline_sanitizer.py` brackets and sanitizes in-memory historical text strings.

 |
| **Formula Injection** | Strict data mutation sanitization gates.

 | Validates cell boundaries, appending `tab-char` escapes before raw symbols (`=`, `+`, `-`, `@`).

 |
| **Destructive Actions** | Automated validation confirmation gates.

 | Intercepts structural mutations via `Planner`; `Agent` blocks until `confirmed=True`.

 |
| **State Failures & Rollbacks** | High-performance file checkpoint management.

 | Automatic pre-execution disk snapshots cached inside a bounded in-memory `deque` matrix.

 |
| **Data Synchronization** | Atomic data transactions on disk.

 | File outputs are modified using isolated temporary file structures (`tempfile` ➔ `os.replace`).

 |
| **Internal Data Exposure** | Natural-language mapping abstractions.

 | Domain-aware `ErrorHandler` classifies trace exceptions into clear natural voice alerts.

 |
| **LLM Fail Cascades** | Temporal circuit-breaking runtime controls.

 | Implements an automated cooling threshold (3 consecutive failures blocks processing for 60s).

 |
| **Audio Thread Freezes** | Decoupled background workers.

 | Runs text-to-speech processing on a custom `pyttsx3` daemon thread fed by thread-safe queues.

 |
| **Schema Validation** | Strict structured schema data enforcement.

 | Validates type coercions and fields via a formalized `Pydantic v2` data parser layer.

 |

---

## 📂 Project Repository Layout

```
backend/
├── app.py                  # Flask server routing endpoints, confirmation context hooks[cite: 2, 5]
├── agent.py                # Core application orchestrator (handle_text / handle_action workflows)[cite: 2]
├── llm_router.py           # Three-tier language model routing engine with cloud priorities[cite: 2]
├── llm_parser.py           # Heuristic matching, RobustJsonExtractor, Pydantic v2 validators[cite: 2]
├── planner.py              # Step verification, structural data index range, undo management mapping[cite: 2]
├── excel_controller.py     # openpyxl workbook interface layer, atomic storage engine, rollback deques[cite: 2]
├── error_handler.py        # Exception classification parser generating natural vocal explanations[cite: 2]
├── pipeline_sanitizer.py   # Boundary security sanitizer filtering text-based exploit structures[cite: 2]
├── memory.py               # SQLite infrastructure layer coupled with ConversationBuffer deques[cite: 2]
├── tts_engine.py           # Asynchronous decoupled pyttsx3 daemon background worker thread[cite: 2]
├── assistant_runtime.py    # Mic intake stream capturer, wake-word loop filters, STT routing controllers[cite: 2]
├── workflow_router.py      # Abstract preconditions and schema declarations for file-based tracking[cite: 2]
├── workflow_engine.py      # Dynamic declarative YAML script processing pipeline engine[cite: 2]
├── state/
│   └── state_manager.py    # Runtime context manager (handles working spreadsheets and active column indexes)[cite: 2]
└── tools/
    ├── base_tool.py        # Abstract Base Class schema specifying granular system tools[cite: 2]
    └── tool_registry.py    # Execution tracker monitoring performance metrics and route mappings[cite: 2]

```

---

## ⚡ Quick Start & Run Instructions

### 1. Installation and Virtual Environment Setup

Ensure that Python 3.10+ is available locally before configuring dependencies:

```bash
# Initialize isolated python workspace environment
python -m venv .venv

# Activate and pull system runtime packages
# Windows:
.venv\Scripts\activate
pip install -r backend/requirements.txt

```

### 2. Environment Configuration (`backend/.env`)

Create a local configurations file inside the backend workspace folder using these parameters:

```ini
# Core LLM Processing Settings
GROQ_FIRST=true        # Flips operational pipeline sequence to default to Groq Cloud[cite: 5]
WHISPER_MODEL=medium   # Adjusts the offline accuracy precision depth of faster-whisper engines[cite: 5]
ENABLE_TTS=true        # Enables/Disables background Text-To-Speech engine pipelines[cite: 5]
TTS_RATE=160           # Configures the speed thresholds for raw spoken sentences[cite: 5]

# Wake Word Intakes
WAKE_BUILTIN_KEYWORD=jarvis  # Sets the system wake phrase target structure[cite: 5]
WAKE_THRESHOLD=0.38          # Configures input target listening decibel thresholds[cite: 5]

```

### 3. Execution Options

* **Automated Boot:** Double-click on `start_sara.bat` to launch the persistent backend Flask server alongside the client-side UI popup view simultaneously.


* **Manual Step-By-Step Configuration:**
```bash
# Shell 1: Spin up back-end core services
python backend/app.py

# Shell 2: Boot desktop presentation interfaces
python desktop/sara_ui.py

```



```

---

## 🧪 Operational Demo Testing Guide

You can pass raw data text blocks directly into `/api/parse` via standard JSON structures (`{"text": "<command>", "use_llm": false}`) or trigger them straight through vocal captures[cite: 3]:

* **Table Initialization:** `"create excel with headers \"Name,Score\""`[cite: 3]
  * *Expected Output Struct:* `{"action":"create_table","headers":["Name","Score"]}`[cite: 3]
* **Row Insertion:** `"add row Tom ninety"`[cite: 3]
  * *Expected Output Struct:* `{"action":"add_row","values":["Tom",90]}`[cite: 3]
* **Column Expansion:** `"add column Score"`[cite: 3]
  * *Expected Output Struct:* `{"action":"add_column","name":"Score"}`[cite: 3]
* **Destructive Truncations:** `"delete the third row"`[cite: 3] *(Will prompt the system confirmation gate before mutation)[cite: 2].*
  * *Expected Output Struct:* `{"action":"delete_row","index":3}`[cite: 3]
* **Mathematical Operations:** `"calculate average for column two"`[cite: 3]
  * *Expected Output Struct:* `{"action":"calculate_average","column":2}`[cite: 3]

```
