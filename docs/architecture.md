3️⃣ docs/architecture.md
# Sara System Architecture

Sara is a **research-quality voice-driven AI assistant** for Excel automation,
built on the **WAT (Workflows–Agents–Tools) architecture** with persistent
memory, structured metrics, and transactional execution.

---

## High-Level Pipeline

```
Wake Word Detection  (wakeword/wake_detector.py)
        ↓
Floating Assistant UI (desktop/sara_ui.py  |  frontend/index.html)
        ↓
Speech-to-Text  (backend/speech_recognition.py)
  ├─ Google STT REST
  ├─ Google STT SDK
  └─ Groq Whisper
        ↓
LLM Command Router  (backend/llm_router.py)
  ├─ Groq (LLaMA 3.3 70B, cloud)
  ├─ Ollama (qwen3:8b, local)
  └─ Heuristic fallback (regex)
        ↓
Workflow Router  (backend/workflow_router.py)
  └─ Precondition checking
        ↓
Agent Orchestrator  (backend/agent.py)
  ├─ Transactional multi-step execution
  └─ Auto-correction / recovery
        ↓
Excel Controller  (backend/excel_controller.py)
  └─ 14 action types via openpyxl
        ↓
Persistent Memory  (backend/memory.py, SQLite)
        ↓
Metrics Collector  (backend/metrics.py, SQLite)
        ↓
Structured JSON Logging  (app.py middleware)
```

---

## Layer Responsibilities

### 1. Agent Layer (`backend/agent.py`)

The central orchestrator that ties together parsing, routing, execution, and memory.

- `Agent.handle_text(text)` — parse → route → execute → memory update
- `Agent.handle_action(action)` — route → preconditions → execute → memory update
- Returns a uniform `AgentResult` dataclass for every operation

**Transactional Execution**: Multi-step plans are wrapped in a transactional
boundary. A file backup is taken before step 1; if any step fails, the file is
rolled back to the pre-execution state and an error report is returned listing
completed and failed steps.

### 2. Memory Layer (`backend/memory.py`)

Persistent context store backed by **SQLite** (WAL mode).

| Field           | Purpose                                  |
|----------------|------------------------------------------|
| `active_file`  | Currently targeted Excel file             |
| `active_sheet` | Active sheet within the workbook          |
| `recent_commands` | Bounded deque of last 20 executed actions |

Data persists across server restarts via `backend/data/memory.db`.

### 3. Workflow Layer (`backend/workflow_router.py`, `workflows/*.md`)

Defines structured markdown procedures for each task type.

Each workflow includes:
- Objective
- Inputs with validation
- Preconditions (e.g., file must exist)
- Execution steps with tool calls
- Destructive action classification

11 registered workflows: `create_excel_file`, `open_excel_file`, `insert_data`,
`add_column`, `delete_column`, `calculate_sum`, `calculate_average`,
`format_cells`, `create_chart`, `create_sales_report`.

### 4. Tool Layer (`backend/excel_controller.py`)

Deterministic Excel operations via **openpyxl** with atomic saves.

| Action Type              | Description                       |
|--------------------------|-----------------------------------|
| `create_excel`           | Create new workbook with headers  |
| `create_sheet_and_add_row` | Create + insert first row       |
| `add_row`                | Append row                        |
| `delete_row`             | Delete row by index               |
| `add_column`             | Add named column                  |
| `delete_column`          | Delete column by letter/index     |
| `write_cell`             | Write value to cell               |
| `write_formula`          | Write Excel formula to cell       |
| `calculate_row_sum`      | Sum numeric values in a row       |
| `calculate_average`      | Average of a column               |
| `open_file`              | Open and return file metadata     |
| `format_cells`           | Bold, italic, color, number fmt   |
| `create_chart`           | Bar, Line, Pie, Scatter charts    |
| `save`                   | No-op (all ops save atomically)   |

### 5. LLM Routing (`backend/llm_router.py`process_command("delete cell A1") )

Three-tier hybrid routing with full metrics instrumentation:

```
route_llm(text)
├── 1. Groq (cloud)       — LLaMA 3.3 70B via HTTP API
├── 2. Ollama (local)      — qwen3:8b, if USE_OLLAMA=true
└── 3. Heuristic fallback  — regex pattern matching
```

Each call records to `metrics.db`: provider, latency_ms, fallback flag,
action_type, success/failure.

### 6. Speech-to-Text (`backend/speech_recognition.py`)

Three-tier fallback for audio transcription:

```
transcribe_audio_file(path, language)
├── 1. Google STT REST API
├── 2. Google Cloud STT SDK (grpc)
└── 3. Groq Whisper
```

---

## Research & Evaluation Infrastructure

### Metrics Collector (`backend/metrics.py`)

Thread-safe SQLite-backed collector (WAL mode, `data/metrics.db`).

Three tables:
- **`llm_calls`** — provider, model, latency_ms, fallback, action_type, success, timestamp
- **`executions`** — action_type, success, error, latency_ms, timestamp
- **`requests`** — endpoint, method, status, latency_ms, timestamp

Exposed via:
- `GET /api/metrics` — aggregated summary (total calls, avg latency, fallback rate, success rate)
- `POST /api/metrics/reset` — clear all metrics (pre-benchmark)

### Evaluation Framework (`backend/evaluation.py`)

Benchmark suite measuring intent parsing accuracy:

- 21 built-in test cases covering all action types + unknown/edge cases
- Measures: accuracy, avg latency, fallback rate, per-type breakdowns
- CLI: `python evaluation.py --report --csv results.csv`
- Custom datasets via `--dataset custom.json`

### Structured Logging

All log output is **JSON-lines** format with:
- ISO timestamp, log level, logger name
- `request_id` (UUID hex, 12 chars) attached per HTTP request
- Per-request summary: method, path, status, latency_ms, user_command, llm_provider

---

## API Endpoints

| Method | Path               | Purpose                              |
|--------|--------------------|--------------------------------------|
| POST   | `/api/parse`       | Parse text → action via LLM          |
| POST   | `/api/execute`     | Execute action on Excel file         |
| POST   | `/api/confirm`     | Confirm/cancel destructive action    |
| POST   | `/api/undo`        | Restore from backup checkpoint       |
| GET    | `/api/backups`     | List available checkpoints           |
| POST   | `/api/upload-audio`| Upload audio → transcribe → execute  |
| GET    | `/api/memory`      | Get current memory state             |
| POST   | `/api/memory`      | Update memory fields                 |
| DELETE | `/api/memory`      | Clear all memory                     |
| GET    | `/api/workflows`   | List registered workflows            |
| GET    | `/api/system_status` | Monitoring snapshot                |
| GET    | `/api/metrics`     | Aggregated performance metrics       |
| POST   | `/api/metrics/reset` | Reset metrics                      |
| GET    | `/api/files`       | List user Excel files with metadata  |
| POST   | `/api/delete_file` | Delete file and all backups          |
| GET    | `/files/<name>`    | Download file                        |

---

## Desktop UI Architecture (`desktop/sara_ui.py`)

PyQt6 floating window with three tabs:

| Tab          | Purpose                                    |
|--------------|--------------------------------------------|
| **Assistant** | Mic recording, transcript, AI response     |
| **Status**    | Live system state, memory, recent commands |
| **Workflows** | Browse registered workflows                |

State machine: `Idle → Listening → Processing → Success / Error`

Features: system tray, wake word activation, recording animation, smart
suggestions, developer mode, draggable frameless always-on-top window.

---

## Environment Variables

| Variable        | Default       | Purpose                    |
|-----------------|---------------|----------------------------|
| `GROQ_API_KEY`  | —             | Cloud LLM API key          |
| `GOOGLE_API_KEY`| —             | Google STT API key         |
| `USE_OLLAMA`    | `false`       | Enable local LLM fallback  |
| `OLLAMA_MODEL`  | `qwen3:8b`   | Local Ollama model name    |
| `ENABLE_TTS`    | `false`       | Enable text-to-speech      |
| `MAX_BACKUPS`   | `5`           | Rotating backup count      |

---
 
## Key Design Principles

1. **LLM = Planning, Tools = Execution** — AI decides *what*, deterministic code does *how*
2. **Graceful Degradation** — Every component has a fallback chain
3. **Transactional Safety** — Multi-step operations roll back on failure
4. **Persistent State** — SQLite-backed memory and metrics survive restarts
5. **Research Instrumentation** — Every LLM call and request is metered for evaluation
 