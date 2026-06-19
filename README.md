# Sara — Voice-Driven Enterprise Excel Automation Agent

Sara is an intelligent, voice-controlled workflow assistant engineered to automate complex Microsoft Excel operations and structured data transformations using natural language. Built with an enterprise-grade hybrid instruction-parsing pipeline, Sara bridges the gap between flexible human speech and deterministic system controls—guaranteeing absolute execution safety, memory persistence, and anti-hallucination runtime guards during data mutations.

---

## 🏗️ System Architecture & Request Pipeline

Sara is decoupled cleanly across a **Workflow-Agent-Tools (WAT)** design pattern, operating either as a client-server desktop system or an offline-first headless listener.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INTERFACE LAYERS                               │
│  ┌───────────────────────────────┐     ┌─────────────────────────────────┐  │
│  │   Desktop UI (sara_ui.py)     │     │     Web Frontend (HTML5)        │  │
│  │   • PyQt6 Floating Panel      │     │     • Browser-based Mic/Audio   │  │
│  │   • Native Mic Stream Capture │     │     • REST Preview & Dashboard  │  │
│  └───────────────┬───────────────┘     └─────────────────┬───────────────┘  │
└──────────────────┼───────────────────────────────────────┼──────────────────┘
                   │ raw audio / text commands             │
                   ▼                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            FLASK API BACKEND                                │
│                                (app.py)                                     │
│                                                                             │
│  GET / ➔ Serves Web UI  |  POST /api/parse ➔ Intent Classification           │
│  POST /api/execute ➔ Action Engine  |  POST /api/upload-audio ➔ STT Pipe    │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │ 
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     CORE COGNITIVE & PROCESSING PIPELINE                    │
│                                                                             │
│  1. SPEECH-TO-TEXT LAYER (speech_recognition.py)                             │
│     ➔ Primary: faster-whisper (Local Engine) with Speech Context Biasing    │
│     ➔ Fallback: Google Cloud Speech-to-Text / Cloud API Providers           │
│                                                                             │
│  2. SANITIZATION BOUNDARY (pipeline_sanitizer.py)                           │
│     ➔ Prompt Injection Shielding & Formula-Injection Neutralization         │
│                                                                             │
│  3. TRIPLE-TIER LLM ROUTER (llm_router.py / llm_parser.py)                  │
│     ➔ Tier 1: Heuristic Engine (30+ Regex Rules, 0ms Latency Fast-Path)     │
│     ➔ Tier 2: Local Ollama (qwen2.5-coder:3b / DeepSeek Context Aware)      │
│     ➔ Tier 3: Groq Cloud API API Gateway (Llama-3 High-Throughput Fallback) │
│     ➔ Validation: Pydantic v2 Type-Enforcement & RobustJsonExtractor        │
│                                                                             │
│  4. ORCHESTRATION & PLANNING (agent.py / planner.py)                        │
│     ➔ State Enrichment (memory.py SQLite Context & multi-turn Deque history)│
│     ➔ Precondition Auditing & Validation via workflow_router.py            │
│     ➔ Confirmation Gate Interceptor for High-Risk Destructive Actions      │
│                                                                             │
│  5. WORKFLOW MUTATION ENGINE (workflow_engine.py)                           │
│     ➔ Compiles declarative YAML execution sequences into transactional steps │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │ Verified Action Payload
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DETERMINISTIC WORKSPACE EXECUTION                        │
│                          (excel_controller.py)                              │
│                                                                             │
│  • openpyxl & Windows COM Integration  • Active Header Tracking Matrix      │
│  • Bounded Rolling Undo Queue Stack    • Atomic Persistence (Temp ➔ Replace) │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │ Response Meta & Output Voice Strings
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DAEMON TEXT-TO-SPEECH (TTS)                          │
│                               (tts_engine.py)                               │
│                                                                             │
│  • Thread-Safe Asynchronous pyttsx3 Loop fed by decoupled worker Queues     │
└───────────────────────────────────┬─────────────────────────────────────────┘

```

---

## 🚀 Dual Runtime Execution Modes

Sara adapts dynamically to your workspace needs, offering two distinct execution paths:

### 1. Unified Desktop Environment (Default GUI Mode)

Triggered via `start_sara.bat` or `start_sara.pyw`. This script initializes the decoupled systems simultaneously:

* **The Backend Container (`app.py`):** Spins up the core Flask API on `port 5000` to serve endpoints, handle context parsing, manage transaction states, and log runtime analytics.


* **The Native Viewport (`sara_ui.py`):** A lightweight, borderless, floating **PyQt6 desktop interface** that captures microphone inputs, relays audio frames via REST payloads, renders execution JSON, and manages sheet data actions.



### 2. Headless Runtime Loop (Ambient Native Mode)

Triggered directly via `python backend/run_sara.py`. This mode bypasses graphical processing entirely to run an ambient background daemon:

* Integrates an interactive `openwakeword` loop continuously listening for the system wake phrase (**"Hey Jarvis"**).


* Upon activation, it opens a microphone capture window, pipes audio data through `faster-whisper`, maps intents via the `llm_router.py` cascade, modifies target files via `excel_controller.py`, and vocalizes direct summaries using the isolated `tts_engine.py` background thread.



---

## 📋 Interactive Command Cheat Sheet

The parsing pipeline natively supports natural language flexibilities across these primary syntax targets. Commands can be directly spoken into the interface or issued via client endpoints.

### 1. File & Workbook Management

| Target Operation | Example Spoken Voice Command | Functional Description |
| --- | --- | --- |
| **Create File** | `create excel sales.xlsx` | Creates a completely new workbook container with the specified filename.

 |
| **Launch App** | `open excel` | Instantiates Microsoft Excel or reveals the working project directory. |
| **Open Target** | `open file filterr.xlsx` | Targets, reads, and loads a specific workbook file structure into session memory. |
| **Rename File** | `rename file to test_rename_deo.xlsx` | Safe hot-swapping of the physical workbook filename on disk storage. |
| **Save Workbook** | `save` | Forces the system to commit all active memory cells securely onto disk storage.

 |
| **Terminate App** | `close excel` | Commits remaining data frames and safely closes active execution handles. |

### 2. Structural Row & Column Modifications

| Target Operation | Example Spoken Voice Command | Functional Description |
| --- | --- | --- |
| **Append Row** | `add row Mine 90` | Appends a fresh row entry populated with specified text and value arrays.

 |
| **Insert Row** | `insert row 1` | Injects an empty structural row precisely at the specified vertical index slot. |
| **Append Columns** | `add column Name Score Age` | Expands workspace bounds by generating multiple column headers at once.

 |
| **Insert Column** | `insert column B` | Injects an empty vertical data vector precisely at the specified column letter. |

### 3. Granular Cell Assertions & Deletions

| Target Operation | Example Spoken Voice Command | Functional Description |
| --- | --- | --- |
| **Write Cell** | `write cell C1 Name` | Writes plaintext or numerical attributes to exact coordinate keys. |
| **Set Cell Value** | `set cell D1 Test` | Targets an individual cell block grid coordinate to overwrite its value array. |
| **Clear Range** | `delete range A1:B3` | Truncates cellular data matrices cleanly while keeping cell structures intact. |
| **Delete Row** | `delete row 5` | Purges the entire target vertical data line and shifts rows upward.

 |
| **Delete Column** | `delete column C` | Drops the designated column vector entirely and recalculates grid letters. |
| **Clear Cell** | `delete cell A2` | Flushes structural data contents inside a singular target cell block. |

### 4. Mathematical Formula Injection

| Target Operation | Example Spoken Voice Command | Functional Description |
| --- | --- | --- |
| **Calculate Mean** | `average B2:B5 into C7` | Processes an explicit mathematical mean and routes the formula output.

 |
| **Calculate Total** | `sum B2:B5 into C5` | Compiles targets into an automated arithmetic summation sequence block.

 |

### 5. Formatting, Layout Styling & Alignments

| Category Target | Example Spoken Voice Command | Functional Description |
| --- | --- | --- |
| **Cell Typography** | `bold A1` | `italic C6` | Sets clean styling structures onto targeted grid tracking nodes.

 |
| **Row Typography** | `bold row 6` | `format row 3 bold` | Changes entire structural row weights simultaneously.

 |
| **Col Typography** | `italic column C` | `format column D bold` | Scales formatting configurations down an entire vertical dataset.

 |
| **Grid Alignment** | `center B2` | `center column B` | Justifies elements evenly along the document center axis. |
| **Bound Alignment** | `align A2:B2 right` | `align row 3 left` | Aligns text properties to chosen page boundary orientations. |
| **Static Sizing** | `set font size row 3 16` | Forces the tracking grid to adopt precise font layout pixel bounds. |
| **Dynamic Scaling** | `increase font size A1` | `decrease font size row 2` | Shifts typographic size profiles up or down on target elements. |
| **Text Color** | `make A3 orange` | `color row 3 blue` | Alters foreground font color channels across coordinates. |
| **Cell Fills** | `fill row 2 with red` | `fill A4 pink` | Injects vibrant background colors directly behind text arrays.

 |

### 6. Interface Controls, Layout Sorting & Sheet Filters

| Target Operation | Example Spoken Voice Command | Functional Description |
| --- | --- | --- |
| **Freeze Header** | `freeze header` | Locks row 1 securely in place above scrolling document vectors.

 |
| **Freeze Panes** | `freeze panes` | Locks workspace windows dynamically at active coordinate positions. |
| **Unfreeze Global** | `unfreeze panes` | `unfreeze` | Returns the viewport layout back to default scrolling operations.

 |
| **Merge Blocks** | `merge cells A1:C1` | Fuses separated grid systems into a singular shared interface element. |
| **Unmerge Blocks** | `unmerge cells A1:C1` | Splits merged layouts back into individual independent coordinates. |
| **Column Widths** | `set column width A 5` | Modifies spatial column lines to fit strict sizing metrics. |
| **Auto Fit Width** | `auto fit all columns` | Scans text strings to wrap column width structures perfectly with zero data overflow. |
| **Find & Replace** | `replace Tom with John` | Executes search sweeps to overwrite string matches sheet-wide. |
| **Data Sorting** | `sort by B ascending` | `sort by A descending` | Sorts row rows sequentially based on chosen column data trends. |
| **Create Worksheet** | `create sheet Report` | Injects an entirely new sheet page layer tab into the working session. |
| **Rename Sheet** | `rename sheet Sales to Market` | Switches target sheet reference titles to keep datasets clearly organized. |
| **Switch Context** | `switch to Buy` | Points the active backend pipeline tools to a completely different sub-sheet tab. |
| **Drop Worksheet** | `delete sheet Report` | Destroys specified sub-sheet structures permanently from the file tree. |
| **Apply Filter** | `filter column B = 90` | Conceals rows that fail to match designated criteria parameters. |
| **Flush Filters** | `clear filter` | Flushes structural filter locks to reveal hidden data matrices. |
| **Revert Change** | `undo` | Pops the last state snapshot from the rolling deque to instantly undo mistakes.

 |

---

## 🔒 Key Design Decisions & Core Safeguards

| Concern | Operational Solution | Technical Implementation |
| --- | --- | --- |
| **Offline Isolation** | Local processing infrastructure fallback.

 | Integrates `faster-whisper` and local `Ollama` (`qwen2.5-coder:3b`) models.

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

## 📂 Project Directory Architecture

```
Ai-assistant-excel-Controller/
├── .github/workflows/         # Continuous integration and automation actions
├── backend/                   # Core application service matrix[cite: 2]
│   ├── state/                 # State management context boundaries[cite: 2]
│   │   └── state_manager.py   # Runtime context tracking active worksheets & boundaries[cite: 2]
│   ├── tools/                 # Atomic execution utilities[cite: 2]
│   │   ├── base_tool.py       # Abstract Base Class schema specifying granular tools[cite: 2]
│   │   └── tool_registry.py   # System execution dispatcher tracking operational metrics[cite: 2]
│   ├── app.py                 # Core Flask REST routing layer & transaction confirmation hooks[cite: 2, 5]
│   ├── agent.py               # Orchestrator coordinating core analysis & pipeline transformations[cite: 2]
│   ├── llm_router.py          # Three-tier language model routing engine with cloud priorities[cite: 2]
│   ├── llm_parser.py          # Heuristics matching, RobustJsonExtractor, Pydantic v2 validation[cite: 2]
│   ├── planner.py             # Multi-step layout tracking & target mapping verifiers[cite: 2]
│   ├── excel_controller.py    # openpyxl engine workbook interface layer & atomic transaction replace[cite: 2]
│   ├── error_handler.py       # Domain exception processing generating natural voice outputs[cite: 2]
│   ├── pipeline_sanitizer.py  # Security perimeter escaping formula and string exploits[cite: 2]
│   ├── memory.py              # Persistent SQLite interaction schema with active cell histories[cite: 2]
│   ├── tts_engine.py          # Asynchronous decoupled pyttsx3 daemon background worker thread[cite: 2]
│   ├── speech_recognition.py  # Audio normalizer, faster-whisper interface & biasing arrays
│   ├── run_sara.py            # Main entrypoint loop for headless native microphone execution
│   ├── workflow_router.py     # Structural validation constraints matching actions to steps[cite: 2]
│   └── workflow_engine.py     # Dynamic declarative YAML layout instruction interpreter[cite: 2]
├── desktop/                   # Client view abstractions[cite: 5]
│   └── sara_ui.py             # Borderless floating PyQt6 audio record and response display panel[cite: 5]
├── frontend/                  # Web companion alternative interface
│   ├── index.html             # HTML5 layout providing alternate microphone controls
│   ├── script.js              # Client endpoint communications handler
│   └── style.css              # Frontend visual interface styles
├── workflows/                 # Markdown/YAML Declarative Step Blueprints[cite: 2]
├── files/                     # Sandbox environment containing active generated spreadsheet outputs[cite: 5]
├── start_sara.bat             # Single-click unified system launcher script[cite: 5]
└── README.md                  # Main documentation file[cite: 2]

```

---

## ⚡ Setup & Launch Instructions

### 1. Initialize Virtual Environment and Workspace Dependencies

Ensure Python 3.10+ is available locally before configuring dependencies:

```bash
# Initialize isolated Python workspace environment
python -m venv .venv

# Activate environment and pull system runtime packages
# On Windows:
.venv\Scripts\activate
pip install -r backend/requirements.txt

```

### 2. Configure Environment Options (`backend/.env`)

Create a local configurations file inside your backend directory using these parameters:

```ini
# Core LLM Processing Settings
GROQ_FIRST=true        # Flips operational pipeline sequence to default to Groq Cloud API[cite: 5]
WHISPER_MODEL=medium   # Adjusts the offline accuracy precision depth of faster-whisper engines[cite: 5]
ENABLE_TTS=true        # Enables or disables background Text-To-Speech engine pipelines[cite: 5]
TTS_RATE=160           # Configures the speed thresholds for raw spoken sentences[cite: 5]

# Wake Word Intakes
WAKE_BUILTIN_KEYWORD=jarvis  # Sets the system wake phrase target structure[cite: 5]
WAKE_THRESHOLD=0.38          # Configures input target listening decibel thresholds[cite: 5]

```

### 3. Execution Options

* **Unified Boot (Desktop Mode):** Double-click on `start_sara.bat` in the root folder. This automatically boots the backend API alongside the floating PyQt6 voice panel interface (`desktop/sara_ui.py`).


* **Headless Boot (Ambient Mode):** Run `python backend/run_sara.py` within an activated virtual environment terminal to track instructions purely via hotkeys and vocal indicators.

