# Sara — Final Architecture

## Complete Pipeline (Voice → Excel → Voice)

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


## Key Design Decisions

| Concern               | Solution                                                |
|-----------------------|---------------------------------------------------------|
| Offline-first         | Ollama (qwen3:8b) as Tier 2, Groq as cloud fallback     |
| Fast online mode      | GROQ_FIRST=true env var flips Tier 2/3 order            |
| JSON robustness       | RobustJsonExtractor — 4 strategies, never hard-fails    |
| Prompt injection      | pipeline_sanitizer.py brackets all memory/history text  |
| Formula injection     | Tab-prefix escape for =, +, -, @ at boundary            |
| Dangerous actions     | Planner flags destructive; Agent gates on confirmation  |
| Undo / rollback       | File-level snapshot (bytes) in bounded in-memory deque  |
| Transaction safety    | Backup → execute steps → rollback on failure            |
| Error speech          | ErrorHandler — 3 categories, never exposes internals    |
| LLM failure cascade   | Circuit breaker (3 Ollama fails → 60s cooldown)         |
| TTS stability         | Daemon thread + queue; deadlock-safe shutdown           |
| Schema enforcement    | Pydantic v2 validates every LLM output before dispatch  |


## File Map

```
backend/
├── app.py                  Flask routes, confirmation flow
├── agent.py                Central orchestrator (handle_text / handle_action)
├── llm_router.py           Three-tier LLM routing with GROQ_FIRST support
├── llm_parser.py           Heuristic + LLM parse, RobustJsonExtractor, Pydantic v2
├── planner.py              Step validation, cell/sheet/range checks, undo routing
├── excel_controller.py     openpyxl operations + undo stack + atomic save
├── error_handler.py        Domain-aware error → natural voice message mapping
├── pipeline_sanitizer.py   Injection prevention at all system boundaries
├── memory.py               SQLite + ConversationBuffer
├── tts_engine.py           pyttsx3 daemon thread TTS
├── assistant_runtime.py    Wake-word loop, mic capture, STT orchestration
├── workflow_router.py      Pre-built workflow definitions and preconditions
├── workflow_engine.py      YAML workflow execution engine
├── state/
│   └── state_manager.py    Active file/sheet/headers tracking + enrich_text()
└── tools/
    ├── base_tool.py         BaseTool ABC (ExcelReadTool, ExcelWriteTool, …)
    └── tool_registry.py     ToolRegistry: register, dispatch, timing, assert_registered
```
