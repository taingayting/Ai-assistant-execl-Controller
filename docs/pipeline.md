# Sara Execution Pipeline

This document describes the **complete execution pipeline** of Sara.

---

# Full System Pipeline


Wake Word Detection
        ↓
Floating UI (PyQt)
        ↓
Speech Recognition
        ↓
Multilingual Processing
        ↓
🧠 Command Router
        ↓
Rule-Based Parser
        ↓
LLM Router
        ↓
🧠 Task Planner
        ↓
Memory
        ↓
Workflow Selection
        ↓
Validator
        ↓
Executor (Tools)
        ↓
Excel Controller
        ↓
Error Recovery
        ↓
Feedback UI



---

# Pipeline Explanation

## Wake Word Detection

Detects activation phrase such as:


Hey Sara


---

## Floating UI

A small desktop interface that appears when Sara is activated.

Built using **PyQt**.

---

## Speech Recognition

Converts voice input into text.

Example technologies:

- Whisper
- Google Speech API

---

## Command Router

Determines how to process the command.

Routes commands to appropriate modules.

---

## LLM Router

Chooses between available models:

| Model | Purpose |
|------|------|
| Ollama | Local inference |
| Groq | Complex reasoning |

---

## Task Planner

Converts natural language commands into structured tasks.

---

## Validator

Checks whether the planned actions are valid and safe.

---

## Executor

Executes deterministic tool functions.

---

## Excel Controller

Handles spreadsheet operations using Python.

---

## Error Recovery

Handles errors and retries failed operations.

---

## Feedback UI

Returns results to the user interface.