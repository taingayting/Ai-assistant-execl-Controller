README.md
# 🎤 Sara — Voice Controlled Excel Assistant

Sara is a **voice-controlled AI assistant for Microsoft Excel** that allows users to perform spreadsheet operations using **natural language commands**.

Instead of manually operating Excel, users can simply speak or type commands such as:

> "Hey Sara, create a sales report and calculate total revenue."

Sara interprets the command and automatically performs the required Excel operations.

---

# 🚀 Key Features

- 🎤 Voice commands for Excel
- 🌍 Multilingual support
- 🧠 Hybrid LLM reasoning (Groq + Ollama)
- 🧩 WAT Architecture (Workflows–Agents–Tools)
- 🧠 Context Memory for multi-step commands
- 📊 Automatic Excel operations
- 🖥 Desktop UI with microphone support

---

# 🧠 System Architecture

Sara follows an AI agent architecture based on the **WAT framework with Memory support**.


User Voice
     ↓
Speech Recognition
     ↓
Command Router
     ↓
Intent Understanding
     ↓
Memory Lookup
     ↓
Workflow Selection
     ↓
Tool Execution
     ↓
Excel Operation
     ↓
User Feedback

---

# 📚 Documentation

Detailed documentation is available in the **docs folder**:

- System Architecture → `docs/architecture.md`
- WAT Framework → `docs/wat_framework.md`
- Execution Pipeline → `docs/pipeline.md`

---

# 🧪 Example Commands

Example user commands:


Create a new Excel file named sales.xlsx
Add a column called Revenue
Calculate the total revenue
Create a sales chart


Sara automatically converts these commands into structured Excel operations.

---

# 🛠 Technology Stack

Backend

- Python
- Flask API
- OpenPyXL

AI

- Groq LLM
- Ollama Local LLM

Speech

- Whisper / Google STT

Desktop UI

- PyQt

---

# 📂 Project Structure


Sara/

README.md
AI_CONTEXT.md

docs/
architecture.md
wat_framework.md
pipeline.md

backend/
desktop/
workflows/
tools/
tests/


---

# 🎯 Project Goal

Sara aims to demonstrate how **AI agents can automate traditional software tools** through natural language interaction.

The system converts **human instructions → structured workflows → deterministic Excel operations**.

---

# ⚙️ Installation

## Prerequisites

- **Python 3.11+** (tested up to 3.13)
- A virtual environment is recommended

## Setup

```bash
# Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt
```

## PyAudio on Windows

PyAudio requires the PortAudio native library. On Windows the standard `pip install pyaudio` may fail with a build error. Use one of these approaches:

### Option A — pipwin (easiest)

```bash
pip install pipwin
pipwin install pyaudio
```

### Option B — Pre-built wheel from Christoph Gohlke's archive

Download the `.whl` matching your Python version from
<https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio> and install it:

```bash
pip install PyAudio‑0.2.14‑cp311‑cp311‑win_amd64.whl
```

### Option C — conda

```bash
conda install pyaudio
```

## Known Issue: PyQt6 + onnxruntime DLL conflict (Windows)

On Windows, loading PyQt6 before onnxruntime causes a DLL initialization failure. The desktop UI (`desktop/sara_ui.py`) already handles this by pre-importing `onnxruntime` before PyQt6. If you encounter `ImportError: DLL load failed while importing onnxruntime_pybind11_state`, ensure onnxruntime is imported first in your entry point.