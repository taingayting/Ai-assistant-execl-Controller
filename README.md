# Sara: Control Excel With Your Voice

**Say what you want. Sara changes your real Excel file and tells you what she did.**

👉 **Download for Windows:** [Sara 1.0 (latest release)](https://github.com/taingayting/Ai-assistant-execl-Controller/releases/latest)
No Python needed. Unzip it and double-click **Sara.exe**.

Made by [Taing Ayting](https://taingayting.github.io/portfolio/). This was my final-year project for my Computer Science degree at Shenyang Ligong University.

---

## Why I built it

Excel is powerful, but it is hard to remember every formula, every menu and every format. I often knew **what** I wanted ("add the total here", "make this row bold") but not **how** to do it. Many people at work feel the same, and they lose time searching for help.

So I built Sara. You say **"Hey Jarvis"**, then a normal sentence like *"sum B2 to B6 into C3"*, and Sara does it in Excel for you.

## Why it matters

- **Excel for everyone.** You do not need to remember formulas or menus. You only need to say what you want.
- **Faster work.** Common tasks take one sentence instead of many clicks.
- **Accessible.** It helps people who find the mouse and keyboard hard to use.
- **Safe.** Sara checks every action before it runs, asks before deleting, and can **undo** any change.
- **Works offline.** Speech recognition runs on your own computer. An AI model is optional.

## See it work

These are real recordings: I speak, and Sara changes the real Excel file. The demo videos are at the end of this page.

| "Add row John 90" | "Sum B2 B6 into C3" | "Undo" |
| :---: | :---: | :---: |
| Adds a new row with the name and score | Writes a SUM formula into C3 | Takes back the last change |

## How to use it

1. Download **Sara-Windows.zip** from the [latest release](https://github.com/taingayting/Ai-assistant-execl-Controller/releases/latest) and unzip it.
2. Open Microsoft Excel.
3. Double-click **Sara.exe** in the Sara folder. Windows may warn you because the app is new: click **More info → Run anyway**.
4. Allow the microphone. Say **"Hey Jarvis"**, wait for Sara's window, then say your command.
5. Sara does it and answers you. Say **"undo"** if you change your mind.

The first start needs internet for about a minute, to download the speech model.
You need Windows 10 or 11, Microsoft Excel and a microphone.

## What you can say

### Files and workbooks
| What | Say |
| :--- | :--- |
| Create a file | `create excel sales.xlsx` |
| Open Excel | `open excel` |
| Open a file | `open file report.xlsx` |
| Rename the file | `rename file to report_2026.xlsx` |
| Save | `save` |
| Close Excel | `close excel` |

### Rows and columns
| What | Say |
| :--- | :--- |
| Add a row of data | `add row John 90` |
| Insert an empty row | `insert row 1` |
| Add column headers | `add column Name Score Age` |
| Insert an empty column | `insert column B` |
| Delete a row or column | `delete row 5` · `delete column C` |

### Cells
| What | Say |
| :--- | :--- |
| Write in a cell | `write cell C1 Name` · `set cell D1 Test` |
| Clear a cell or range | `delete cell A2` · `delete range A1:B3` |

### Formulas
| What | Say |
| :--- | :--- |
| Total | `sum B2:B5 into C5` |
| Average | `average B2:B5 into C7` |

### Formatting
| What | Say |
| :--- | :--- |
| Bold or italic | `bold A1` · `bold row 6` · `italic column C` |
| Alignment | `center B2` · `align row 3 left` · `align A2:B2 right` |
| Font size | `set font size row 3 16` · `increase font size A1` |
| Text colour | `make A3 orange` · `color row 3 blue` |
| Fill colour | `fill row 2 with red` · `fill A4 pink` |
| Column width | `set column width A 5` · `auto fit all columns` |
| Merge cells | `merge cells A1:C1` · `unmerge cells A1:C1` |

### Sheets, sorting and filters
| What | Say |
| :--- | :--- |
| Freeze the top row | `freeze header` · `freeze panes` · `unfreeze` |
| Find and replace | `replace Tom with John` |
| Sort | `sort by B ascending` · `sort by A descending` |
| Filter | `filter column B = 90` · `clear filter` |
| Sheets | `create sheet Report` · `rename sheet Sales to Market` · `switch to Buy` · `delete sheet Report` |
| Undo | `undo` |

## How it works

```
 🎤 "Hey Jarvis ... sum B2 to B6 into C3"
        │
        ▼
 1. Listen        wake word (openWakeWord) → speech to text (faster-whisper, on your computer)
        │
        ▼
 2. Understand    clean the text → fast rules first → AI model only if the rules are not sure
        │
        ▼
 3. Plan & check  Agent → Planner → Router → Validator (is it safe? does the cell exist?)
        │
        ▼
 4. Do it         workflow → tool → Excel (openpyxl + Excel COM), with a backup for undo
        │
        ▼
 5. Answer        Sara's window shows the result, and her voice says what she did
```

**Rules first, AI second.** Most commands, like "add row" or "bold A1", are understood by fast rules that need no AI. Only unclear sentences go to an AI model: Ollama on your computer, or Groq in the cloud. This keeps Sara fast, free and private for daily use.

## How Sara keeps your file safe

| Risk | What Sara does |
| :--- | :--- |
| Deleting the wrong thing | Asks you to confirm before risky changes, like deleting |
| A mistake | Saves a snapshot before each change, so **undo** works |
| A broken file | Writes to a temporary file first, then replaces the real file in one step |
| Dangerous cell text | Text starting with `=`, `+`, `-` or `@` is made safe, so it cannot run as a formula |
| Strange AI answers | Checks the AI's answer with strict rules (Pydantic) and tries several ways to read it |
| Tricky voice input | Cleans the text before it reaches the AI (prompt-injection protection) |
| Cloud AI keeps failing | After 3 Groq failures in a row, Sara stops calling Groq for 60 seconds |
| Errors | Explains problems in simple words instead of technical messages |

## One hard decision I made

**I did not send every sentence to an AI.** An AI model understands more sentences, but it is slower, sometimes wrong, and can cost money. I decided to try simple rules first and use AI only when the rules are not sure. Sara became faster and more predictable, and it works without internet.

## Tools I used

| Part | Tool |
| :--- | :--- |
| Language | Python |
| Wake word | openWakeWord ("Hey Jarvis") |
| Speech to text | faster-whisper (offline), Google Speech-to-Text (optional) |
| Understanding | Rule-based parser + Ollama (local) or Groq (cloud) |
| Checking AI output | Pydantic v2 |
| Backend | Flask (REST API on port 5000) |
| Excel | openpyxl and Microsoft Excel COM automation |
| Desktop window | PyQt6 (small floating window) |
| Voice answers | pyttsx3 (text to speech) |
| Memory | SQLite (history of commands and cells) |
| Tests | pytest, GitHub Actions |
| Windows app | PyInstaller |

## Run it from the code (for developers)

You need Python 3.10 or newer on Windows.

```
python -m venv .venv
.venv\Scripts\activate
pip install -r backend/requirements.txt
pip install faster-whisper
```

Optional: create `backend/.env` for the AI helpers and settings:

```ini
GROQ_API_KEY=your-free-groq-key   # for unclear sentences (cloud)
GROQ_FIRST=false                  # true = try Groq before Ollama
OLLAMA_MODEL=qwen2.5:3b           # local AI model, if you use Ollama
WHISPER_MODEL=base.en             # speech model size
ENABLE_TTS=true                   # Sara speaks her answers
```

Then start Sara:

- **Desktop mode:** double-click `start_sara.bat`. It starts the backend and the floating window.
- **Background mode (no window):** `python backend/run_sara.py`

## Project files

```
backend/
  app.py                 Flask API: understand, confirm and run commands
  agent.py               The coordinator: understand → plan → check → do
  llm_router.py          Rules first, then Ollama, then Groq
  llm_parser.py          Reads and checks the AI's answer
  planner.py             Turns a command into steps, finds risky ones
  excel_controller.py    Changes the Excel file, safe save, undo snapshots
  speech_recognition.py  Speech to text
  pipeline_sanitizer.py  Makes input text and cell values safe
  error_handler.py       Simple error messages
  memory.py              Command history (SQLite)
  tts_engine.py          Sara's voice
  run_sara.py            Background mode with the wake word
  tools/                 Tool registry: links each action to its Excel code
  tests/                 Automated tests
desktop/sara_ui.py       The floating window
workflows/               Step-by-step recipes for bigger tasks
start_sara.bat           One-click start
```

## Demo videos

https://github.com/user-attachments/assets/9d777456-4fde-45dc-a015-4c22385d1d13

https://github.com/user-attachments/assets/4b3bce16-fba1-47bf-a48d-e3557e168335

https://github.com/user-attachments/assets/fba63ba9-2afe-4382-a29d-c3a218f17f84

https://github.com/user-attachments/assets/9f7bccf1-fed0-4304-a20a-6a7415abae12

https://github.com/user-attachments/assets/73a54352-c8f3-4e80-9858-312485f53944

https://github.com/user-attachments/assets/9af9d692-1782-480a-97dd-338097750431

https://github.com/user-attachments/assets/feba4136-efef-401e-973c-19979ca28c1e

https://github.com/user-attachments/assets/3f0b5e7d-d675-43e9-8315-838d465f38b6

https://github.com/user-attachments/assets/52862c40-8615-4dd1-ae06-5c39284a3f9c

https://github.com/user-attachments/assets/19f4604a-baaa-41d6-b3c7-8bd982871ade

https://github.com/user-attachments/assets/4bdf7068-377c-4fad-90a1-f76b39bf34e4

https://github.com/user-attachments/assets/9dad2b9d-cf60-4a8f-87e3-eebfbabd949c

https://github.com/user-attachments/assets/32023b66-f345-4990-83bf-c7447346bea7

https://github.com/user-attachments/assets/1703c692-115d-4098-81a8-f495e221f5ba

https://github.com/user-attachments/assets/68391cac-f624-49c2-b1e0-8bfb9a54eff6

https://github.com/user-attachments/assets/a5e3769b-039a-4e04-a2a3-993dca1db186

https://github.com/user-attachments/assets/97d6219c-e7c1-40d4-a36f-25bded1c5e3d

https://github.com/user-attachments/assets/e4b4e186-e842-40f4-af51-a15fe434b4ef

https://github.com/user-attachments/assets/7dd30c54-536d-4571-8281-5cf35557d3bb

https://github.com/user-attachments/assets/3bddf5ed-4e20-4655-aefe-15bfcd3018c6

https://github.com/user-attachments/assets/13186a80-4353-4255-bd56-19a5f0670c4d

https://github.com/user-attachments/assets/3c57cf3b-5385-425a-9edd-ed08cdff6dd4

https://github.com/user-attachments/assets/66183142-a3b0-4086-bff8-cac74dfdf2c2

https://github.com/user-attachments/assets/5a18e03c-b2a0-4fba-95c8-20b213674961

https://github.com/user-attachments/assets/86abab57-71eb-4e3c-9071-17036969acfb

https://github.com/user-attachments/assets/eb17232c-0b8b-44d5-9733-c2e1a90308e1

https://github.com/user-attachments/assets/c3304061-2cd6-494a-a325-8b8ff3a834bf

https://github.com/user-attachments/assets/ccdf93be-86c5-4c7a-bebc-8fec9a76d0f1

https://github.com/user-attachments/assets/a0db0154-7517-4151-a6ea-96bb6155fd1e

https://github.com/user-attachments/assets/00175caa-9739-4230-8a4c-c5fb6719cf2d

https://github.com/user-attachments/assets/dbb54446-4e08-42ee-9165-01bc6c516dad

https://github.com/user-attachments/assets/2676017c-0976-45f2-bad5-5ca76f3f8d5f

https://github.com/user-attachments/assets/06647bbf-6975-44fd-bb64-7291fcfca131

https://github.com/user-attachments/assets/4d4ec856-336f-41a4-b461-d45b93c79640

https://github.com/user-attachments/assets/5172876e-17a7-4d79-84f1-463b472156bc

https://github.com/user-attachments/assets/a6d26507-cdd9-4cbf-95f9-40e7059d3af1

https://github.com/user-attachments/assets/0dd8ec19-4a34-444f-8f0d-5a13e0f347b8

https://github.com/user-attachments/assets/477f18d6-4e04-4682-b214-28591e10133d

https://github.com/user-attachments/assets/91a64b75-1b67-4760-a531-11970d2630dd

https://github.com/user-attachments/assets/28a774b4-23d5-4c34-b6a0-fd962a0d83d0

https://github.com/user-attachments/assets/2783cb9b-481a-4865-95cc-b66f8d894b87

https://github.com/user-attachments/assets/2fa6440e-2662-46bf-ba51-5f4e48c28276

https://github.com/user-attachments/assets/4dcb35cb-0fd3-46df-8b36-76da44585dfe

https://github.com/user-attachments/assets/b9d7cd30-fb0e-48a1-8db1-3f472d3eaa84

https://github.com/user-attachments/assets/f39cf9e0-23dc-4627-91ca-041906e0940f

https://github.com/user-attachments/assets/befa3391-8815-4b58-b68d-7f1fed4d3eae

https://github.com/user-attachments/assets/f5a1c51d-ba5b-4dda-ab63-b22f9ac67e16

https://github.com/user-attachments/assets/04967559-7ba3-4544-8cf6-82282c02b4ba

https://github.com/user-attachments/assets/ac07ec68-04af-4be7-b1ca-3e1d9d291750

https://github.com/user-attachments/assets/e22ce63f-6cb9-4221-b93b-af6fd9480beb

https://github.com/user-attachments/assets/09c9dbfc-db02-4634-94f9-2ce5e80ffda4

https://github.com/user-attachments/assets/c3ede2be-b322-4f63-a6e6-8ec34c0b81b6

https://github.com/user-attachments/assets/beec8ac0-2347-46eb-ad9b-805011f57a25

https://github.com/user-attachments/assets/485df6b1-1186-4653-8b1b-43ad444fd87d

https://github.com/user-attachments/assets/0ff2a192-a6ef-426a-842d-aa4e398ede80

https://github.com/user-attachments/assets/c45efd01-5662-43ad-b127-1d1bbf0479fc
