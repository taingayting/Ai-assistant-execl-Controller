"""Sara on the web: talk to a spreadsheet in your browser.

This page runs Sara's real backend (backend/): the same rule parser, agent,
planner, validator and Excel controller as the Windows app. The difference:
instead of Microsoft Excel on your computer, Sara edits an .xlsx file on the
server, shows it here, and lets you download it.

Every visitor gets their own file. Files are deleted after 30 minutes.
"""
import glob
import html
import json
import os
import re
import sys
import tempfile
import threading
import time
import uuid

import streamlit as st
import streamlit.components.v1 as components

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND = os.path.join(ROOT, "backend")
sys.path.insert(0, BACKEND)
# The web edition edits files only. It must never open or drive a desktop Excel.
sys.modules["win32com"] = None
sys.modules["win32com.client"] = None
sys.modules["pythoncom"] = None

# Web settings for Sara's backend: no desktop voice, no local Ollama server.
os.environ.setdefault("ENABLE_TTS", "false")
os.environ.setdefault("USE_OLLAMA", "false")
os.environ.setdefault("WHISPER_MODEL", "base.en")
try:  # optional: a Groq key in Streamlit secrets lets Sara understand free-form sentences
    if "GROQ_API_KEY" in st.secrets:
        os.environ.setdefault("GROQ_API_KEY", st.secrets["GROQ_API_KEY"])
except Exception:
    pass

st.set_page_config(page_title="Sara · talk to your spreadsheet", page_icon="🎙️", layout="wide")

KEEP_MINUTES = 30
EXAMPLES = [
    "add row Dara 88",
    "sum B2:B5 into D2",
    "average B2:B5 into D3",
    "bold row 1",
    "fill row 1 with yellow",
    "sort by B descending",
    "add column Grade",
    "delete row 3",
    "undo",
]
STARTER = [
    ["Name", "Score"],
    ["Sokha", 92],
    ["Vanna", 75],
    ["Lina", 81],
    ["Rith", 67],
]


@st.cache_resource(show_spinner="Starting Sara...")
def load_sara():
    """Import Sara's backend once per server."""
    import app as sara_app  # backend/app.py: Flask API + the Agent
    import excel_controller as ec
    return sara_app, ec


@st.cache_resource
def big_lock():
    # Sara's Excel controller keeps one open workbook at a time, so one command runs at a time.
    return threading.Lock()


@st.cache_resource(show_spinner="Loading the speech model (first time only, about 1 minute)...")
def load_ears():
    import speech_recognition as sr
    sr._load_whisper_model()
    return sr


def clean_old_files(files_dir):
    cutoff = time.time() - KEEP_MINUTES * 60
    for p in glob.glob(os.path.join(files_dir, "web-*.xlsx*")):
        try:
            if os.path.getmtime(p) < cutoff:
                os.remove(p)
        except OSError:
            pass


NUMBER_WORDS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty".split())}


def tidy_speech(text):
    """Small clean-ups for spoken sentences, so the rule parser understands them without an AI model.

    "Sum B2 to B5 into D2." -> "Sum B2:B5 into D2"    "Bold row one." -> "Bold row 1"    "Adro Dara 88" -> "add row Dara 88"
    """
    t = text.strip().rstrip(".!?").strip()
    t = re.sub(r"^\s*(add a row|add roll|at row|adro)\s*", "add row ", t, flags=re.I)
    t = re.sub(r"\b([A-Za-z]{1,2}\d{1,5})\s+(?:to|through|till|until|-)\s+([A-Za-z]{1,2}\d{1,5})\b",
               lambda m: f"{m.group(1).upper()}:{m.group(2).upper()}", t)
    t = re.sub(r"\b(row|rows|column|size)\s+(" + "|".join(NUMBER_WORDS) + r")\b",
               lambda m: f"{m.group(1)} {NUMBER_WORDS[m.group(2).lower()]}", t, flags=re.I)
    t = re.sub(r"\b(into|in|to|cell)\s+([a-z]{1,2}\d{1,5})\b", lambda m: f"{m.group(1)} {m.group(2).upper()}", t, flags=re.I)
    return t


def run_command(text, confirmed_action=None):
    """Understand *text* with Sara's agent, then run it through Sara's /api/execute."""
    sara_app, ec = load_sara()
    fn = st.session_state.file
    client = sara_app.app.test_client()
    with big_lock():
        # Sara remembers the active file between commands; point that memory at this visitor's file.
        try:
            sara_app.memory.set_active_file(fn)
            sara_app.memory.set_active_sheet("Sheet1")
        except Exception:
            pass
        if confirmed_action is not None:
            out = client.post("/api/execute", json={"action": confirmed_action, "filename": fn, "confirmed": True}).get_json()
            action = confirmed_action
        else:
            parsed = sara_app.agent.handle_text(text, filename=fn, auto_execute=False)
            action = parsed.action or {}
            if parsed.error or not action or action.get("type") == "unknown":
                tips = parsed.suggestions or []
                msg = parsed.speech_text or "Sorry, I did not understand that. Try one of the example commands."
                return {"ok": False, "reply": msg, "tips": tips[:3], "action": action}
            out = client.post("/api/execute", json={"action": action, "filename": fn}).get_json() or {}
        ec._flush_session(os.path.join(sara_app.FILES_DIR, fn))
    if out.get("requires_confirmation"):
        return {"ok": True, "confirm": out.get("message") or "Are you sure?", "action": action}
    reply = out.get("speech_text") or out.get("message") or ("Done." if out.get("success") else out.get("error") or "Something went wrong.")
    return {"ok": bool(out.get("success")), "reply": reply, "action": action}


def new_file():
    sara_app, _ = load_sara()
    clean_old_files(sara_app.FILES_DIR)
    st.session_state.file = f"web-{uuid.uuid4().hex[:10]}.xlsx"
    st.session_state.log = []
    st.session_state.pending = None
    # The practice sheet is made directly, so every visitor starts with the same clean file.
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    for row in STARTER:
        ws.append(row)
    wb.save(os.path.join(sara_app.FILES_DIR, st.session_state.file))


def sheet_table():
    """Read the visitor's file and return rows for display."""
    import openpyxl
    sara_app, _ = load_sara()
    path = os.path.join(sara_app.FILES_DIR, st.session_state.file)
    if not os.path.exists(path):
        return None, None
    wb = openpyxl.load_workbook(path)
    ws = wb.active
    rows = []
    for r in ws.iter_rows(min_row=1, max_row=max(ws.max_row, 1), max_col=max(ws.max_column, 1)):
        rows.append(r)
    return ws, rows


def render_sheet(ws, rows):
    cols = max((len(r) for r in rows), default=1)
    cols = max(cols, 4)
    letters = [openpyxl_letter(i + 1) for i in range(cols)]
    out = ['<div class="sheet"><table><thead><tr><th></th>' + "".join(f"<th>{l}</th>" for l in letters) + "</tr></thead><tbody>"]
    n = max(len(rows), 8)
    for i in range(n):
        cells = rows[i] if i < len(rows) else []
        tds = []
        for j in range(cols):
            c = cells[j] if j < len(cells) else None
            v = "" if c is None or c.value is None else str(c.value)
            style = []
            if c is not None:
                if c.font is not None and c.font.b:
                    style.append("font-weight:700")
                if c.font is not None and c.font.i:
                    style.append("font-style:italic")
                fill = c.fill.fgColor.rgb if c.fill is not None and c.fill.fill_type == "solid" else None
                if isinstance(fill, str) and len(fill) >= 6 and fill not in ("00000000",):
                    style.append(f"background:#{fill[-6:]};color:#1d1815")
                col = c.font.color.rgb if c.font is not None and c.font.color is not None else None
                if isinstance(col, str) and len(col) >= 6 and col not in ("FF000000", "00000000"):
                    style.append(f"color:#{col[-6:]}")
                if c.alignment is not None and c.alignment.horizontal in ("center", "right", "left"):
                    style.append(f"text-align:{c.alignment.horizontal}")
            cls = ' class="f"' if v.startswith("=") else ""
            tds.append(f'<td{cls} style="{";".join(style)}">{html.escape(v)}</td>')
        out.append(f"<tr><th>{i + 1}</th>{''.join(tds)}</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def openpyxl_letter(n):
    from openpyxl.utils import get_column_letter
    return get_column_letter(n)


def say_aloud(text):
    """Sara answers with the browser's voice."""
    components.html(
        "<script>try{const u=new SpeechSynthesisUtterance(%s);u.lang='en-US';u.rate=1.05;"
        "window.parent.speechSynthesis.cancel();window.parent.speechSynthesis.speak(u);}catch(e){}</script>" % json.dumps(text),
        height=0,
    )


def handle(text, source):
    text = (text or "").strip()
    if not text:
        return
    res = run_command(text)
    st.session_state.log.insert(0, {"you": text, "source": source, **res})
    if res.get("confirm"):
        st.session_state.pending = {"text": text, "action": res["action"], "msg": res["confirm"]}
    elif st.session_state.get("speak", True):
        st.session_state.to_say = res.get("reply")


# ---------------------------------------------------------------- page

st.markdown(
    """
<style>
.block-container { padding-top: 2rem; max-width: 1200px; }
h1 { font-family: Georgia, serif; font-weight: 400; }
.sub { color: #bcaa98; font-size: 1.05rem; margin-top: -0.4rem; }
.sheet { overflow-x: auto; border-radius: 14px; border: 1px solid #463930; }
.sheet table { border-collapse: collapse; width: 100%; font-size: 0.95rem; background: #fffaf3; color: #1d1815; }
.sheet th { background: #efe2d0; color: #6b5646; font-weight: 600; padding: 6px 10px; border: 1px solid #e0d2bf; min-width: 34px; }
.sheet td { padding: 6px 10px; border: 1px solid #eadfcf; min-width: 90px; height: 32px; white-space: nowrap; }
.sheet td.f { color: #a96f3a; font-family: Consolas, monospace; font-size: 0.88rem; }
.bubble { border-radius: 14px; padding: 10px 14px; margin: 6px 0; }
.you { background: #362c26; }
.sara { background: #2a231f; border-left: 3px solid #d59b5f; }
.small { color: #bcaa98; font-size: 0.85rem; }
</style>
""",
    unsafe_allow_html=True,
)

if "file" not in st.session_state:
    with st.spinner("Making your practice sheet..."):
        new_file()

st.title("🎙️ Sara: talk to your spreadsheet")
st.markdown(
    '<p class="sub">Say or type what you want, like <b>“add row Dara 88”</b> or <b>“sum B2:B5 into D2”</b>. '
    "Sara changes the file and tells you what she did. This is Sara's real code, running in your browser.</p>",
    unsafe_allow_html=True,
)

left, right = st.columns([2, 3], gap="large")

with left:
    st.subheader("1. Tell Sara")
    audio = st.audio_input("🎤 Press the mic, say a command, then press stop")
    if audio is not None:
        digest = hash(audio.getvalue())
        if st.session_state.get("last_audio") != digest:
            st.session_state.last_audio = digest
            with st.spinner("Listening..."):
                sr = load_ears()
                tmp = os.path.join(tempfile.gettempdir(), f"sara-{uuid.uuid4().hex}.wav")
                with open(tmp, "wb") as fh:
                    fh.write(audio.getvalue())
                try:
                    heard = sr.transcribe_audio_file(tmp, language="en")
                finally:
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
            if not heard or heard.startswith("[transcription_error]"):
                st.warning("I could not hear a command. Please try again, a little closer to the mic.")
            else:
                handle(tidy_speech(heard), "voice")

    with st.form("typed", clear_on_submit=True, border=False):
        typed = st.text_input("⌨️ Or type a command", placeholder="add row Dara 88")
        if st.form_submit_button("Send", type="primary"):
            handle(typed, "typed")

    st.caption("Try one:")
    ex_cols = st.columns(3)
    for i, ex in enumerate(EXAMPLES):
        if ex_cols[i % 3].button(ex, key=f"ex{i}", use_container_width=True):
            handle(ex, "example")

    p = st.session_state.get("pending")
    if p:
        st.warning(f"⚠️ {p['msg']}")
        y, n = st.columns(2)
        if y.button("Yes, do it", type="primary", use_container_width=True):
            res = run_command(p["text"], confirmed_action=p["action"])
            st.session_state.log.insert(0, {"you": "yes", "source": "confirm", **res})
            st.session_state.pending = None
            st.session_state.to_say = res.get("reply")
            st.rerun()
        if n.button("No, cancel", use_container_width=True):
            st.session_state.pending = None
            st.session_state.log.insert(0, {"you": "no", "source": "confirm", "ok": True, "reply": "Okay, I did not change anything."})
            st.rerun()

    st.session_state.speak = st.toggle("🔊 Sara speaks her answers", value=st.session_state.get("speak", True))

    st.subheader("2. What Sara did")
    if not st.session_state.log:
        st.markdown('<p class="small">Your conversation with Sara appears here.</p>', unsafe_allow_html=True)
    for item in st.session_state.log[:8]:
        icon = {"voice": "🎤", "typed": "⌨️", "example": "👆", "confirm": "✅"}.get(item.get("source"), "")
        st.markdown(f'<div class="bubble you">{icon} <b>You:</b> {html.escape(item["you"])}</div>', unsafe_allow_html=True)
        if item.get("confirm"):
            st.markdown(f'<div class="bubble sara"><b>Sara:</b> {html.escape(item["confirm"])}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bubble sara"><b>Sara:</b> {html.escape(item.get("reply") or "")}</div>', unsafe_allow_html=True)
        if item.get("tips"):
            st.markdown('<p class="small">Did you mean: ' + " · ".join(html.escape(str(t)) for t in item["tips"]) + "</p>", unsafe_allow_html=True)

with right:
    st.subheader("3. Your sheet")
    ws, rows = sheet_table()
    if rows is None:
        st.info("Your sheet is not ready. Press “Start over”.")
    else:
        st.markdown(render_sheet(ws, rows), unsafe_allow_html=True)
        st.markdown('<p class="small">Formulas (orange) calculate when you open the file in Excel or Google Sheets.</p>', unsafe_allow_html=True)
    sara_app, _ = load_sara()
    path = os.path.join(sara_app.FILES_DIR, st.session_state.file)
    b1, b2 = st.columns(2)
    if os.path.exists(path):
        with open(path, "rb") as fh:
            b1.download_button("⬇️ Download my .xlsx", fh.read(), file_name="sara-sheet.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
    if b2.button("↺ Start over", use_container_width=True):
        new_file()
        st.rerun()

    with st.expander("What can I say?"):
        st.markdown(
            """
- **Rows and columns:** `add row Dara 88` · `add column Grade` · `insert row 2` · `delete row 3` · `delete column C`
- **Cells:** `write cell C1 Grade` · `delete cell A2`
- **Formulas:** `sum B2:B5 into D2` · `average B2:B5 into D3`
- **Format:** `bold row 1` · `italic A2` · `center column B` · `fill row 1 with yellow` · `make A3 red` · `set font size row 1 14`
- **Sort and find:** `sort by B descending` · `replace Lina with Lisa`
- **Mistakes:** `undo`
"""
        )
    with st.expander("How this works"):
        st.markdown(
            """
1. **Listen:** your voice is turned into text with faster-whisper, the same speech model as the Windows app.
2. **Understand:** Sara's rule parser turns the sentence into an action (for example `add_row ["Dara", 88]`).
3. **Check:** the planner and validator make sure the action is safe. Risky actions ask you first.
4. **Do it:** the Excel controller changes the file, with a snapshot so `undo` always works.

The **Windows app** does the same thing in real Microsoft Excel, with the wake word “Hey Jarvis”.
[Download Sara for Windows](https://github.com/taingayting/Ai-assistant-execl-Controller/releases/latest) ·
[Read the code](https://github.com/taingayting/Ai-assistant-execl-Controller) ·
Made by [Taing Ayting](https://taingayting.github.io/portfolio/)
"""
        )
    st.caption(f"Your file is private to this visit and is deleted after {KEEP_MINUTES} minutes.")

msg = st.session_state.pop("to_say", None)
if msg and st.session_state.get("speak", True):
    say_aloud(msg)
