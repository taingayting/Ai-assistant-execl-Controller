from flask import Flask, request, jsonify, send_from_directory, g
import os
import logging
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from flask_cors import CORS

# Load local .env (if present) so `llm_parser` reads GROQ_* env vars at import time.
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
if os.path.exists(_env_path):
    load_dotenv(_env_path, override=True)

from llm_parser import parse_with_llm
from llm_parser import generate_suggestions
from speech_recognition import transcribe_audio_file
from excel_controller import execute_action
from excel_controller import ensure_workbook
from memory import memory
from metrics import metrics as _metrics
from agent import Agent
import workflow_router
import tts_engine
import llm_router
import shutil
import uuid
from threading import Lock
import json
import time

# --------------- Structured Logging ---------------

class _StructuredFormatter(logging.Formatter):
    """Emit JSON-lines log records for easy downstream analysis."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Attach request_id from Flask ``g`` when available
        try:
            rid = getattr(g, 'request_id', None)
            if rid:
                entry["request_id"] = rid
        except RuntimeError:
            pass  # outside request context
        if record.exc_info and record.exc_info[0]:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False)


_handler = logging.StreamHandler()
_handler.setFormatter(_StructuredFormatter(datefmt='%Y-%m-%dT%H:%M:%S'))
logging.basicConfig(level=logging.DEBUG, handlers=[_handler])
log = logging.getLogger('app')

# Backup settings
MAX_BACKUPS = int(os.environ.get('MAX_BACKUPS', '5'))
MAX_BACKUP_BYTES = int(os.environ.get('MAX_BACKUP_BYTES', str(50 * 1024 * 1024)))  # 50 MB default
CONF_TTL_SECONDS = int(os.environ.get('CONF_TTL_SECONDS', '3600'))  # 1 hour

# Validation limits
MAX_TEXT_LENGTH = 2000
ALLOWED_EXTENSIONS = {'.xlsx', '.xls', '.csv'}




# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.normpath(os.path.join(BASE_DIR, '..', 'frontend'))
FILES_DIR = os.path.join(BASE_DIR, 'files')
os.makedirs(FILES_DIR, exist_ok=True)

# Confirmation storage on disk
CONF_DIR = os.path.join(BASE_DIR, 'confirmations')
os.makedirs(CONF_DIR, exist_ok=True)

# In-memory pending confirmations: confirmation_id -> {action, filepath, backup}
CONFIRMATIONS = {}
CONF_LOCK = Lock()

# Flask app serving API and frontend
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path='')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB upload limit
CORS(app)

# --------------- Request-level instrumentation ---------------

@app.before_request
def _before_request():
    g.request_id = uuid.uuid4().hex[:12]
    g.start_time = time.perf_counter()


@app.after_request
def _after_request_logging(response):
    """Log a structured summary for every request and record it in metrics."""
    latency_ms = (time.perf_counter() - getattr(g, 'start_time', time.perf_counter())) * 1000
    log.info(
        'request  rid=%s  method=%s  path=%s  status=%d  latency_ms=%.1f  user_command=%s  llm_provider=%s',
        getattr(g, 'request_id', '?'),
        request.method,
        request.path,
        response.status_code,
        latency_ms,
        getattr(g, 'user_command', None),
        getattr(g, 'llm_provider', None),
    )
    _metrics.record_request(
        endpoint=request.path,
        method=request.method,
        status=response.status_code,
        latency_ms=round(latency_ms, 2),
        request_id=getattr(g, 'request_id', None),
    )
    return response

# Central agent instance (stateless — state lives in the memory singleton)
agent = Agent(files_dir=FILES_DIR)

# Initialise TTS engine (no-op when ENABLE_TTS is not true)
tts_engine.initialize_engine()


@app.after_request
def _tts_after_request(response):
    """Speak *speech_text* from JSON API responses in a background thread.

    /api/execute is excluded: the desktop popup owns TTS for execution results
    via its own TTSWorker.  Speaking here too would share the pyttsx3 engine
    across two threads and cause the same text to be spoken twice.
    """
    if request.path == '/api/execute':
        return response
    if response.content_type and 'application/json' in response.content_type:
        try:
            data = response.get_json(silent=True)
            if isinstance(data, dict) and data.get('speech_text'):
                tts_engine.speak(data['speech_text'])
        except Exception:
            pass
    return response


@app.route('/')
def index():
    # Serve the frontend index.html
    return app.send_static_file('index.html')


@app.route('/api/parse', methods=['POST'])
def parse_endpoint():
    data = request.get_json() or {}
    text = data.get('text', '')

    # --- input validation ---
    if not isinstance(text, str) or not text.strip():
        log.warning('parse: empty or non-string text')
        return jsonify({"success": False, "error": "text is required and must be a non-empty string"}), 400
    if len(text) > MAX_TEXT_LENGTH:
        log.warning('parse: text too long (%d chars)', len(text))
        return jsonify({"success": False, "error": f"text exceeds {MAX_TEXT_LENGTH} characters"}), 400

    log.info('parse: text=%r', text[:120])
    g.user_command = text[:120]
    action = parse_with_llm(text)
    log.info('parse: action=%s', action)
    resp = {"success": True, "action": action}
    resp['cleaned'] = text
    # If unknown action, produce suggestions (LLM-backed if enabled/backend available)
    if not action or action.get('type') == 'unknown' or action.get('action') == 'unknown':
        log.info('parse: unknown action, generating suggestions')
        try:
            resp['suggestions'] = generate_suggestions(text)
        except Exception:
            resp['suggestions'] = []
    return jsonify(resp)


@app.route('/api/execute', methods=['POST'])
def execute_endpoint():
    data = request.get_json() or {}
    action = data.get('action')
    auto_correct = bool(data.get('auto_correct', False))
    # Resolve filename: explicit request param > memory active_file > default
    raw_filename = data.get('filename') or ''
    if not raw_filename.strip():
        raw_filename = memory.get_active_file() or 'data.xlsx'
    filename = secure_filename(raw_filename)
    filepath = os.path.join(FILES_DIR, filename)

    # --- input validation ---
    if not action or not isinstance(action, dict):
        log.warning('execute: missing or invalid action')
        return jsonify({"success": False, "error": "action must be a non-empty JSON object"}), 400
    has_steps = isinstance(action.get('steps'), list) and len(action.get('steps')) > 0
    if not action.get('type') and not has_steps:
        log.warning('execute: action missing type/steps field: %s', action)
        return jsonify({"success": False, "error": "action.type is required (or provide action.steps)"}), 400
    if not filename:
        log.warning('execute: filename resolved to empty after sanitization')
        return jsonify({"success": False, "error": "Invalid filename"}), 400
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        log.warning('execute: disallowed extension %r', ext)
        return jsonify({"success": False, "error": f"File extension '{ext}' not allowed"}), 400

    log.info('execute: action=%s  file=%s  confirmed=%s', action, filename, data.get('confirmed'))
    # Always create a rotating backup checkpoint before executing (if file exists)
    def create_checkpoint(path):
        if not os.path.exists(path):
            return None
        # rotate existing backups: bak.(MAX_BACKUPS-1) -> bak.MAX_BACKUPS
        for i in range(MAX_BACKUPS, 1, -1):
            prev = f"{path}.bak.{i-1}"
            dest = f"{path}.bak.{i}"
            if os.path.exists(prev):
                try:
                    # overwrite destination
                    if os.path.exists(dest):
                        os.remove(dest)
                    shutil.move(prev, dest)
                except Exception:
                    pass
        # copy current file to bak.1
        first = f"{path}.bak.1"
        try:
            if os.path.exists(first):
                os.remove(first)
            shutil.copy2(path, first)
            # also keep a convenience .bak file pointing to latest
            bak_link = f"{path}.bak"
            try:
                if os.path.exists(bak_link):
                    os.remove(bak_link)
                shutil.copy2(first, bak_link)
            except Exception:
                pass
            return first
        except Exception:
            return None

    bak_path = None
    try:
        bak_path = create_checkpoint(filepath)
        _enforce_backup_budget(filepath)
    except Exception:
        bak_path = None

    # Lazily purge stale confirmation files
    try:
        _purge_stale_confirmations()
    except Exception:
        pass

    # If action is destructive, ask for confirmation first unless client passed confirm flag
    destructive = workflow_router.is_destructive(action)
    if destructive and not data.get('confirmed'):
        # produce a contextual message (e.g., rows affected)
        message = 'Are you sure you want to perform this destructive action?'
        try:
            # attempt to enrich message for delete_column
            if action.get('type') == 'delete_column':
                # count rows in column
                wb, ws = ensure_workbook(filepath)
                rows = ws.max_row
                col = str(action.get('column', '')).upper()
                message = f"Are you sure you want to delete column {col}? It contains {rows} rows of data."
        except Exception:
            pass

        # store pending confirmation on disk
        cid = str(uuid.uuid4())
        pending = {"action": action, "filepath": filepath, "backup": bak_path, "ts": time.time()}
        conf_file = os.path.join(CONF_DIR, cid + '.json')
        try:
            with open(conf_file, 'w', encoding='utf-8') as fh:
                json.dump(pending, fh)
        except Exception:
            # fallback to in-memory store if disk write fails
            with CONF_LOCK:
                CONFIRMATIONS[cid] = pending

        return jsonify({"success": True, "requires_confirmation": True, "confirmation_id": cid, "message": message, "speech_text": message})

    # --- Workflow-aware precondition check ---
    wf = workflow_router.route(action)
    if wf:
        snap = memory.snapshot()
        if data.get('filename') and filename:
            snap = dict(snap, active_file=filename)
        pre_errors = workflow_router.check_preconditions(wf, snap, filepath)
        if pre_errors:
            missing_file_ctx = any(('No active file' in err) or ('not found' in err) for err in pre_errors)
            if auto_correct and missing_file_ctx:
                pre_errors = []
            else:
                if missing_file_ctx:
                    return jsonify({
                        "success": False,
                        "error": pre_errors[0],
                        "speech_text": "No Excel file is open. Would you like me to create a new file?",
                        "recovery": {
                            "suggestions": [
                                "Set auto_correct=true to create a file automatically",
                                "Or run create_excel first",
                            ],
                        },
                    }), 400
            if pre_errors:
                return jsonify({"success": False, "error": pre_errors[0]}), 400

    # Execute immediately via the agent
    log.info('execute: running action %s on %s (workflow=%s)',
             action.get('type') if not has_steps else 'steps', filename,
             wf['id'] if wf else None)
    _t_exec = time.perf_counter()
    agent_result = agent.handle_action(
        action,
        filename=filename,
        confirmed=True,
        auto_correct=auto_correct,
    )
    log.info('[LATENCY] execution=%.0fms action=%s',
             (time.perf_counter() - _t_exec) * 1000,
             action.get('type') if not has_steps else 'steps')
    log.info('execute: agent result success=%s', agent_result.success)

    resp = agent_result.to_dict()
    resp['download_url'] = f"/files/{filename}"
    return jsonify(resp)


@app.route('/api/confirm', methods=['POST'])
def confirm_endpoint():
    data = request.get_json() or {}
    cid = data.get('confirmation_id')
    confirm = data.get('confirm', False)
    if not cid or not isinstance(cid, str):
        log.warning('confirm: missing or non-string confirmation_id')
        return jsonify({"success": False, "error": "No confirmation_id provided"}), 400
    # Only allow UUID-shaped ids to prevent path traversal
    import re as _re
    if not _re.fullmatch(r'[0-9a-f\-]{36}', cid):
        log.warning('confirm: invalid confirmation_id format: %r', cid)
        return jsonify({"success": False, "error": "Invalid confirmation_id format"}), 400
    log.info('confirm: cid=%s  confirm=%s', cid, confirm)
    # try to load pending confirmation from disk, then fallback to in-memory
    conf_file = os.path.join(CONF_DIR, cid + '.json')
    pending = None
    if os.path.exists(conf_file):
        try:
            with open(conf_file, 'r', encoding='utf-8') as fh:
                pending = json.load(fh)
            # remove file after loading
            try:
                os.remove(conf_file)
            except Exception:
                pass
        except Exception:
            pending = None
    else:
        with CONF_LOCK:
            pending = CONFIRMATIONS.pop(cid, None)

    if not pending:
        return jsonify({"success": False, "error": "Unknown or expired confirmation"}), 404
    if not confirm:
        log.info('confirm: user canceled cid=%s', cid)
        return jsonify({"success": True, "canceled": True, "message": "Action canceled"})
    # perform the stored action via the agent
    action = pending.get('action')
    filepath = pending.get('filepath')
    filename = os.path.basename(filepath)
    log.info('confirm: executing deferred action=%s on %s', action, filepath)
    agent_result = agent.handle_action(action, filename=filename, confirmed=True)
    log.info('confirm: agent result success=%s', agent_result.success)
    resp = agent_result.to_dict()
    resp['download_url'] = f"/files/{filename}"
    return jsonify(resp)


@app.route('/api/undo', methods=['POST'])
def undo_endpoint():
    data = request.get_json() or {}
    filename = data.get('filename', 'data.xlsx')
    filename = secure_filename(filename)
    filepath = os.path.join(FILES_DIR, filename)
    # support multi-level undo: steps=1 (latest) .. MAX_BACKUPS
    try:
        steps = int(data.get('steps', 1))
    except (TypeError, ValueError):
        log.warning('undo: non-integer steps value: %r', data.get('steps'))
        return jsonify({"success": False, "error": "steps must be an integer"}), 400
    if steps < 1 or steps > MAX_BACKUPS:
        log.warning('undo: steps=%d out of range', steps)
        return jsonify({"success": False, "error": "invalid_steps"}), 400
    log.info('undo: file=%s  steps=%d', filename, steps)
    bak_n = f"{filepath}.bak.{steps}"
    if not os.path.exists(bak_n):
        return jsonify({"success": False, "error": "No checkpoint available for requested step"}), 404
    try:
        shutil.copy2(bak_n, filepath)
        msg = f'Undo successful, restored checkpoint {steps}.'
        log.info('undo: restored checkpoint %d for %s', steps, filename)
        return jsonify({"success": True, "message": msg, "speech_text": msg, "download_url": f"/files/{filename}"})
    except Exception as e:
        log.error('undo: failed to restore checkpoint %d: %s', steps, e)
        return jsonify({"success": False, "error": str(e)}), 500


def _enforce_backup_budget(filepath):
    """Delete oldest numbered backups until total size is within MAX_BACKUP_BYTES."""
    # Collect all numbered backups for this file
    backups = []
    for i in range(1, MAX_BACKUPS + 1):
        p = f"{filepath}.bak.{i}"
        if os.path.exists(p):
            backups.append((i, p, os.path.getsize(p)))
    # Also count the convenience .bak
    bak_link = f"{filepath}.bak"
    link_size = 0
    if os.path.exists(bak_link):
        link_size = os.path.getsize(bak_link)
    total = sum(s for _, _, s in backups) + link_size
    # Remove oldest (highest numbered) first until under budget
    backups.sort(key=lambda x: x[0], reverse=True)  # oldest = highest number
    for idx, p, sz in backups:
        if total <= MAX_BACKUP_BYTES:
            break
        try:
            os.remove(p)
            total -= sz
        except Exception:
            pass


def _purge_stale_confirmations():
    """Remove confirmation JSON files older than CONF_TTL_SECONDS."""
    now = time.time()
    for name in os.listdir(CONF_DIR):
        if not name.endswith('.json'):
            continue
        fpath = os.path.join(CONF_DIR, name)
        try:
            age = now - os.path.getmtime(fpath)
            if age > CONF_TTL_SECONDS:
                os.remove(fpath)
        except Exception:
            pass


@app.route('/api/backups', methods=['GET'])
def backups_info():
    """Return available checkpoints and total backup size for a file."""
    filename = request.args.get('filename', 'data.xlsx')
    filename = secure_filename(filename)
    filepath = os.path.join(FILES_DIR, filename)
    checkpoints = []
    total_bytes = 0
    for i in range(1, MAX_BACKUPS + 1):
        p = f"{filepath}.bak.{i}"
        if os.path.exists(p):
            sz = os.path.getsize(p)
            checkpoints.append({"step": i, "size_bytes": sz, "modified": os.path.getmtime(p)})
            total_bytes += sz
    return jsonify({
        "success": True,
        "filename": filename,
        "checkpoints": checkpoints,
        "total_bytes": total_bytes,
        "budget_bytes": MAX_BACKUP_BYTES,
    })


@app.route('/api/upload-audio', methods=['POST'])
def upload_audio():
    if 'file' not in request.files:
        log.warning('upload-audio: no file in request')
        return jsonify({"success": False, "error": "No file uploaded"}), 400
    audio = request.files['file']
    filename = secure_filename(audio.filename)
    if not filename:
        log.warning('upload-audio: empty filename after sanitization')
        return jsonify({"success": False, "error": "Invalid audio filename"}), 400
    # restrict audio extensions
    audio_ext = os.path.splitext(filename)[1].lower()
    if audio_ext not in {'.wav', '.mp3', '.flac', '.ogg', '.webm', '.m4a'}:
        log.warning('upload-audio: disallowed audio extension %r', audio_ext)
        return jsonify({"success": False, "error": f"Audio extension '{audio_ext}' not supported"}), 400
    # Use a UUID-prefixed temp name to avoid collisions on concurrent uploads.
    audio_ext = os.path.splitext(filename)[1].lower()
    temp_filename = f"audio_{uuid.uuid4().hex}{audio_ext}"
    saved_path = os.path.join(FILES_DIR, temp_filename)
    audio.save(saved_path)
    log.info('upload-audio: saved %s (%d bytes)', temp_filename, os.path.getsize(saved_path))

    try:
        # Transcribe audio — pass the full BCP-47 tag (e.g. 'en-US', 'th-TH')
        # so Google Cloud Speech-to-Text can use the most specific model.
        # speech_recognition.py normalises short codes ('en') automatically.
        language = request.form.get('language', 'en-US') or 'en-US'
        _t_stt = time.perf_counter()
        text = transcribe_audio_file(saved_path, language=language)
        log.info('[LATENCY] stt=%.0fms model=%s', (time.perf_counter() - _t_stt) * 1000,
                 os.environ.get('WHISPER_MODEL', 'base.en'))
        log.info('upload-audio: transcript=%r', text[:200] if text else '')

        # If speech recognition failed entirely, return a clear error instead
        # of wasting an LLM call on an error string.
        if not text or text.startswith('[transcription_error]'):
            reason = text or '[transcription_error] empty result'
            log.warning('upload-audio: transcription failed: %s', reason)
            return jsonify({"success": False, "error": "Speech recognition failed", "detail": reason}), 422

        auto_execute = request.form.get('auto_execute', 'false').lower() == 'true'
        auto_correct = request.form.get('auto_correct', 'false').lower() == 'true'
        raw_target = request.form.get('filename', '') or ''
        if not raw_target.strip():
            raw_target = memory.get_active_file() or 'data.xlsx'
        target_filename = secure_filename(raw_target)

        # Delegate to the agent for parsing + optional execution
        agent_result = agent.handle_text(
            text,
            filename=target_filename,
            auto_execute=auto_execute,
            auto_correct=auto_correct,
        )

        response = {"success": True, "transcript": text, "cleaned": text,
                     "action": agent_result.action}

        if agent_result.suggestions:
            response['suggestions'] = agent_result.suggestions
        if agent_result.recovery:
            response['recovery'] = agent_result.recovery
        if agent_result.auto_corrected:
            response['auto_corrected'] = True
        if agent_result.speech_text:
            response['speech_text'] = agent_result.speech_text

        if auto_execute and agent_result.executed:
            response['executed'] = True
            response['exec_result'] = agent_result.result
            response['download_url'] = agent_result.download_url or f"/files/{target_filename}"
            if agent_result.workflow_id:
                response['workflow_id'] = agent_result.workflow_id
            if agent_result.speech_text:
                response['speech_text'] = agent_result.speech_text
        elif auto_execute and not agent_result.executed:
            response['executed'] = False
            if agent_result.error:
                response['exec_note'] = agent_result.error
            elif not agent_result.action or (isinstance(agent_result.action, dict) and agent_result.action.get('type') in ('unknown', None)):
                response['exec_note'] = 'Command not recognised — execution skipped'
        else:
            response['executed'] = False

        return jsonify(response)
    except Exception as exc:
        log.exception('upload-audio: unhandled error')
        return jsonify({"success": False, "error": str(exc)}), 500
    finally:
        # Always remove the temporary audio file — we only need the transcript.
        try:
            if os.path.exists(saved_path):
                os.remove(saved_path)
                log.debug('upload-audio: removed temp file %s', temp_filename)
        except Exception:
            pass


# --------------- Memory helpers ---------------

def _update_memory_after_execution(action, filename, result):
    """Update the memory layer after a successful action execution."""
    if isinstance(result, dict) and 'error' in result:
        return  # don't update memory on failure

    # Set active file
    memory.set_active_file(filename)

    # Set active sheet when a workbook is created or opened
    atype = action.get('type', '') if isinstance(action, dict) else ''
    if atype in ('create_excel', 'create_sheet_and_add_row'):
        memory.set_active_sheet('Sheet1')
    elif atype == 'open_file' and isinstance(result, dict):
        memory.set_active_sheet(result.get('active_sheet', 'Sheet1'))

    # Record the command
    memory.add_command(action)


# --------------- Memory API ---------------

@app.route('/api/memory', methods=['GET'])
def memory_get():
    """Return the current memory state."""
    return jsonify({"success": True, "memory": memory.snapshot()})


@app.route('/api/memory', methods=['POST'])
def memory_set():
    """Update specific memory fields."""
    data = request.get_json() or {}
    if 'active_file' in data:
        val = data['active_file']
        if val is not None and not isinstance(val, str):
            return jsonify({"success": False, "error": "active_file must be a string or null"}), 400
        memory.set_active_file(val)
    if 'active_sheet' in data:
        val = data['active_sheet']
        if val is not None and not isinstance(val, str):
            return jsonify({"success": False, "error": "active_sheet must be a string or null"}), 400
        memory.set_active_sheet(val)
    return jsonify({"success": True, "memory": memory.snapshot()})


@app.route('/api/memory', methods=['DELETE'])
def memory_clear():
    """Reset all memory state."""
    memory.clear()
    return jsonify({"success": True, "memory": memory.snapshot()})


# --------------- Workflow API ---------------

@app.route('/api/workflows', methods=['GET'])
def workflows_list():
    """Return all registered workflow definitions."""
    return jsonify({"success": True, "workflows": workflow_router.list_workflows()})


@app.route('/api/system_status', methods=['GET'])
def system_status():
    """Return a monitoring snapshot of assistant state for debugging."""
    snap = memory.snapshot()
    workflows = workflow_router.list_workflows()
    internet = llm_router.check_internet_connection()
    ollama = llm_router.is_ollama_available()
    groq = llm_router.check_groq_available()
    model_installed = llm_router.check_ollama_model_installed()
    return jsonify({
        "active_file": snap.get("active_file", ""),
        "active_sheet": snap.get("active_sheet", ""),
        "recent_commands": snap.get("recent_commands", []),
        "memory_state": snap,
        "available_workflows": [wf.get("id") for wf in workflows],
        "internet_available": internet,
        "groq_available": groq,
        "ollama_available": ollama,
        "ollama_model": llm_router.OLLAMA_MODEL,
        "ollama_model_installed": model_installed,
        "mode": "online" if internet else "offline",
    })


@app.route('/files/<path:filename>', methods=['GET'])
def download_file(filename):
    return send_from_directory(FILES_DIR, filename, as_attachment=True)


@app.route('/favicon.ico')
def favicon():
    # Serve the SVG favicon from the frontend folder so browsers won't 404
    return send_from_directory(FRONTEND_DIR, 'favicon.svg')


# --------------- Metrics API ---------------

@app.route('/api/metrics', methods=['GET'])
def metrics_endpoint():
    """Return aggregated performance metrics for research evaluation."""
    s = _metrics.summary()
    providers = s['llm'].get('provider_breakdown', {})
    return jsonify({
        "success": True,
        "commands_processed": s['llm']['total_calls'],
        "avg_latency": round(s['llm']['avg_latency_ms'], 2),
        "groq_usage": providers.get('groq', 0),
        "ollama_usage": providers.get('ollama', 0),
        "heuristic_usage": providers.get('heuristic', 0),
        "success_rate": round(s['llm']['success_rate'], 4),
        "details": s,
        "pipeline_stages": _metrics.pipeline_stage_summary(),
        "sessions": _metrics.session_summary(),
    })


@app.route('/api/metrics/reset', methods=['POST'])
def metrics_reset():
    """Reset all collected metrics (useful before a benchmark run)."""
    _metrics.reset()
    return jsonify({"success": True, "message": "Metrics reset"})


# --------------- File Management API ---------------

@app.route('/api/files', methods=['GET'])
def list_files():
    """Return a list of user-uploaded Excel files with metadata."""
    files = []
    for name in sorted(os.listdir(FILES_DIR)):
        # Skip backup files
        if '.bak' in name:
            continue
        fpath = os.path.join(FILES_DIR, name)
        if not os.path.isfile(fpath):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            continue
        files.append({
            "name": name,
            "size_bytes": os.path.getsize(fpath),
            "modified": os.path.getmtime(fpath),
        })
    return jsonify({"success": True, "files": files})


@app.route('/api/delete_file', methods=['POST'])
def delete_file():
    """Delete a user file and all its backups."""
    data = request.get_json() or {}
    raw = data.get('filename', '')
    if not raw or not isinstance(raw, str):
        return jsonify({"success": False, "error": "filename is required"}), 400
    filename = secure_filename(raw)
    if not filename:
        return jsonify({"success": False, "error": "Invalid filename"}), 400
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"success": False, "error": f"Extension '{ext}' not allowed"}), 400
    filepath = os.path.join(FILES_DIR, filename)
    if not os.path.exists(filepath):
        return jsonify({"success": False, "error": "File not found"}), 404
    # Delete the file and all its backups
    os.remove(filepath)
    for suffix in ['.bak'] + [f'.bak.{i}' for i in range(1, MAX_BACKUPS + 1)]:
        bak = filepath + suffix
        if os.path.exists(bak):
            try:
                os.remove(bak)
            except Exception:
                pass
    # Clear memory if this was the active file
    if memory.get_active_file() == filename:
        memory.set_active_file(None)
        memory.set_active_sheet(None)
    log.info('delete_file: removed %s and backups', filename)
    return jsonify({"success": True, "message": f"Deleted {filename}"})


if __name__ == '__main__':
    # ── Startup readiness check ──
    internet = llm_router.check_internet_connection()
    ollama  = llm_router.is_ollama_available()
    groq    = llm_router.check_groq_available()
    mode    = 'ONLINE' if internet else 'OFFLINE'

    model_ok = llm_router.check_ollama_model_installed()

    print('\n┌─────────────────────────────────────┐')
    print('│        Sara — Startup Status        │')
    print('├─────────────────────────────────────┤')
    print(f'│  Internet : {"+" if internet else "-":>3}                     │')
    print(f'│  Groq API : {"+" if groq else "-":>3}                     │')
    print(f'│  Ollama   : {"+" if ollama else "-":>3}  ({llm_router.OLLAMA_MODEL})       │')
    print(f'│  Model    : {"+" if model_ok else "-":>3}  ({llm_router.OLLAMA_MODEL})       │')
    print(f'│  Mode     : {mode:<8}                  │')
    print('└─────────────────────────────────────┘\n')

    if not model_ok and ollama:
        print(f'⚠  Required Ollama model not found. Run:')
        print(f'   ollama pull {llm_router.OLLAMA_MODEL}\n')

    if not internet and not ollama:
        log.warning('Neither Groq nor Ollama available — only heuristic parsing will work')

    app.run(host='127.0.0.1', port=5000, debug=True)
