"""Worker de Kick Clips (Mision 002: hardening).

Contratos con n8n: se mantienen los nombres de endpoints y campos existentes.
Cambios de compatibilidad:
  - Todas las rutas (excepto /health) exigen el header X-Worker-Token.
  - open_ai_key ya no se usa ni se devuelve; la clave se lee de OPENAI_API_KEY.
  - Los archivos viven en /files/jobs/<job_id>/ (ver README).
"""
import datetime
import hmac
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from functools import wraps
from urllib.parse import urlparse

import requests as http_client
from flask import Flask, request, jsonify, send_file as flask_send_file

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 ** 3  # 8 GB (VODs largos y videos de Drive)

FILES_DIR = os.environ.get('FILES_DIR', '/files')
JOBS_SUBDIR = 'jobs'
ALLOWED_URL_HOST_SUFFIXES = ('kick.com', 'cloudfront.net')
QUALITY_RE = re.compile(r'^(best|worst|\d{3,4}p)$')
JOB_ID_RE = re.compile(r'^[a-f0-9]{32}$')
CHUNK_NAME_RE = re.compile(r'^chunk_\d{3}\.mp3$')
VIDEO_EXTENSIONS = ('.mp4', '.mkv', '.ts', '.mov', '.webm')
SECRET_KEYS = {'open_ai_key', 'openai_api_key', 'api_key', 'authorization', 'token', 'worker_token'}
RETRY_STATUS = {429, 500, 502, 503, 504}
MAX_RETRIES = 3
OPENAI_TIMEOUT = 300


# ---------------------------------------------------------------------------
# Logging estructurado (JSON por linea). Nunca registra secretos.
# ---------------------------------------------------------------------------
class JsonFormatter(logging.Formatter):
    def format(self, record):
        out = {
            'ts': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'level': record.levelname,
            'msg': record.getMessage(),
        }
        out.update(getattr(record, 'fields', {}))
        return json.dumps(out, ensure_ascii=False, default=str)


logger = logging.getLogger('worker')
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(JsonFormatter())
logger.addHandler(_handler)
logger.propagate = False


def log(level, msg, **fields):
    logger.log(level, msg, extra={'fields': fields})


# ---------------------------------------------------------------------------
# Errores tipados. El handler global los convierte en respuestas JSON.
# ---------------------------------------------------------------------------
class WorkerError(Exception):
    status = 500
    code = 'internal_error'


class ValidationError(WorkerError):
    status = 422
    code = 'invalid_input'


class MediaError(WorkerError):
    status = 422
    code = 'media_error'


class ExternalToolError(WorkerError):
    status = 502
    code = 'tool_failed'


class ToolTimeoutError(WorkerError):
    status = 504
    code = 'timeout'


class RetryableApiError(WorkerError):
    status = 502
    code = 'api_retryable'


class PermanentApiError(WorkerError):
    status = 502
    code = 'api_error'


@app.errorhandler(WorkerError)
def handle_worker_error(err):
    log(logging.ERROR, str(err), error_code=err.code, status=err.status)
    return jsonify({'error': str(err), 'error_code': err.code}), err.status


@app.errorhandler(413)
def handle_too_large(_err):
    return jsonify({'error': 'Archivo demasiado grande', 'error_code': 'payload_too_large'}), 413


# ---------------------------------------------------------------------------
# Autenticacion: token compartido entre n8n y worker. Falla cerrada.
# ---------------------------------------------------------------------------
def require_token(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        expected = os.environ.get('WORKER_TOKEN', '')
        if not expected:
            log(logging.ERROR, 'WORKER_TOKEN no configurado', error_code='config_missing')
            return jsonify({'error': 'Worker mal configurado', 'error_code': 'config_missing'}), 500
        provided = request.headers.get('X-Worker-Token', '')
        if not hmac.compare_digest(provided.encode('utf-8'), expected.encode('utf-8')):
            log(logging.WARNING, 'token invalido', endpoint=request.path)
            return jsonify({'error': 'No autorizado', 'error_code': 'unauthorized'}), 401
        return fn(*args, **kwargs)
    return wrapper


def openai_key():
    key = os.environ.get('OPENAI_API_KEY', '')
    if not key:
        raise WorkerError('OPENAI_API_KEY no configurada en el worker')
    return key


# ---------------------------------------------------------------------------
# Utilidades de rutas y jobs
# ---------------------------------------------------------------------------
def jobs_root():
    return os.path.realpath(os.path.join(FILES_DIR, JOBS_SUBDIR))


def safe_real_path(path, root):
    """Ruta real (symlinks resueltos) si esta dentro de root; si no, None."""
    if not isinstance(path, str) or not path or '\x00' in path:
        return None
    real = os.path.realpath(path)
    root_real = os.path.realpath(root)
    try:
        if os.path.commonpath([real, root_real]) != root_real:
            return None
    except ValueError:
        return None
    return real


def job_dir_for(path):
    """Directorio de job (/files/jobs/<id>) al que pertenece 'path', o None."""
    real = safe_real_path(path, jobs_root())
    if not real or real == jobs_root():
        return None
    parts = os.path.relpath(real, jobs_root()).split(os.sep)
    if not parts or not JOB_ID_RE.match(parts[0]):
        return None
    return os.path.join(jobs_root(), parts[0])


def require_job_file(path):
    """Valida que 'path' sea un archivo existente dentro de un job. Devuelve (real, job_dir)."""
    real = safe_real_path(path, jobs_root())
    job_dir = job_dir_for(path)
    if not real or not job_dir or not os.path.isfile(real):
        raise ValidationError('Archivo fuera del workspace de jobs o inexistente')
    return real, job_dir


def new_job():
    job_id = uuid.uuid4().hex
    job_dir = os.path.join(jobs_root(), job_id)
    os.makedirs(job_dir, exist_ok=False)
    return job_id, job_dir


def discard_job(job_dir):
    """Borra un job creado por este worker. Solo acepta directorios dentro de jobs_root()."""
    if job_dir and job_dir_for(job_dir) == os.path.realpath(job_dir):
        shutil.rmtree(job_dir, ignore_errors=True)


def passthrough_from(data, exclude):
    """Campos no consumidos que se devuelven a n8n. Nunca incluye secretos."""
    return {
        k: v for k, v in data.items()
        if k not in exclude and k.lower() not in SECRET_KEYS
    }


def request_json():
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


def clamp_int(value, default, low, high):
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, n))


_REDACT_RE = re.compile(r'(https?://[^\s\'"?]+)\?[^\s\'"]*')


def redact(text):
    """Quita query strings (pueden llevar firmas) de textos de error de herramientas."""
    return _REDACT_RE.sub(r'\1?<redactado>', text or '')


def run_tool(cmd, timeout, name):
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise ToolTimeoutError(f'Timeout en {name}')
    if result.returncode != 0:
        raise ExternalToolError(f'{name} fallo: ' + redact(result.stderr[-1500:]))
    return result


# ---------------------------------------------------------------------------
# Validaciones de entrada
# ---------------------------------------------------------------------------
def validate_vod_url(url):
    if not isinstance(url, str) or not url or len(url) > 2048:
        raise ValidationError('URL invalida')
    try:
        parsed = urlparse(url)
        port = parsed.port
    except ValueError:
        raise ValidationError('URL invalida')
    if parsed.scheme != 'https':
        raise ValidationError('Solo se aceptan URLs https')
    if parsed.username or parsed.password or port not in (None, 443):
        raise ValidationError('URL con credenciales o puerto no permitido')
    host = (parsed.hostname or '').lower()
    if not any(host == s or host.endswith('.' + s) for s in ALLOWED_URL_HOST_SUFFIXES):
        raise ValidationError('Dominio no permitido')
    return url


def validate_quality(quality):
    if not isinstance(quality, str) or not QUALITY_RE.match(quality):
        raise ValidationError('Calidad no permitida')
    return quality


# ---------------------------------------------------------------------------
# Media (ffprobe / ffmpeg)
# ---------------------------------------------------------------------------
def probe_duration(path):
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', path],
            capture_output=True, text=True, timeout=60
        )
    except subprocess.TimeoutExpired:
        raise ToolTimeoutError('Timeout en ffprobe')
    if result.returncode != 0:
        raise MediaError('ffprobe no pudo leer el archivo')
    try:
        duration = float(result.stdout.strip())
    except ValueError:
        raise MediaError('Duracion no disponible')
    if duration <= 0:
        raise MediaError('Duracion cero')
    return duration


def remove_if_exists(path):
    try:
        if path and os.path.isfile(path):
            os.remove(path)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# OpenAI: llamadas con validacion de status, reintentos y backoff
# ---------------------------------------------------------------------------
def openai_post(url, api_key, label, **kwargs):
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = http_client.post(
                url,
                headers={'Authorization': f'Bearer {api_key}'},
                timeout=OPENAI_TIMEOUT,
                **kwargs
            )
        except (http_client.Timeout, http_client.ConnectionError) as e:
            last_error = f'red/timeout ({type(e).__name__})'
        else:
            if resp.status_code == 200:
                try:
                    return resp.json()
                except ValueError:
                    raise PermanentApiError(f'{label}: respuesta no es JSON valido')
            if resp.status_code in RETRY_STATUS:
                last_error = f'HTTP {resp.status_code}'
            else:
                raise PermanentApiError(f'{label}: HTTP {resp.status_code}')
        log(logging.WARNING, 'reintento OpenAI', label=label, attempt=attempt, reason=last_error)
        if attempt < MAX_RETRIES:
            time.sleep(2 ** attempt)
    raise RetryableApiError(f'{label}: agotados {MAX_RETRIES} intentos ({last_error})')


def transcribe_chunk(api_key, chunk_path, chunk_name, offset_secs):
    """Devuelve lineas '[m:ss] texto'. Lanza error si la respuesta no es valida."""
    with open(chunk_path, 'rb') as f:
        payload = openai_post(
            'https://api.openai.com/v1/audio/transcriptions', api_key,
            label=f'whisper {chunk_name}',
            files={'file': (chunk_name, f, 'audio/mpeg')},
            data={'model': 'whisper-1', 'response_format': 'verbose_json',
                  'timestamp_granularities[]': 'segment', 'language': 'es'},
        )
    if not isinstance(payload, dict) or not isinstance(payload.get('segments', []), list):
        raise PermanentApiError(f'whisper {chunk_name}: respuesta sin segmentos validos')
    lines = []
    for seg in payload.get('segments', []):
        if not isinstance(seg, dict):
            continue
        start = seg.get('start')
        text = seg.get('text')
        if not isinstance(start, (int, float)) or not isinstance(text, str):
            continue
        text = ' '.join(text.split())
        if not text:
            continue
        t = offset_secs + float(start)
        lines.append(f'[{int(t // 60)}:{int(t % 60):02d}] {text}')
    return lines


def split_transcript(transcript, section_minutes=15):
    lines_by_section = {}
    last_idx = 0
    for line in transcript.split('\n'):
        line = line.strip()
        if not line:
            continue
        match = re.match(r'\[(\d+):(\d+)\]', line)
        if match:
            total_secs = int(match.group(1)) * 60 + int(match.group(2))
            last_idx = total_secs // (section_minutes * 60)
        lines_by_section.setdefault(last_idx, []).append(line)
    return [
        (idx * section_minutes * 60, '\n'.join(lines))
        for idx, lines in sorted(lines_by_section.items())
    ]


def quote_is_real(quote, section_text):
    """Verifica que la cita tenga palabras que realmente aparezcan en la transcripcion."""
    if not quote or len(quote) < 5:
        return True
    words = [w.lower().strip('.,!?¡¿"\'()') for w in quote.split() if len(w) > 3]
    if not words:
        return True
    text_lower = section_text.lower()
    matches = sum(1 for w in words if w in text_lower)
    return (matches / len(words)) >= 0.5


def deduplicate(moments, min_gap=90):
    result = []
    used = []
    for m in sorted(moments, key=lambda x: x.get('timestamp_seconds', 0)):
        ts = m.get('timestamp_seconds', 0)
        if not any(abs(ts - u) < min_gap for u in used):
            result.append(m)
            used.append(ts)
    return result


def call_gpt_for_section(api_key, channel, section_text, section_start, moments_per_section=8):
    # Prompt sin cambios respecto a la version desplegada (Mision 002 no redisena prompts).
    section_start_min = section_start // 60
    section_end_min = section_start_min + 15

    system_msg = (
        f'Eres un editor experto de TikTok para streams de {channel} en espanol latinoamericano. '
        f'Analiza SOLO el texto que te doy (minutos {section_start_min} a {section_end_min} del stream) '
        f'y encuentra momentos reales para clips.\n\n'
        f'TIPOS DE MOMENTOS: revelaciones personales, opiniones fuertes, anuncios, '
        f'situaciones en stream, interacciones con chat, historias, reacciones, frases memorables, retos.\n\n'
        f'REGLAS ABSOLUTAS:\n'
        f'1. SOLO incluye momentos que esten TEXTUALMENTE en la transcripcion que te doy\n'
        f'2. El campo "quote" DEBE ser una frase que aparezca literalmente en el texto\n'
        f'3. El timestamp_seconds es el segundo ABSOLUTO del stream (no relativo a la seccion)\n'
        f'   Ejemplo: si ves [45:30] en el texto, timestamp_seconds = 2730\n'
        f'4. Si no hay momentos buenos en este fragmento, devuelve moments: []\n'
        f'5. NO inventes nada. Solo lo que esta escrito.\n'
        f'6. Devuelve UNICAMENTE JSON valido, sin texto adicional'
    )

    user_msg = (
        f'Fragmento del stream de {channel} ({section_start_min} a {section_end_min} min). '
        f'Encuentra hasta {moments_per_section} momentos. '
        f'Timestamps en [minutos:segundos].\n\n'
        f'{section_text}\n\n'
        f'JSON:\n'
        f'{{"moments":[{{"timestamp_seconds":2730,"type":"revelacion",'
        f'"tiktok_title":"Titulo llamativo con emoji",'
        f'"tiktok_description":"2 lineas describiendo el momento",'
        f'"call_to_action":"Frase de engagement",'
        f'"reason":"por que es viral",'
        f'"quote":"frase textual del stream"}}]}}'
    )

    payload = openai_post(
        'https://api.openai.com/v1/chat/completions', api_key,
        label=f'gpt seccion {section_start_min}min',
        json={
            'model': 'gpt-4o-mini',
            'max_tokens': 3000,
            'messages': [
                {'role': 'system', 'content': system_msg},
                {'role': 'user', 'content': user_msg},
            ],
        },
    )
    try:
        content = payload['choices'][0]['message']['content'].strip()
    except (KeyError, IndexError, TypeError):
        raise PermanentApiError('gpt: respuesta sin contenido')
    if content.startswith('```'):
        lines = [l for l in content.split('\n') if not l.startswith('```')]
        content = '\n'.join(lines).strip()
    try:
        parsed = json.loads(content)
    except ValueError:
        raise PermanentApiError('gpt: JSON de momentos invalido')
    moments = parsed.get('moments', []) if isinstance(parsed, dict) else []
    if not isinstance(moments, list):
        raise PermanentApiError('gpt: campo moments invalido')

    valid = []
    for m in moments:
        if not isinstance(m, dict):
            continue
        ts = m.get('timestamp_seconds')
        if not isinstance(ts, (int, float)) or isinstance(ts, bool):
            continue
        m['timestamp_seconds'] = int(ts)
        if quote_is_real(m.get('quote', ''), section_text):
            valid.append(m)
    return valid


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.route('/health')
def health():
    return jsonify({'status': 'ok'})


@app.route('/download', methods=['POST'])
@require_token
def download():
    data = request_json()
    url = validate_vod_url(data.get('url'))
    quality = validate_quality(data.get('quality', 'best'))
    exclude = {'url', 'quality'}

    job_id, job_dir = new_job()
    log(logging.INFO, 'descarga iniciada', job_id=job_id, endpoint='/download')
    try:
        run_tool([
            'yt-dlp', '-f', quality, '-o', os.path.join(job_dir, 'vod.%(ext)s'),
            '--no-playlist', '--no-warnings',
            '--add-header', 'Referer:https://kick.com',
            '--add-header', 'Origin:https://kick.com',
            '--hls-use-mpegts', '--concurrent-fragments', '1', '--fragment-retries', '10',
            '--', url,
        ], timeout=7200, name='yt-dlp')
        candidates = [
            os.path.join(job_dir, f) for f in os.listdir(job_dir)
            if f.startswith('vod.') and not f.endswith('.part')
        ]
        candidates = [c for c in candidates if os.path.isfile(c) and os.path.getsize(c) > 0]
        if not candidates:
            raise ExternalToolError('yt-dlp no genero un archivo de video')
        vod_path = max(candidates, key=os.path.getmtime)
        duration = probe_duration(vod_path)
    except WorkerError:
        discard_job(job_dir)
        raise
    log(logging.INFO, 'descarga completa', job_id=job_id, duration_seconds=round(duration, 1))
    return jsonify({
        'file': vod_path,
        'job_id': job_id,
        'duration_seconds': round(duration, 2),
        **passthrough_from(data, exclude),
    })


@app.route('/extract_audio', methods=['POST'])
@require_token
def extract_audio():
    data = request_json()
    input_file = data.get('input', '')
    real_input, job_dir = require_job_file(input_file)
    chunk_minutes = clamp_int(data.get('chunk_minutes', 15), 15, 1, 30)

    audio_dir = os.path.join(job_dir, 'audio')
    os.makedirs(audio_dir, exist_ok=True)
    for stale in os.listdir(audio_dir):  # cada ejecucion empieza limpia dentro de su job
        remove_if_exists(os.path.join(audio_dir, stale))

    run_tool([
        'ffmpeg', '-y', '-i', real_input,
        '-vn', '-acodec', 'mp3', '-ar', '16000', '-ac', '1', '-b:a', '64k',
        '-f', 'segment', '-segment_time', str(chunk_minutes * 60), '-reset_timestamps', '1',
        os.path.join(audio_dir, 'chunk_%03d.mp3'),
    ], timeout=3600, name='ffmpeg audio')

    chunks = sorted(f for f in os.listdir(audio_dir) if CHUNK_NAME_RE.match(f))
    if not chunks or any(os.path.getsize(os.path.join(audio_dir, c)) == 0 for c in chunks):
        raise MediaError('ffmpeg no genero chunks de audio validos')
    return jsonify({
        'dir': audio_dir,
        'chunks': chunks,
        'vod_file': input_file,
        'job_id': os.path.basename(job_dir),
        **passthrough_from(data, {'input', 'chunk_minutes'}),
    })


@app.route('/transcribe', methods=['POST'])
@require_token
def transcribe():
    data = request_json()
    audio_dir = data.get('dir', '')
    job_dir = job_dir_for(audio_dir)
    audio_real = safe_real_path(audio_dir, jobs_root())
    if not job_dir or not audio_real or not os.path.isdir(audio_real):
        raise ValidationError('Directorio de audio fuera del workspace de jobs')
    chunks = data.get('chunks', [])
    if not isinstance(chunks, list) or not all(isinstance(c, str) and CHUNK_NAME_RE.match(c) for c in chunks):
        raise ValidationError('Lista de chunks invalida')
    chunk_minutes = clamp_int(data.get('chunk_minutes', 15), 15, 1, 30)
    api_key = openai_key()

    parts, failed = [], []
    for i, name in enumerate(chunks):
        path = os.path.join(audio_real, name)
        if not os.path.isfile(path):
            failed.append({'chunk': name, 'error': 'archivo inexistente'})
            continue
        try:
            parts.extend(transcribe_chunk(api_key, path, name, i * chunk_minutes * 60))
        except WorkerError as e:
            failed.append({'chunk': name, 'error': str(e)})
            log(logging.ERROR, 'chunk fallido', chunk=name, error_code=e.code)

    # Un chunk fallido deja huecos en la transcripcion; se falla todo en vez de producir clips sobre texto incompleto.
    if failed:
        return jsonify({
            'error': f'Whisper fallo en {len(failed)} de {len(chunks)} chunk(s)',
            'error_code': 'transcription_failed',
            'failed_chunks': failed,
        }), 502

    transcript = '\n'.join(parts) + ('\n' if parts else '')
    if not transcript.strip():
        raise PermanentApiError('transcripcion vacia (sin segmentos en ningun chunk)')
    transcript_path = os.path.join(job_dir, 'transcript.txt')
    with open(transcript_path, 'w', encoding='utf-8') as f:
        f.write(transcript)
    return jsonify({
        'transcript': transcript,
        'transcript_file': transcript_path,
        'audio_dir': audio_real,
        'job_id': os.path.basename(job_dir),
        **passthrough_from(data, {'dir', 'chunks', 'chunk_minutes'}),
    })


@app.route('/detect_moments', methods=['POST'])
@require_token
def detect_moments():
    data = request_json()
    transcript = data.get('transcript', '')
    if not isinstance(transcript, str) or not transcript.strip():
        raise ValidationError('Transcripcion vacia')
    channel = re.sub(r'[^\w\- ]', '', str(data.get('channel', 'stream')))[:50].strip() or 'stream'
    max_moments = clamp_int(data.get('max_moments', 50), 50, 1, 100)
    section_minutes = 15
    api_key = openai_key()

    job_dir = job_dir_for(data.get('vod_file', '')) or job_dir_for(data.get('audio_dir', ''))
    if not job_dir:
        raise ValidationError('Falta vod_file o audio_dir dentro de un job')

    sections = split_transcript(transcript, section_minutes=section_minutes)
    moments_per_section = max(5, max_moments // max(len(sections), 1))
    all_moments, errors = [], []
    for section_start, section_text in sections:
        if not section_text.strip():
            continue
        try:
            all_moments.extend(call_gpt_for_section(api_key, channel, section_text, section_start, moments_per_section))
        except WorkerError as e:
            errors.append(f'Seccion {section_start // 60}min: {e}')
            log(logging.ERROR, 'seccion fallida', section_min=section_start // 60, error_code=e.code)

    analyzed = [s for s in sections if s[1].strip()]
    if analyzed and len(errors) == len(analyzed):
        return jsonify({
            'error': 'Todas las secciones fallaron en GPT',
            'error_code': 'detection_failed',
            'errors': errors,
        }), 502

    all_moments.sort(key=lambda m: m.get('timestamp_seconds', 0))
    all_moments = deduplicate(all_moments, min_gap=90)[:max_moments]

    today = datetime.date.today().strftime('%Y-%m-%d')
    report_path = os.path.join(job_dir, 'report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f'REPORTE DE CLIPS - {channel.upper()} - {today}\n')
        f.write(f'Total: {len(all_moments)} clips en {len(sections)} secciones de {section_minutes} min\n')
        if errors:
            f.write(f'Errores: {"; ".join(errors)}\n')
        f.write('=' * 60 + '\n\n')
        for i, m in enumerate(all_moments, 1):
            ts = m.get('timestamp_seconds', 0)
            start = max(0, ts - 10)
            f.write(f'=== CLIP {i} ===\n')
            f.write(f'Tiempo en stream: {ts // 60}:{ts % 60:02d}\n')
            f.write(f'Tipo: {m.get("type", "")}\n')
            f.write(f'Titulo TikTok: {m.get("tiktok_title", "")}\n')
            f.write(f'Descripcion: {m.get("tiktok_description", "")}\n')
            f.write(f'Call to Action: {m.get("call_to_action", "")}\n')
            f.write(f'Frase clave: {m.get("quote", "")}\n')
            f.write(f'Por que es viral: {m.get("reason", "")}\n')
            f.write(f'Inicio sugerido: {start}s\n\n')

    # vod_file, audio_dir y clip_duration deben viajar: Parsear Momentos en n8n los necesita.
    passthrough = passthrough_from(data, {'transcript', 'channel', 'max_moments'})
    enriched = [{**m, **passthrough, 'report_file': report_path} for m in all_moments]
    return jsonify({
        'moments': enriched,
        'count': len(enriched),
        'sections_analyzed': len(sections),
        'report_file': report_path,
        'errors': errors,
        'job_id': os.path.basename(job_dir),
    })


@app.route('/clip', methods=['POST'])
@require_token
def clip():
    data = request_json()
    real_input, job_dir = require_job_file(data.get('input', ''))
    try:
        start = max(0, int(float(data.get('start', 0))))
        requested = max(1, int(float(data.get('duration', 55))))
    except (TypeError, ValueError):
        raise ValidationError('start o duration no numericos')

    vod_duration = probe_duration(real_input)
    if start >= vod_duration:
        raise ValidationError(f'start ({start}s) esta fuera del VOD ({int(vod_duration)}s)')
    effective = min(requested, int(vod_duration - start))
    if effective < 1:
        raise ValidationError('El clip resultante duraria menos de 1 segundo')

    clip_dir = os.path.join(job_dir, 'clips')
    os.makedirs(clip_dir, exist_ok=True)
    output_file = os.path.join(clip_dir, f'clip_{start}s_{uuid.uuid4().hex[:8]}.mp4')

    try:
        run_tool([
            'ffmpeg', '-n', '-ss', str(start), '-i', real_input,
            '-t', str(effective),
            '-vf', 'scale=608:1080:force_original_aspect_ratio=decrease,pad=608:1080:(ow-iw)/2:(oh-ih)/2:black',
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
            '-c:a', 'aac', '-b:a', '128k', output_file,
        ], timeout=300, name='ffmpeg clip')
        out_duration = probe_duration(output_file)
        if out_duration < 1.0:
            raise MediaError('El clip generado es demasiado corto')
    except WorkerError:
        remove_if_exists(output_file)
        raise

    return jsonify({
        'file': output_file,
        'job_id': os.path.basename(job_dir),
        'start_seconds': start,
        'duration_seconds': round(out_duration, 2),
        **passthrough_from(data, {'input', 'start', 'duration'}),
    })


@app.route('/serve_file')
@require_token
def serve_file():
    real = safe_real_path(request.args.get('path', ''), jobs_root())
    if not real or not os.path.isfile(real) or job_dir_for(real) is None:
        return jsonify({'error': 'Archivo no encontrado', 'error_code': 'not_found'}), 404
    mimetype = 'text/plain' if real.endswith('.txt') else 'video/mp4'
    return flask_send_file(real, mimetype=mimetype)


@app.route('/cleanup', methods=['POST'])
@require_token
def cleanup():
    data = request_json()
    real = safe_real_path(data.get('path', ''), jobs_root())
    if not real or real == jobs_root() or job_dir_for(real) is None:
        raise ValidationError('Ruta fuera del workspace de jobs')
    if os.path.isdir(real) and not os.path.islink(real):
        shutil.rmtree(real)
    elif os.path.isfile(real):
        os.remove(real)
    else:
        return jsonify({'status': 'not_found', 'removed': real})
    return jsonify({'status': 'ok', 'removed': real})


@app.route('/save_file', methods=['POST'])
@require_token
def save_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No se recibio ningun archivo', 'error_code': 'missing_file'}), 400
    upload = request.files['file']
    original = re.sub(r'[^a-zA-Z0-9._-]', '_', upload.filename or 'video.mp4')
    ext = os.path.splitext(original)[1].lower()
    if ext not in VIDEO_EXTENSIONS:
        raise ValidationError('Extension de video no permitida')

    job_id, job_dir = new_job()
    path = os.path.join(job_dir, 'source' + ext)
    try:
        upload.save(path)
        duration = probe_duration(path)
    except WorkerError:
        discard_job(job_dir)
        raise
    return jsonify({
        'file': path,
        'job_id': job_id,
        'duration_seconds': round(duration, 2),
        'source_name': original,
        **passthrough_from(request.form.to_dict(), {'file'}),
    })


@app.route('/resolve_kick_url', methods=['POST'])
@require_token
def resolve_kick_url():
    data = request_json()
    url = validate_vod_url(data.get('url'))
    result = run_tool([
        'yt-dlp', '-j', '--no-playlist',
        '--add-header', 'Referer:https://kick.com',
        '--add-header', 'Origin:https://kick.com',
        '--', url,
    ], timeout=60, name='yt-dlp -j')
    try:
        info = json.loads(result.stdout)
    except ValueError:
        raise ExternalToolError('yt-dlp devolvio JSON invalido')
    hls_url = info.get('url', '')
    if not hls_url and info.get('formats'):
        for fmt in reversed(info['formats']):
            u = fmt.get('url', '')
            if 'm3u8' in u or str(fmt.get('protocol', '')).startswith('m3u8'):
                hls_url = u
                break
        if not hls_url:
            hls_url = info['formats'][-1].get('url', '')
    if not hls_url:
        raise ExternalToolError('No se encontro URL del stream en la respuesta de yt-dlp')
    return jsonify({'hls_url': hls_url, **passthrough_from(data, {'url'})})


if __name__ == '__main__':
    # Solo para desarrollo local. En produccion usar gunicorn (ver README).
    app.run(host='0.0.0.0', port=8080, debug=False)
