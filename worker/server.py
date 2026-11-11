from flask import Flask, request, jsonify, send_file as flask_send_file
import subprocess, os, json, glob, shutil, datetime, re
import requests as http_client

app = Flask(__name__)
FILES_DIR = '/files'


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
    """Elimina momentos duplicados o demasiado proximos entre si."""
    result = []
    used = []
    for m in sorted(moments, key=lambda x: x.get('timestamp_seconds', 0)):
        ts = m.get('timestamp_seconds', 0)
        if not any(abs(ts - u) < min_gap for u in used):
            result.append(m)
            used.append(ts)
    return result


def call_gpt_for_section(api_key, channel, section_text, section_start, moments_per_section=8):
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

    resp = http_client.post(
        'https://api.openai.com/v1/chat/completions',
        headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'},
        json={
            'model': 'gpt-4o-mini',
            'max_tokens': 3000,
            'messages': [
                {'role': 'system', 'content': system_msg},
                {'role': 'user', 'content': user_msg}
            ]
        },
        timeout=120
    )
    resp.raise_for_status()
    content = resp.json()['choices'][0]['message']['content'].strip()
    if content.startswith('```'):
        lines = content.split('\n')
        lines = [l for l in lines if not l.startswith('```')]
        content = '\n'.join(lines).strip()
    moments = json.loads(content).get('moments', [])

    # Filtrar alucinaciones: la quote debe existir en el texto
    real = [m for m in moments if quote_is_real(m.get('quote', ''), section_text)]
    return real


@app.route('/health')
def health():
    return jsonify({'status': 'ok'})


@app.route('/download', methods=['POST'])
def download():
    data = request.json or {}
    url = data.get('url', '')
    quality = data.get('quality', '360p')
    passthrough = {k: v for k, v in data.items() if k not in ('url', 'quality')}
    os.makedirs(FILES_DIR, exist_ok=True)
    output_template = os.path.join(FILES_DIR, 'vod_%(id)s.%(ext)s')
    try:
        result = subprocess.run(
            ['yt-dlp', '-f', quality, '-o', output_template,
             '--no-playlist', '--no-warnings',
             '--add-header', 'Referer:https://kick.com',
             '--add-header', 'Origin:https://kick.com',
             '--hls-use-mpegts', '--concurrent-fragments', '1',
             '--fragment-retries', '10', url],
            capture_output=True, text=True, timeout=7200
        )
        if result.returncode != 0:
            return jsonify({'error': result.stderr[-2000:]}), 500
        files = sorted(glob.glob(os.path.join(FILES_DIR, 'vod_*')), key=os.path.getmtime, reverse=True)
        if not files:
            return jsonify({'error': 'No se encontro el archivo descargado'}), 500
        return jsonify({'file': files[0], **passthrough})
    except subprocess.TimeoutExpired:
        return jsonify({'error': 'Timeout en la descarga'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/extract_audio', methods=['POST'])
def extract_audio():
    data = request.json or {}
    input_file = data.get('input', '')
    chunk_minutes = int(data.get('chunk_minutes', 15))
    passthrough = {k: v for k, v in data.items() if k not in ('input', 'chunk_minutes')}
    base = os.path.splitext(os.path.basename(input_file))[0]
    audio_dir = os.path.join(FILES_DIR, f'audio_{base}')
    os.makedirs(audio_dir, exist_ok=True)
    try:
        result = subprocess.run([
            'ffmpeg', '-y', '-i', input_file,
            '-vn', '-acodec', 'mp3', '-ar', '16000', '-ac', '1', '-b:a', '64k',
            '-f', 'segment', '-segment_time', str(chunk_minutes * 60),
            '-reset_timestamps', '1',
            os.path.join(audio_dir, 'chunk_%03d.mp3')
        ], capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            return jsonify({'error': result.stderr[-2000:]}), 500
        chunks = sorted(f for f in os.listdir(audio_dir) if f.endswith('.mp3'))
        return jsonify({'dir': audio_dir, 'chunks': chunks, 'vod_file': input_file, **passthrough})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/transcribe', methods=['POST'])
def transcribe():
    data = request.json or {}
    audio_dir = data.get('dir', '')
    chunks = data.get('chunks', [])
    api_key = data.get('api_key', '')
    chunk_minutes = int(data.get('chunk_minutes', 15))
    passthrough = {k: v for k, v in data.items()
                   if k not in ('dir', 'chunks', 'api_key', 'chunk_minutes')}
    full_transcript = ''
    for i, chunk_name in enumerate(chunks):
        chunk_path = os.path.join(audio_dir, chunk_name)
        offset_secs = i * chunk_minutes * 60
        if not os.path.exists(chunk_path):
            continue
        try:
            with open(chunk_path, 'rb') as f:
                resp = http_client.post(
                    'https://api.openai.com/v1/audio/transcriptions',
                    headers={'Authorization': f'Bearer {api_key}'},
                    files={'file': (chunk_name, f, 'audio/mpeg')},
                    data={'model': 'whisper-1', 'response_format': 'verbose_json',
                          'timestamp_granularities[]': 'segment', 'language': 'es'},
                    timeout=300
                )
            whisper_data = resp.json()
            for seg in whisper_data.get('segments', []):
                t = offset_secs + seg.get('start', 0)
                m, s = int(t // 60), int(t % 60)
                full_transcript += f'[{m}:{s:02d}] {seg["text"].strip()}\n'
        except Exception as e:
            full_transcript += f'[chunk {i} error: {e}]\n'
    if not full_transcript.strip():
        return jsonify({'error': 'No se pudo transcribir el audio'}), 500
    # Guardar transcripcion para debug
    with open('/files/transcript_latest.txt', 'w') as f: f.write(full_transcript)
    return jsonify({'transcript': full_transcript, 'audio_dir': audio_dir, **passthrough})


@app.route('/detect_moments', methods=['POST'])
def detect_moments():
    data = request.json or {}
    transcript = data.get('transcript', '')
    api_key = data.get('api_key', '')
    channel = data.get('channel', 'stream')
    max_moments = int(data.get('max_moments', 50))
    section_minutes = 15
    passthrough = {k: v for k, v in data.items()
                   if k not in ('transcript', 'api_key', 'channel', 'max_moments')}

    sections = split_transcript(transcript, section_minutes=section_minutes)
    moments_per_section = max(5, max_moments // max(len(sections), 1))

    all_moments = []
    errors = []

    for section_start, section_text in sections:
        if not section_text.strip():
            continue
        try:
            section_moments = call_gpt_for_section(
                api_key, channel, section_text, section_start, moments_per_section
            )
            all_moments.extend(section_moments)
        except Exception as e:
            errors.append(f'Seccion {section_start//60}min: {str(e)}')

    # Ordenar, deduplicar y limitar
    all_moments.sort(key=lambda m: m.get('timestamp_seconds', 0))
    all_moments = deduplicate(all_moments, min_gap=90)
    all_moments = all_moments[:max_moments]

    today = datetime.date.today().strftime('%Y-%m-%d')
    report_path = os.path.join(FILES_DIR, 'report_latest.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f'REPORTE DE CLIPS - {channel.upper()} - {today}\n')
        f.write(f'Total: {len(all_moments)} clips en {len(sections)} secciones de {section_minutes} min\n')
        if errors:
            f.write(f'Errores: {", ".join(errors)}\n')
        f.write('=' * 60 + '\n\n')
        for i, m in enumerate(all_moments, 1):
            ts = m.get('timestamp_seconds', 0)
            start = max(0, ts - 10)
            mins, secs = ts // 60, ts % 60
            f.write(f'=== CLIP {i} ===\n')
            f.write(f'Tiempo en stream: {mins}:{secs:02d}\n')
            f.write(f'Tipo: {m.get("type", "")}\n')
            f.write(f'Titulo TikTok: {m.get("tiktok_title", "")}\n')
            f.write(f'Descripcion: {m.get("tiktok_description", "")}\n')
            f.write(f'Call to Action: {m.get("call_to_action", "")}\n')
            f.write(f'Frase clave: {m.get("quote", "")}\n')
            f.write(f'Por que es viral: {m.get("reason", "")}\n')
            f.write(f'Archivo clip: clip_{start}s.mp4\n\n')

    enriched = [{**m, **passthrough, 'report_file': report_path} for m in all_moments]
    return jsonify({
        'moments': enriched,
        'count': len(enriched),
        'sections_analyzed': len(sections),
        'report_file': report_path,
        'errors': errors
    })


@app.route('/clip', methods=['POST'])
def clip():
    data = request.json or {}
    input_file = data.get('input', '')
    start = int(data.get('start', 0))
    duration = int(data.get('duration', 55))
    output_file = os.path.join(FILES_DIR, f'clip_{start}s.mp4')
    try:
        result = subprocess.run([
            'ffmpeg', '-y', '-ss', str(start), '-i', input_file,
            '-t', str(duration),
            '-vf', 'scale=608:1080:force_original_aspect_ratio=decrease,pad=608:1080:(ow-iw)/2:(oh-ih)/2:black',
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
            '-c:a', 'aac', '-b:a', '128k', output_file
        ], capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            return jsonify({'error': result.stderr[-2000:]}), 500
        return jsonify({'file': output_file})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/serve_file')
def serve_file():
    path = request.args.get('path', '')
    if os.path.exists(path):
        mimetype = 'text/plain' if path.endswith('.txt') else 'video/mp4'
        return flask_send_file(path, mimetype=mimetype)
    return jsonify({'error': 'File not found'}), 404


@app.route('/cleanup', methods=['POST'])
def cleanup():
    data = request.json or {}
    path = data.get('path', '')
    try:
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)
        return jsonify({'status': 'ok', 'removed': path})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/save_file', methods=['POST'])
def save_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No se recibio ningun archivo'}), 400
    f = request.files['file']
    filename = f.filename or 'video_drive.mp4'
    safe_name = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
    path = os.path.join(FILES_DIR, safe_name)
    f.save(path)
    extra = {k: v for k, v in request.form.items()}
    return jsonify({'file': path, **extra})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)


# NOTA: esta ruta esta DESPUES de app.run() en el archivo desplegado.
# Si el worker arranca con 'python server.py', no se registra. Pendiente de corregir
# tras confirmar cómo arranca el contenedor (ver kick-clips-m001-audit.md, H-01).
@app.route('/resolve_kick_url', methods=['POST'])
def resolve_kick_url():
    data = request.json or {}
    url = data.get('url', '')
    passthrough = {k: v for k, v in data.items() if k != 'url'}
    try:
        result = subprocess.run(
            ['yt-dlp', '-j', '--no-playlist',
             '--add-header', 'Referer:https://kick.com',
             '--add-header', 'Origin:https://kick.com',
             url],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            return jsonify({'error': result.stderr[-2000:]}), 500
        info = json.loads(result.stdout)
        hls_url = info.get('url', '')
        if not hls_url and info.get('formats'):
            for fmt in reversed(info['formats']):
                u = fmt.get('url', '')
                if 'm3u8' in u or fmt.get('protocol', '').startswith('m3u8'):
                    hls_url = u
                    break
            if not hls_url:
                hls_url = info['formats'][-1].get('url', '')
        if not hls_url:
            return jsonify({'error': 'No se encontro URL del stream en la respuesta de yt-dlp'}), 500
        return jsonify({'hls_url': hls_url, **passthrough})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
