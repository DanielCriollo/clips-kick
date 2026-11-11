"""Pruebas del worker de Kick Clips (Mision 002).

Todo se ejecuta con mocks: no hay descargas, ni llamadas a OpenAI, ni ffmpeg/yt-dlp reales.
Ejecutar desde la raiz del repo:  python -m pytest -q tests
"""
import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'worker'))

import server  # noqa: E402
import engine  # noqa: E402
import requests  # noqa: E402

TOKEN = 'token-de-prueba-123'
SECRET = 'sk-test-SECRET-NO-DEBE-APARECER'
HEADERS = {'X-Worker-Token': TOKEN}
VALID_KICK_URL = 'https://kick.com/sorning24/videos/019f809e-da10-7dae-a044-c6c4b0c69956'


# ---------------------------------------------------------------------------
# Fixtures y mocks
# ---------------------------------------------------------------------------
class FakeResponse:
    def __init__(self, status=200, payload=None, text=''):
        self.status_code = status
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError('no json')
        return self._payload


@pytest.fixture
def env(tmp_path, monkeypatch):
    files = tmp_path / 'files'
    files.mkdir()
    monkeypatch.setattr(server, 'FILES_DIR', str(files))
    monkeypatch.setenv('WORKER_TOKEN', TOKEN)
    monkeypatch.setenv('OPENAI_API_KEY', SECRET)
    state = {
        'calls': [],
        'ytdlp_rc': 0,
        'vod_duration': 3600.0,
        'clip_out': 55.0,
        'probe_fail': False,
    }

    def fake_run(cmd, capture_output=False, text=False, timeout=None, **_):
        state['calls'].append(cmd)
        tool = cmd[0]
        if tool == 'yt-dlp':
            if '-j' in cmd:
                return subprocess.CompletedProcess(cmd, 0, stdout='{"url": "https://d26.cloudfront.net/x/1.m3u8?sig=abc"}', stderr='')
            if state['ytdlp_rc'] != 0:
                return subprocess.CompletedProcess(cmd, state['ytdlp_rc'], stdout='', stderr='ERROR https://x.test/a?sig=SECRETQ')
            template = cmd[cmd.index('-o') + 1]
            out = template.replace('%(ext)s', 'mp4')
            with open(out, 'wb') as f:
                f.write(b'\x00' * 1000)
            return subprocess.CompletedProcess(cmd, 0, stdout='', stderr='')
        if tool == 'ffmpeg':
            if '-segment_time' in cmd:
                pattern = cmd[-1]
                for i in range(2):
                    with open(pattern % i, 'wb') as f:
                        f.write(b'\x01' * 500)
            else:
                with open(cmd[-1], 'wb') as f:
                    f.write(b'\x02' * 500)
            return subprocess.CompletedProcess(cmd, 0, stdout='', stderr='')
        if tool == 'ffprobe':
            path = cmd[-1]
            if state['probe_fail']:
                return subprocess.CompletedProcess(cmd, 1, stdout='', stderr='bad')
            if 'clip_' in os.path.basename(path):
                value = state['clip_out']
            else:
                value = state['vod_duration']
            return subprocess.CompletedProcess(cmd, 0, stdout=f'{value}\n', stderr='')
        raise AssertionError(f'comando inesperado: {tool}')

    monkeypatch.setattr(server.subprocess, 'run', fake_run)
    monkeypatch.setattr(server.time, 'sleep', lambda s: None)
    server.app.config['TESTING'] = True
    client = server.app.test_client()
    return {'client': client, 'files': str(files), 'state': state, 'monkeypatch': monkeypatch}


def make_job(files_dir):
    job_id = 'a' * 32
    job_dir = os.path.join(files_dir, 'jobs', job_id)
    os.makedirs(job_dir, exist_ok=True)
    vod = os.path.join(job_dir, 'vod.mp4')
    with open(vod, 'wb') as f:
        f.write(b'\x00' * 100)
    return job_dir, vod


# ---------------------------------------------------------------------------
# Autenticacion
# ---------------------------------------------------------------------------
def test_health_does_not_require_token(env):
    r = env['client'].get('/health')
    assert r.status_code == 200 and r.get_json() == {'status': 'ok'}


def test_missing_token_is_401(env):
    r = env['client'].post('/cleanup', json={'path': '/x'})
    assert r.status_code == 401


def test_wrong_token_is_401(env):
    r = env['client'].post('/cleanup', json={'path': '/x'}, headers={'X-Worker-Token': 'nope'})
    assert r.status_code == 401


def test_missing_server_token_fails_closed(env):
    env['monkeypatch'].delenv('WORKER_TOKEN')
    r = env['client'].post('/cleanup', json={'path': '/x'}, headers=HEADERS)
    assert r.status_code == 500 and r.get_json()['error_code'] == 'config_missing'


# ---------------------------------------------------------------------------
# Rutas, symlinks y validacion de URLs
# ---------------------------------------------------------------------------
def test_safe_real_path_rejects_traversal(env):
    root = server.jobs_root()
    assert server.safe_real_path(os.path.join(root, '..', 'secreto'), root) is None


def test_job_dir_for_rejects_non_job_paths(env):
    assert server.job_dir_for(os.path.join(env['files'], 'vod_viejo.mp4')) is None
    assert server.job_dir_for(os.path.join(env['files'], 'jobs', 'no-hex', 'vod.mp4')) is None


def test_symlink_escape_is_rejected(env, tmp_path):
    job_dir, _ = make_job(env['files'])
    outside = tmp_path / 'afuera.txt'
    outside.write_text('dato')
    link = os.path.join(job_dir, 'enlace.txt')
    os.symlink(str(outside), link)
    r = env['client'].get('/serve_file', query_string={'path': link}, headers=HEADERS)
    assert r.status_code == 404


def test_cleanup_refuses_files_outside_jobs(env):
    stray = os.path.join(env['files'], 'vod_viejo.mp4')
    open(stray, 'wb').close()
    r = env['client'].post('/cleanup', json={'path': stray}, headers=HEADERS)
    assert r.status_code == 422 and os.path.exists(stray)


def test_cleanup_refuses_jobs_root(env):
    os.makedirs(server.jobs_root(), exist_ok=True)
    r = env['client'].post('/cleanup', json={'path': server.jobs_root()}, headers=HEADERS)
    assert r.status_code == 422 and os.path.isdir(server.jobs_root())


def test_cleanup_removes_job_file(env):
    job_dir, vod = make_job(env['files'])
    r = env['client'].post('/cleanup', json={'path': vod}, headers=HEADERS)
    assert r.status_code == 200 and not os.path.exists(vod)


def test_serve_file_refuses_outside_jobs(env):
    stray = os.path.join(env['files'], 'report_latest.txt')
    open(stray, 'w').write('x')
    r = env['client'].get('/serve_file', query_string={'path': stray}, headers=HEADERS)
    assert r.status_code == 404


@pytest.mark.parametrize('url', [
    'http://kick.com/x/videos/1',
    'https://evil.com/kick.com',
    'https://kick.com.evil.com/videos/1',
    '-x',
    'https://user:pass@kick.com/videos/1',
    'https://kick.com:8443/videos/1',
    'file:///etc/passwd',
    '',
    None,
])
def test_vod_url_rejections(url):
    with pytest.raises(server.ValidationError):
        server.validate_vod_url(url)


def test_vod_url_accepts_kick_page_and_cdn():
    assert server.validate_vod_url(VALID_KICK_URL) == VALID_KICK_URL
    cdn = 'https://d26yk4zpyhjeeq.cloudfront.net/v1/manifest/abc/1.m3u8'
    assert server.validate_vod_url(cdn) == cdn


@pytest.mark.parametrize('q,ok', [('best', True), ('720p', True), ('--exec', False), ('best; rm', False), ('', False)])
def test_quality_allowlist(q, ok):
    if ok:
        assert server.validate_quality(q) == q
    else:
        with pytest.raises(server.ValidationError):
            server.validate_quality(q)


def test_redact_strips_signed_query():
    assert '<redactado>' in server.redact('ERROR https://x.test/a.m3u8?sig=ABC123 fallo')
    assert 'ABC123' not in server.redact('ERROR https://x.test/a.m3u8?sig=ABC123 fallo')


# ---------------------------------------------------------------------------
# Descarga y aislamiento por job
# ---------------------------------------------------------------------------
def test_download_creates_job_and_never_echoes_secrets(env):
    body = {'url': VALID_KICK_URL, 'quality': 'best', 'open_ai_key': SECRET, 'channel': 'jaap-z'}
    r = env['client'].post('/download', json=body, headers=HEADERS)
    assert r.status_code == 200
    data = r.get_json()
    assert data['job_id'] and data['file'].startswith(os.path.join(env['files'], 'jobs'))
    assert data['channel'] == 'jaap-z'
    assert 'open_ai_key' not in data and SECRET not in r.get_data(as_text=True)


def test_download_passes_url_after_double_dash(env):
    env['client'].post('/download', json={'url': VALID_KICK_URL}, headers=HEADERS)
    cmd = [c for c in env['state']['calls'] if c[0] == 'yt-dlp'][0]
    assert cmd[-2] == '--' and cmd[-1] == VALID_KICK_URL


def test_two_downloads_use_different_job_dirs(env):
    a = env['client'].post('/download', json={'url': VALID_KICK_URL}, headers=HEADERS).get_json()
    b = env['client'].post('/download', json={'url': VALID_KICK_URL}, headers=HEADERS).get_json()
    assert a['job_id'] != b['job_id']
    assert os.path.dirname(a['file']) != os.path.dirname(b['file'])


def test_failed_download_discards_its_job(env):
    env['state']['ytdlp_rc'] = 1
    r = env['client'].post('/download', json={'url': VALID_KICK_URL}, headers=HEADERS)
    assert r.status_code == 502
    assert not os.listdir(os.path.join(env['files'], 'jobs'))
    assert 'SECRETQ' not in r.get_data(as_text=True)


def test_download_rejects_bad_url_before_any_subprocess(env):
    r = env['client'].post('/download', json={'url': 'https://evil.com/x'}, headers=HEADERS)
    assert r.status_code == 422
    assert not any(c[0] == 'yt-dlp' for c in env['state']['calls'])


# ---------------------------------------------------------------------------
# Audio, transcripcion y errores de Whisper
# ---------------------------------------------------------------------------
def test_extract_rejects_input_outside_jobs(env):
    stray = os.path.join(env['files'], 'vod_viejo.mp4')
    open(stray, 'wb').write(b'x')
    r = env['client'].post('/extract_audio', json={'input': stray, 'chunk_minutes': 15}, headers=HEADERS)
    assert r.status_code == 422


def test_extract_writes_chunks_inside_job(env):
    _, vod = make_job(env['files'])
    r = env['client'].post('/extract_audio', json={'input': vod, 'chunk_minutes': 15}, headers=HEADERS)
    assert r.status_code == 200
    data = r.get_json()
    assert data['dir'].startswith(os.path.join(server.jobs_root(), 'a' * 32))
    assert data['chunks'] == ['chunk_000.mp3', 'chunk_001.mp3']


def _transcribe_body(audio_dir, chunks=('chunk_000.mp3', 'chunk_001.mp3')):
    for name in chunks:
        with open(os.path.join(audio_dir, name), 'wb') as f:
            f.write(b'\x03' * 100)
    return {'dir': audio_dir, 'chunks': list(chunks), 'chunk_minutes': 15}


def _audio_dir(env):
    job_dir, _ = make_job(env['files'])
    d = os.path.join(job_dir, 'audio')
    os.makedirs(d, exist_ok=True)
    return d


def _ok_whisper(segments):
    return FakeResponse(200, {'segments': segments})


def test_transcribe_retries_then_succeeds(env):
    seq = [FakeResponse(500), FakeResponse(429), _ok_whisper([{'start': 1.0, 'text': ' hola '}])]
    env['monkeypatch'].setattr(requests, 'post', lambda *a, **k: seq.pop(0))
    body = _transcribe_body(_audio_dir(env), chunks=('chunk_000.mp3',))
    r = env['client'].post('/transcribe', json=body, headers=HEADERS)
    assert r.status_code == 200
    assert r.get_json()['transcript'].strip() == '[0:01] hola'


def test_transcribe_non_retryable_status_fails_without_inserting_text(env):
    env['monkeypatch'].setattr(requests, 'post', lambda *a, **k: FakeResponse(401))
    body = _transcribe_body(_audio_dir(env), chunks=('chunk_000.mp3',))
    r = env['client'].post('/transcribe', json=body, headers=HEADERS)
    data = r.get_json()
    assert r.status_code == 502 and data['error_code'] == 'transcription_failed'
    assert 'transcript' not in data and 'error' in data


def test_transcribe_invalid_body_fails(env):
    env['monkeypatch'].setattr(requests, 'post', lambda *a, **k: FakeResponse(200, None, '<html>'))
    body = _transcribe_body(_audio_dir(env), chunks=('chunk_000.mp3',))
    r = env['client'].post('/transcribe', json=body, headers=HEADERS)
    assert r.status_code == 502


def test_transcribe_partial_chunk_failure_fails_whole(env):
    seq = [_ok_whisper([{'start': 0, 'text': 'uno'}]), FakeResponse(401)]
    env['monkeypatch'].setattr(requests, 'post', lambda *a, **k: seq.pop(0))
    body = _transcribe_body(_audio_dir(env))
    r = env['client'].post('/transcribe', json=body, headers=HEADERS)
    assert r.status_code == 502 and len(r.get_json()['failed_chunks']) == 1


def test_transcribe_timeouts_exhaust_retries(env):
    def boom(*a, **k):
        raise requests.Timeout('timeout')
    env['monkeypatch'].setattr(requests, 'post', boom)
    body = _transcribe_body(_audio_dir(env), chunks=('chunk_000.mp3',))
    r = env['client'].post('/transcribe', json=body, headers=HEADERS)
    assert r.status_code == 502
    assert 'agotados' in r.get_json()['failed_chunks'][0]['error']


def test_transcribe_rejects_path_traversal_in_chunk_names(env):
    body = {'dir': _audio_dir(env), 'chunks': ['../../etc/passwd'], 'chunk_minutes': 15}
    r = env['client'].post('/transcribe', json=body, headers=HEADERS)
    assert r.status_code == 422


def test_transcribe_without_openai_key_is_config_error(env):
    env['monkeypatch'].delenv('OPENAI_API_KEY')
    body = _transcribe_body(_audio_dir(env), chunks=('chunk_000.mp3',))
    r = env['client'].post('/transcribe', json=body, headers=HEADERS)
    assert r.status_code == 500


# ---------------------------------------------------------------------------
# Deteccion de momentos (GPT) con mocks
# ---------------------------------------------------------------------------
TRANSCRIPT = '[1:00] el streamer dice algo muy fuerte sobre el chat\n[2:30] se rie mucho del chat\n'


def _gpt(content):
    return FakeResponse(200, {'choices': [{'message': {'content': content}}]})


def _job_vod(env):
    _, vod = make_job(env['files'])
    return vod


def _job_with_segments(env, n=40):
    job_dir, vod = make_job(env['files'])
    raw = [{'start': i * 5.0, 'end': i * 5.0 + 4.0,
            'text': f'frase numero {i} del stream con gente hablando'} for i in range(n)]
    segs = engine.assign_segment_ids(engine.build_segments(raw))
    with open(os.path.join(job_dir, 'segments.json'), 'w', encoding='utf-8') as f:
        json.dump({'version': 1, 'segments': segs}, f)
    return job_dir, vod, segs


def _candidate(start_id='S00010', end_id='S00013', quote='frase numero 10 del stream con gente', **extra):
    base = {'start_id': start_id, 'end_id': end_id, 'quote': quote,
            'criteria': {'interest': 9, 'clarity': 8, 'emotion': 7, 'originality': 6,
                         'retention': 8, 'resolution': 9, 'start_quality': 8, 'end_quality': 8},
            'type': 'historia', 'hook': 'h', 'reason': 'r', 'tiktok_title': 'Titulo',
            'tiktok_description': 'd', 'call_to_action': 'c'}
    base.update(extra)
    return base


def _gpt_candidates(*cands):
    return _gpt(json.dumps({'candidates': list(cands)}))


def test_detect_rejects_context_outside_jobs(env):
    r = env['client'].post('/detect_moments', json={'transcript': TRANSCRIPT, 'vod_file': '/tmp/x.mp4'}, headers=HEADERS)
    assert r.status_code == 422


def test_detect_all_windows_invalid_json_is_502(env):
    _job_with_segments(env)
    env['monkeypatch'].setattr(requests, 'post', lambda *a, **k: _gpt('no es json'))
    vod = os.path.join(server.jobs_root(), 'a' * 32, 'vod.mp4')
    r = env['client'].post('/detect_moments', json={'vod_file': vod}, headers=HEADERS)
    assert r.status_code == 502 and r.get_json()['error_code'] == 'detection_failed'


def test_detect_resolves_ids_to_real_times_and_keeps_fields_for_n8n(env):
    _, vod, segs = _job_with_segments(env)
    env['monkeypatch'].setattr(requests, 'post', lambda *a, **k: _gpt_candidates(_candidate()))
    r = env['client'].post('/detect_moments', json={
        'vod_file': vod, 'channel': 'jaap-z', 'clip_duration': 55, 'open_ai_key': SECRET,
    }, headers=HEADERS)
    assert r.status_code == 200
    moment = r.get_json()['moments'][0]
    # S00010 empieza en 50 s, con 1 s de margen
    assert moment['start_seconds'] == 49.0
    # S00013 termina en 69 s, con 1 s de margen
    assert moment['end_seconds'] == 70.0
    assert moment['duration_seconds'] == round(moment['end_seconds'] - moment['start_seconds'], 2)
    assert moment['timestamp_seconds'] == 59.0          # compatibilidad: start + 10
    assert moment['clip_duration'] == int(round(moment['duration_seconds']))
    assert moment['vod_file'] == vod
    assert moment['source_segment_ids'][0] == 'S00010'
    assert moment['rank'] == 1 and moment['score'] > 0
    assert 'open_ai_key' not in moment and SECRET not in r.get_data(as_text=True)


def test_detect_drops_hallucinated_quote(env):
    _, vod, _ = _job_with_segments(env)
    env['monkeypatch'].setattr(requests, 'post', lambda *a, **k: _gpt_candidates(
        _candidate(quote='frase inventada que nunca se dijo jamas en el stream')))
    r = env['client'].post('/detect_moments', json={'vod_file': vod}, headers=HEADERS)
    data = r.get_json()
    assert r.status_code == 200 and data['count'] == 0


def test_detect_drops_candidates_with_unknown_ids(env):
    _, vod, _ = _job_with_segments(env)
    env['monkeypatch'].setattr(requests, 'post', lambda *a, **k: _gpt_candidates(
        _candidate(start_id='S99999', end_id='S99999')))
    r = env['client'].post('/detect_moments', json={'vod_file': vod}, headers=HEADERS)
    assert r.status_code == 200 and r.get_json()['count'] == 0


def test_detect_falls_back_to_transcript_lines_without_segments_file(env):
    job_dir, vod = make_job(env['files'])
    lines = '\n'.join(f'[{(i * 5) // 60}:{(i * 5) % 60:02d}] frase numero {i} del stream con gente hablando'
                      for i in range(40))
    env['monkeypatch'].setattr(requests, 'post', lambda *a, **k: _gpt_candidates(_candidate()))
    r = env['client'].post('/detect_moments', json={'vod_file': vod, 'transcript': lines}, headers=HEADERS)
    assert r.status_code == 200 and r.get_json()['count'] == 1


def test_detect_writes_candidates_audit_file_in_job(env):
    job_dir, vod, _ = _job_with_segments(env)
    env['monkeypatch'].setattr(requests, 'post', lambda *a, **k: _gpt_candidates(_candidate()))
    env['client'].post('/detect_moments', json={'vod_file': vod}, headers=HEADERS)
    assert os.path.isfile(os.path.join(job_dir, 'candidates.json'))
    assert os.path.isfile(os.path.join(job_dir, 'report.txt'))


def test_transcribe_writes_segments_file_with_ids(env):
    audio_dir = _audio_dir(env)
    seq = [_ok_whisper([{'start': 1.0, 'end': 3.0, 'text': ' hola '}]),
           _ok_whisper([{'start': 0.5, 'end': 2.0, 'text': 'chau'}])]
    env['monkeypatch'].setattr(requests, 'post', lambda *a, **k: seq.pop(0))
    r = env['client'].post('/transcribe', json=_transcribe_body(audio_dir), headers=HEADERS)
    data = r.get_json()
    assert r.status_code == 200 and data['segments_count'] == 2
    with open(data['segments_file'], encoding='utf-8') as f:
        saved = json.load(f)['segments']
    assert [s['id'] for s in saved] == ['S00000', 'S00001']
    assert saved[1]['start'] == 900.5  # offset de 15 min (900 s) del segundo chunk


# ---------------------------------------------------------------------------
# Clips: limites de VOD, duraciones y nombres unicos
# ---------------------------------------------------------------------------
def test_clip_rejects_start_outside_vod_without_running_ffmpeg(env):
    env['state']['vod_duration'] = 100.0
    vod = _job_vod(env)
    r = env['client'].post('/clip', json={'input': vod, 'start': 500, 'duration': 55}, headers=HEADERS)
    assert r.status_code == 422
    assert not any(c[0] == 'ffmpeg' for c in env['state']['calls'])


def test_clip_truncates_duration_to_end_of_vod(env):
    env['state']['vod_duration'] = 100.0
    vod = _job_vod(env)
    r = env['client'].post('/clip', json={'input': vod, 'start': 60, 'duration': 55}, headers=HEADERS)
    assert r.status_code == 200
    ffmpeg_cmd = [c for c in env['state']['calls'] if c[0] == 'ffmpeg'][-1]
    assert float(ffmpeg_cmd[ffmpeg_cmd.index('-t') + 1]) == 40.0


def test_clip_names_are_unique_and_never_overwrite(env):
    vod = _job_vod(env)
    a = env['client'].post('/clip', json={'input': vod, 'start': 60, 'duration': 55}, headers=HEADERS).get_json()
    b = env['client'].post('/clip', json={'input': vod, 'start': 60, 'duration': 55}, headers=HEADERS).get_json()
    assert a['file'] != b['file']
    ffmpeg_cmds = [c for c in env['state']['calls'] if c[0] == 'ffmpeg']
    assert all('-n' in c for c in ffmpeg_cmds)  # ffmpeg no sobrescribe


def test_clip_empty_output_is_removed_and_reported(env):
    vod = _job_vod(env)
    env['state']['clip_out'] = 0.0
    r = env['client'].post('/clip', json={'input': vod, 'start': 60, 'duration': 55}, headers=HEADERS)
    assert r.status_code == 422
    clip_dir = os.path.join(os.path.dirname(vod), 'clips')
    assert not os.listdir(clip_dir)


def test_clip_ffmpeg_failure_is_502(env):
    vod = _job_vod(env)
    original = env['state']
    real_run = server.subprocess.run

    def failing(cmd, **kw):
        if cmd[0] == 'ffmpeg':
            return subprocess.CompletedProcess(cmd, 1, stdout='', stderr='boom')
        return real_run(cmd, **kw)
    env['monkeypatch'].setattr(server.subprocess, 'run', failing)
    r = env['client'].post('/clip', json={'input': vod, 'start': 60, 'duration': 55}, headers=HEADERS)
    assert r.status_code == 502 and original is env['state']


def test_clip_rejects_non_numeric_start(env):
    vod = _job_vod(env)
    r = env['client'].post('/clip', json={'input': vod, 'start': 'abc'}, headers=HEADERS)
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Entrada de video y resolucion de URL
# ---------------------------------------------------------------------------
def test_save_file_rejects_unknown_extension(env):
    import io
    data = {'file': (io.BytesIO(b'abc'), 'payload.sh')}
    r = env['client'].post('/save_file', data=data, headers=HEADERS, content_type='multipart/form-data')
    assert r.status_code == 422


def test_save_file_removes_job_when_video_invalid(env):
    import io
    env['state']['probe_fail'] = True
    data = {'file': (io.BytesIO(b'abc'), 'video.mp4')}
    r = env['client'].post('/save_file', data=data, headers=HEADERS, content_type='multipart/form-data')
    assert r.status_code == 422 and not os.listdir(os.path.join(env['files'], 'jobs'))


def test_save_file_success_returns_job_contract(env):
    import io
    data = {'file': (io.BytesIO(b'abc'), 'video drive.mp4'), 'open_ai_key': SECRET, 'chunk_minutes': '15'}
    r = env['client'].post('/save_file', data=data, headers=HEADERS, content_type='multipart/form-data')
    body = r.get_json()
    assert r.status_code == 200
    assert {'file', 'job_id', 'duration_seconds', 'source_name'} <= set(body)
    assert 'open_ai_key' not in body and SECRET not in r.get_data(as_text=True)


def test_resolve_url_rejects_non_kick_domain(env):
    r = env['client'].post('/resolve_kick_url', json={'url': 'https://evil.com/x'}, headers=HEADERS)
    assert r.status_code == 422


def test_resolve_url_returns_hls_and_passthrough_without_url(env):
    r = env['client'].post('/resolve_kick_url', json={'url': VALID_KICK_URL, 'channel': 'jaap-z'}, headers=HEADERS)
    body = r.get_json()
    assert r.status_code == 200 and body['hls_url'].endswith('1.m3u8?sig=abc')
    assert body['channel'] == 'jaap-z' and 'url' not in body


# ---------------------------------------------------------------------------
# Contratos: claves que n8n espera
# ---------------------------------------------------------------------------
def test_download_contract_keys(env):
    body = env['client'].post('/download', json={'url': VALID_KICK_URL}, headers=HEADERS).get_json()
    assert {'file', 'job_id', 'duration_seconds'} <= set(body)


def test_extract_contract_keys(env):
    vod = _job_vod(env)
    body = env['client'].post('/extract_audio', json={'input': vod}, headers=HEADERS).get_json()
    assert {'dir', 'chunks', 'vod_file', 'job_id'} <= set(body)


def test_clip_contract_keys(env):
    vod = _job_vod(env)
    body = env['client'].post('/clip', json={'input': vod, 'start': 60, 'duration': 55}, headers=HEADERS).get_json()
    assert {'file', 'job_id', 'start_seconds', 'duration_seconds'} <= set(body)


def test_no_secret_leaks_in_any_success_response(env):
    vod = _job_vod(env)
    responses = [
        env['client'].post('/download', json={'url': VALID_KICK_URL, 'open_ai_key': SECRET}, headers=HEADERS),
        env['client'].post('/extract_audio', json={'input': vod, 'api_key': SECRET}, headers=HEADERS),
        env['client'].post('/clip', json={'input': vod, 'start': 1, 'open_ai_key': SECRET}, headers=HEADERS),
    ]
    for r in responses:
        assert SECRET not in r.get_data(as_text=True)
