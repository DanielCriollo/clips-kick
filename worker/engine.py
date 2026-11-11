"""Motor de seleccion y recorte de clips (Mision 003).

Modulo puro: no hace red, ni ffmpeg, ni escritura en disco. Todo lo que
toca el exterior vive en server.py. Esto permite probar cada regla con
pruebas unitarias y sin consumir creditos.

Principios:
  - GPT no calcula segundos. Solo devuelve IDs de segmentos (S00042).
  - El backend resuelve start/end desde los segmentos reales de Whisper.
  - Las citas se validan por tokens completos, no por subcadenas.
  - La puntuacion usa criterios explicitos. No predice viralidad.
"""
import json
import re
import statistics
import unicodedata

# ---------------------------------------------------------------------------
# Parametros (ajustables; ver INFORME-M003.md)
# ---------------------------------------------------------------------------
WINDOW_SECONDS = 15 * 60       # ventana de analisis
OVERLAP_SECONDS = 3 * 60       # solapamiento entre ventanas consecutivas
MIN_CLIP_SECONDS = 15.0
MAX_CLIP_SECONDS = 120.0
MAX_GAP_SECONDS = 4.0          # hueco maximo para considerar dos segmentos contiguos
PRE_PAD_SECONDS = 1.0          # margen antes del primer segmento de la historia
POST_PAD_SECONDS = 1.0         # margen despues del ultimo segmento
MIN_SCORE = 5.0                # por debajo de esto el candidato se descarta
MIN_RESOLUTION = 5.0           # criterio "desenlace": por debajo se descarta como incompleto
QUOTE_MIN_TOKEN_RATIO = 0.6    # fraccion de tokens de la cita que debe aparecer en el rango
DUPLICATE_IOU = 0.5            # solapamiento de union para considerar el mismo evento
DUPLICATE_OVERLAP_MIN = 0.7    # solapamiento sobre el mas corto para considerar el mismo evento

# Pesos explicitos de la puntuacion (suman 1.0).
WEIGHTS = {
    'interest': 0.25,        # interes del momento
    'clarity': 0.10,         # se entiende sin contexto externo
    'emotion': 0.15,         # intensidad emocional
    'originality': 0.10,     # no es un tema generico
    'retention': 0.15,       # potencial de mantener la atencion
    'resolution': 0.10,      # existe un desenlace
    'start_quality': 0.075,  # el inicio engancha
    'end_quality': 0.075,    # el cierre es limpio
}

CANDIDATE_TYPES = {
    'historia', 'broma', 'reaccion', 'discusion', 'sorpresa',
    'emocion', 'chat', 'frase', 'anuncio', 'otro',
}
SEGMENT_ID_RE = re.compile(r'^S\d{5}$')


class CandidateError(ValueError):
    """Candidato rechazado. El mensaje se registra como motivo de descarte."""


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def fmt_time(seconds):
    seconds = max(0, int(seconds))
    return f'{seconds // 60}:{seconds % 60:02d}'


def normalize_words(text):
    """Tokens en minusculas, sin acentos ni signos. 'Casarse!' -> ['casarse']."""
    text = unicodedata.normalize('NFKD', str(text or '').lower())
    text = ''.join(c for c in text if not unicodedata.combining(c))
    return re.findall(r'[a-z0-9]+', text)


# ---------------------------------------------------------------------------
# 1. Segmentos con identificadores estables
# ---------------------------------------------------------------------------
def build_segments(raw_segments, offset_seconds=0.0):
    """Convierte segmentos crudos de Whisper en segmentos con ID.

    raw_segments: lista de dicts con 'start', 'end' (opcional) y 'text'.
    offset_seconds: desplazamiento del chunk respecto al inicio del VOD.
    Los IDs se asignan al final, sobre la lista completa y ordenada.
    """
    items = []
    for seg in raw_segments:
        if not isinstance(seg, dict):
            continue
        text = ' '.join(str(seg.get('text', '')).split())
        if not text:
            continue
        try:
            start = float(seg['start']) + offset_seconds
        except (KeyError, TypeError, ValueError):
            continue
        try:
            end = float(seg['end']) + offset_seconds
        except (KeyError, TypeError, ValueError):
            end = start
        items.append({'start': round(start, 3), 'end': round(max(end, start), 3), 'text': text})
    return items


def assign_segment_ids(segments):
    """Ordena por inicio y asigna S00000, S00001, ... (sin huecos)."""
    ordered = sorted(segments, key=lambda s: (s['start'], s['end']))
    for i, seg in enumerate(ordered):
        seg['id'] = f'S{i:05d}'
        seg['index'] = i
    return ordered


def index_by_id(segments):
    return {s['id']: s for s in segments}


# ---------------------------------------------------------------------------
# 2. Ventanas solapadas
# ---------------------------------------------------------------------------
def build_windows(segments, window=WINDOW_SECONDS, overlap=OVERLAP_SECONDS):
    """Ventanas por tiempo de inicio del segmento, con solapamiento.

    Un segmento aparece en todas las ventanas que contienen su inicio. Un
    momento que cruza una frontera queda completo en al menos una ventana.
    """
    if not segments:
        return []
    if overlap >= window:
        raise ValueError('overlap debe ser menor que window')
    total_end = max(s['end'] for s in segments)
    step = window - overlap
    windows = []
    t = 0.0
    while t < total_end:
        lo, hi = t, t + window
        members = [s for s in segments if lo <= s['start'] < hi]
        if members:
            windows.append({'start': lo, 'end': hi, 'segment_ids': [s['id'] for s in members]})
        t += step
    return windows


def format_window_for_prompt(window, segments_by_id):
    lines = []
    for sid in window['segment_ids']:
        seg = segments_by_id[sid]
        lines.append(f"[{sid} | {fmt_time(seg['start'])}] {seg['text']}")
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 3. Contrato de salida del modelo
# ---------------------------------------------------------------------------
def parse_model_content(content):
    """Parsea la respuesta de GPT. Lanza CandidateError si no cumple el contrato."""
    text = (content or '').strip()
    if text.startswith('```'):
        text = '\n'.join(l for l in text.split('\n') if not l.startswith('```')).strip()
    try:
        parsed = json.loads(text)
    except ValueError:
        raise CandidateError('respuesta del modelo no es JSON valido')
    candidates = parsed.get('candidates') if isinstance(parsed, dict) else None
    if not isinstance(candidates, list):
        raise CandidateError('respuesta del modelo sin lista "candidates"')
    return candidates


# ---------------------------------------------------------------------------
# 4. Resolucion de un candidato a rango temporal
# ---------------------------------------------------------------------------
def resolve_candidate(raw, segments_by_id, vod_duration=None,
                      quote_min_ratio=QUOTE_MIN_TOKEN_RATIO):
    """Convierte IDs de segmento en start/end reales. Lanza CandidateError si no es valido."""
    if not isinstance(raw, dict):
        raise CandidateError('candidato no es un objeto')

    start_id = raw.get('start_id')
    end_id = raw.get('end_id')
    for sid in (start_id, end_id):
        if not isinstance(sid, str) or not SEGMENT_ID_RE.match(sid):
            raise CandidateError('ID de segmento con formato invalido')
        if sid not in segments_by_id:
            raise CandidateError('ID de segmento inexistente')

    first = segments_by_id[start_id]
    last = segments_by_id[end_id]
    if last['index'] < first['index']:
        raise CandidateError('end_id anterior a start_id')

    # Rango de segmentos incluido: start_id .. end_id
    range_segments = [s for s in segments_by_id.values()
                      if first['index'] <= s['index'] <= last['index']]
    range_segments.sort(key=lambda s: s['index'])

    start = max(0.0, first['start'] - PRE_PAD_SECONDS)
    end = last['end'] + POST_PAD_SECONDS

    # Clip demasiado corto: extiende solo por segmentos CONTIGUOS (hueco <= MAX_GAP_SECONDS)
    if end - start < MIN_CLIP_SECONDS:
        previous_end = last['end']
        for seg in sorted(segments_by_id.values(), key=lambda s: s['index']):
            if seg['index'] <= last['index']:
                continue
            if seg['start'] - previous_end > MAX_GAP_SECONDS:
                break
            end = seg['end'] + POST_PAD_SECONDS
            range_segments.append(seg)
            previous_end = seg['end']
            if end - start >= MIN_CLIP_SECONDS:
                break

    # Clip demasiado largo: recorta al ultimo segmento que cabe en el maximo
    if end - start > MAX_CLIP_SECONDS:
        kept = [s for s in range_segments if s['end'] + POST_PAD_SECONDS - start <= MAX_CLIP_SECONDS]
        if not kept:
            raise CandidateError('primer segmento ya excede la duracion maxima')
        range_segments = kept
        end = kept[-1]['end'] + POST_PAD_SECONDS

    if vod_duration is not None:
        if start >= vod_duration:
            raise CandidateError('el inicio cae fuera del VOD')
        end = min(end, vod_duration)
    if end - start < 1.0:
        raise CandidateError('clip resultante demasiado corto')

    quote = raw.get('quote', '')
    if not isinstance(quote, str) or not quote.strip():
        raise CandidateError('falta la cita textual')
    if not quote_is_supported(quote, range_segments, quote_min_ratio):
        raise CandidateError('la cita no aparece en los segmentos indicados')

    return {
        'start_seconds': round(start, 2),
        'end_seconds': round(end, 2),
        'duration_seconds': round(end - start, 2),
        'source_segment_ids': [s['id'] for s in range_segments],
        'first_index': first['index'],
        'last_index': last['index'],
    }


def quote_is_supported(quote, range_segments, min_ratio=QUOTE_MIN_TOKEN_RATIO):
    """Compara tokens completos. 'casa' no coincide con 'casarse'."""
    tokens = [t for t in normalize_words(quote) if len(t) > 2]
    if len(tokens) < 3:
        return False
    vocab = set()
    for seg in range_segments:
        vocab.update(normalize_words(seg['text']))
    matched = sum(1 for t in tokens if t in vocab)
    return matched / len(tokens) >= min_ratio


# ---------------------------------------------------------------------------
# 5. Puntuacion con criterios explicitos
# ---------------------------------------------------------------------------
def score_candidate(criteria):
    """Media ponderada de criterios 0-10. Valores ausentes cuentan como 0."""
    if not isinstance(criteria, dict):
        criteria = {}
    total = 0.0
    for key, weight in WEIGHTS.items():
        try:
            value = float(criteria.get(key, 0))
        except (TypeError, ValueError):
            value = 0.0
        value = max(0.0, min(10.0, value))
        total += weight * value
    return round(total, 2)


# ---------------------------------------------------------------------------
# 6. Duplicados reales y ranking global
# ---------------------------------------------------------------------------
def _overlap(a, b):
    inter = max(0.0, min(a['end_seconds'], b['end_seconds']) - max(a['start_seconds'], b['start_seconds']))
    union = max(a['end_seconds'], b['end_seconds']) - min(a['start_seconds'], b['start_seconds'])
    shorter = min(a['duration_seconds'], b['duration_seconds'])
    iou = inter / union if union > 0 else 0.0
    over_short = inter / shorter if shorter > 0 else 0.0
    return iou, over_short


def is_same_event(a, b):
    iou, over_short = _overlap(a, b)
    return iou >= DUPLICATE_IOU or over_short >= DUPLICATE_OVERLAP_MIN


def dedupe_events(candidates):
    """Conserva el de mayor puntuacion entre candidatos que describen el mismo evento.

    Eventos cercanos pero distintos (sin solapamiento significativo) se conservan.
    """
    kept = []
    for cand in sorted(candidates, key=lambda c: (-c['score'], c['start_seconds'])):
        if any(is_same_event(cand, k) for k in kept):
            continue
        kept.append(cand)
    return kept


def process_candidates(raw_candidates, segments, vod_duration=None, max_moments=50,
                       min_score=MIN_SCORE, min_resolution=MIN_RESOLUTION):
    """Pipeline completo sin I/O.

    Devuelve (seleccionados_ordenados_por_rank, descartes), donde descartes es
    una lista de {'reason': str} para auditoria.
    """
    by_id = index_by_id(segments)
    accepted, discards = [], []

    for raw in raw_candidates:
        try:
            resolved = resolve_candidate(raw, by_id, vod_duration)
        except CandidateError as exc:
            discards.append({'reason': str(exc)})
            continue

        criteria = raw.get('criteria') if isinstance(raw.get('criteria'), dict) else {}
        if float(criteria.get('resolution', 0) or 0) < min_resolution:
            discards.append({'reason': 'incompleto: sin desenlace'})
            continue

        score = score_candidate(criteria)
        if score < min_score:
            discards.append({'reason': f'puntuacion baja ({score})'})
            continue

        ctype = str(raw.get('type', 'otro')).lower()
        if ctype not in CANDIDATE_TYPES:
            ctype = 'otro'

        accepted.append({
            **resolved,
            'score': score,
            'criteria': {k: float(criteria.get(k, 0) or 0) for k in WEIGHTS},
            'type': ctype,
            'hook': str(raw.get('hook', ''))[:200],
            'reason': str(raw.get('reason', ''))[:400],
            'quote': str(raw.get('quote', ''))[:300],
            'tiktok_title': str(raw.get('tiktok_title', ''))[:120],
            'tiktok_description': str(raw.get('tiktok_description', ''))[:300],
            'call_to_action': str(raw.get('call_to_action', ''))[:120],
        })

    unique = dedupe_events(accepted)
    dup_count = len(accepted) - len(unique)
    if dup_count:
        discards.append({'reason': f'duplicados del mismo evento: {dup_count}'})

    selected = unique[:max(1, int(max_moments))]
    for rank, cand in enumerate(selected, 1):
        cand['rank'] = rank
    return selected, discards


# ---------------------------------------------------------------------------
# 7. Silencios (helper; no activado en M003.1 por costo de decodificacion)
# ---------------------------------------------------------------------------
_SILENCE_START = re.compile(r'silence_start: (-?[\d.]+)')
_SILENCE_END = re.compile(r'silence_end: (-?[\d.]+)')


def parse_silences(ffmpeg_stderr):
    """Extrae pares (inicio, fin) de la salida de ffmpeg silencedetect."""
    starts = [float(m.group(1)) for m in _SILENCE_START.finditer(ffmpeg_stderr or '')]
    ends = [float(m.group(1)) for m in _SILENCE_END.finditer(ffmpeg_stderr or '')]
    return list(zip(starts, ends))


def snap_end_to_silence(end, silences, min_end, max_shift=2.0):
    """Mueve el final al centro de un silencio cercano, sin cortar antes de min_end."""
    best = None
    for s, e in silences:
        mid = (s + e) / 2.0
        if mid < min_end:
            continue
        if abs(mid - end) <= max_shift and (best is None or abs(mid - end) < abs(best - end)):
            best = mid
    return best if best is not None else end


# ---------------------------------------------------------------------------
# 8. Metricas de evaluacion (sobre etiquetas manuales)
# ---------------------------------------------------------------------------
def evaluate(predictions, labels, match_ratio=0.5):
    """Compara predicciones con eventos etiquetados como buenos.

    predictions: lista de dicts con start_seconds, end_seconds.
    labels: lista de dicts con start, end (eventos buenos etiquetados a mano).
    Un evento se considera cubierto si una prediccion cubre >= match_ratio de su duracion.
    Una prediccion es correcta si cubre >= match_ratio de la duracion de algun evento.
    """
    def cover(pred, lab):
        inter = max(0.0, min(pred['end_seconds'], lab['end']) - max(pred['start_seconds'], lab['start']))
        lab_dur = lab['end'] - lab['start']
        return inter / lab_dur if lab_dur > 0 else 0.0

    covered_labels = set()
    correct_preds = 0
    start_errors, end_errors = [], []
    for pred in predictions:
        best_lab, best_cov = None, 0.0
        for j, lab in enumerate(labels):
            c = cover(pred, lab)
            if c > best_cov:
                best_lab, best_cov = j, c
        if best_lab is not None and best_cov >= match_ratio:
            correct_preds += 1
            covered_labels.add(best_lab)
            start_errors.append(abs(pred['start_seconds'] - labels[best_lab]['start']))
            end_errors.append(abs(pred['end_seconds'] - labels[best_lab]['end']))

    total_labels = len(labels)
    total_preds = len(predictions)
    return {
        'labels_total': total_labels,
        'predictions_total': total_preds,
        'coverage': round(len(covered_labels) / total_labels, 3) if total_labels else None,
        'precision': round(correct_preds / total_preds, 3) if total_preds else None,
        'start_error_median_s': round(statistics.median(start_errors), 2) if start_errors else None,
        'end_error_median_s': round(statistics.median(end_errors), 2) if end_errors else None,
    }


# ---------------------------------------------------------------------------
# 9. Prompts (version M003.1)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_TEMPLATE = (
    'Eres un editor experto de clips para TikTok, Reels y Shorts de streams de {channel}, '
    'en espanol latinoamericano.\n\n'
    'Te doy un fragmento de transcripcion. Cada linea empieza con un identificador de segmento '
    '[S00042 | 45:30]. Tu tarea es encontrar TODOS los momentos que valgan un clip, aunque el '
    'fragmento tenga muchos. Puede haber varios distintos en el mismo fragmento.\n\n'
    'TIPOS: historia, broma, reaccion, discusion, sorpresa, emocion, chat, frase, anuncio, otro.\n\n'
    'REGLAS ABSOLUTAS:\n'
    '1. NO calcules segundos ni minutos. Solo devuelve start_id y end_id, copiados EXACTAMENTE '
    'de la transcripcion.\n'
    '2. start_id es el primer segmento del momento (el gancho). end_id es el ultimo segmento que '
    'cierra el momento. El momento debe tener inicio, desarrollo y desenlace.\n'
    '3. "quote" debe ser una frase que aparezca literalmente en los segmentos entre start_id y end_id.\n'
    '4. NO inventes contenido. Si algo no esta escrito, no existe.\n'
    '5. Si un momento es repetitivo, incompleto o flojo, no lo incluyas.\n'
    '6. Un mismo evento no se repite con titulos distintos. Eventos distintos sí.\n'
    '7. Puntua cada criterio de 0 a 10: interest, clarity, emotion, originality, retention, '
    'resolution (¿hay desenlace?), start_quality, end_quality. No afirmes que la puntuacion '
    'predice viralidad.\n'
    '8. Devuelve UNICAMENTE JSON valido, sin texto adicional.'
)

USER_PROMPT_TEMPLATE = (
    'Fragmento del stream de {channel}, desde el minuto {start_min} hasta el {end_min}.\n\n'
    '{transcript}\n\n'
    'Responde SOLO con este JSON:\n'
    '{{"candidates":[{{"start_id":"S00042","end_id":"S00047","type":"historia",'
    '"hook":"frase de enganche en una linea","reason":"por que vale un clip",'
    '"quote":"frase literal de la transcripcion",'
    '"tiktok_title":"titulo con emoji","tiktok_description":"2 lineas",'
    '"call_to_action":"frase corta",'
    '"criteria":{{"interest":8,"clarity":7,"emotion":6,"originality":5,"retention":7,'
    '"resolution":8,"start_quality":7,"end_quality":8}}}}]}}'
)


def build_prompts(channel, window, segments_by_id):
    transcript = format_window_for_prompt(window, segments_by_id)
    system = SYSTEM_PROMPT_TEMPLATE.format(channel=channel)
    user = USER_PROMPT_TEMPLATE.format(
        channel=channel,
        start_min=int(window['start'] // 60),
        end_min=int(window['end'] // 60),
        transcript=transcript,
    )
    return system, user
