"""Pruebas del motor de seleccion (Mision 003).

IMPORTANTE: todos los datos son sinteticos y prueban REGLAS (IDs, ventanas,
duplicados, duraciones, ranking). NO son una medicion de calidad real de los
clips. La calidad se mide con tools/evaluate.py sobre VODs etiquetados a mano.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'worker'))

import engine  # noqa: E402


def make_segments(n, step=5.0, length=4.0, text='hola que tal como estas hoy amigo'):
    raw = [{'start': i * step, 'end': i * step + length, 'text': f'{text} {i}'} for i in range(n)]
    return engine.assign_segment_ids(engine.build_segments(raw))


GOOD_CRITERIA = {'interest': 9, 'clarity': 8, 'emotion': 7, 'originality': 6,
                 'retention': 8, 'resolution': 9, 'start_quality': 8, 'end_quality': 8}


def cand(start_id, end_id, quote, criteria=None, **extra):
    base = {'start_id': start_id, 'end_id': end_id, 'quote': quote,
            'criteria': criteria or dict(GOOD_CRITERIA), 'type': 'historia',
            'hook': 'h', 'reason': 'r', 'tiktok_title': 't'}
    base.update(extra)
    return base


# ---------------------------------------------------------------------------
# Segmentos e IDs
# ---------------------------------------------------------------------------
def test_ids_are_sequential_and_ordered_by_start():
    raw = [{'start': 10, 'end': 12, 'text': 'b'}, {'start': 0, 'end': 2, 'text': 'a'}]
    segs = engine.assign_segment_ids(engine.build_segments(raw))
    assert [s['id'] for s in segs] == ['S00000', 'S00001']
    assert segs[0]['text'] == 'a' and segs[0]['index'] == 0


def test_offset_applied_to_chunk_segments_without_precision_loss():
    segs = engine.build_segments([{'start': 1.234, 'end': 2.345, 'text': 'x'}], offset_seconds=900)
    assert segs[0]['start'] == 901.234 and segs[0]['end'] == 902.345


def test_empty_text_segments_are_dropped():
    segs = engine.build_segments([{'start': 0, 'end': 1, 'text': '   '}, {'start': 2, 'end': 3, 'text': 'ok'}])
    assert len(segs) == 1 and segs[0]['text'] == 'ok'


# ---------------------------------------------------------------------------
# Ventanas solapadas
# ---------------------------------------------------------------------------
def test_windows_overlap_so_boundary_segments_appear_twice():
    segs = engine.assign_segment_ids(engine.build_segments([
        {'start': 100.0, 'end': 104.0, 'text': 'antes'},
        {'start': 880.0, 'end': 884.0, 'text': 'frontera'},
    ]))
    windows = engine.build_windows(segs, window=900, overlap=180)
    frontera_id = next(s['id'] for s in segs if s['text'] == 'frontera')
    holders = [w for w in windows if frontera_id in w['segment_ids']]
    # el inicio 880 s cae en [0, 900) y en [720, 1620)
    assert len(holders) == 2


def test_windows_cover_every_segment_at_least_once():
    segs = make_segments(2000, step=3.0)
    windows = engine.build_windows(segs, window=300, overlap=60)
    covered = set(sid for w in windows for sid in w['segment_ids'])
    assert covered == {s['id'] for s in segs}


def test_overlap_must_be_smaller_than_window():
    try:
        engine.build_windows(make_segments(3), window=100, overlap=100)
    except ValueError:
        return
    raise AssertionError('debio rechazar overlap >= window')


# ---------------------------------------------------------------------------
# Resolucion de timestamps desde IDs (sin calculos de GPT)
# ---------------------------------------------------------------------------
def test_start_and_end_come_from_real_segments_with_padding():
    segs = make_segments(60, step=5.0, length=4.0)
    by_id = engine.index_by_id(segs)
    res = engine.resolve_candidate(cand('S00010', 'S00014', 'hola que tal como'), by_id, vod_duration=10000)
    assert res['start_seconds'] == round(50.0 - engine.PRE_PAD_SECONDS, 2)
    # S00014 empieza en 70 s y termina en 74 s
    assert res['end_seconds'] == round(74.0 + engine.POST_PAD_SECONDS, 2)
    assert res['source_segment_ids'] == ['S00010', 'S00011', 'S00012', 'S00013', 'S00014']


def test_unknown_or_malformed_ids_are_rejected():
    by_id = engine.index_by_id(make_segments(10))
    for bad in ('S99999', 'segmento3', '42', None):
        try:
            engine.resolve_candidate(cand(bad, 'S00003', 'hola que tal como'), by_id)
        except engine.CandidateError:
            continue
        raise AssertionError(f'debio rechazar {bad!r}')


def test_reversed_range_is_rejected():
    by_id = engine.index_by_id(make_segments(10))
    try:
        engine.resolve_candidate(cand('S00005', 'S00002', 'hola que tal como'), by_id)
    except engine.CandidateError as e:
        assert 'anterior' in str(e)
        return
    raise AssertionError('debio rechazar rango invertido')


def test_clip_never_starts_or_ends_outside_vod():
    segs = make_segments(10, step=5.0)
    by_id = engine.index_by_id(segs)
    res = engine.resolve_candidate(cand('S00008', 'S00009', 'hola que tal como'), by_id, vod_duration=44.0)
    assert res['end_seconds'] <= 44.0
    try:
        engine.resolve_candidate(cand('S00009', 'S00009', 'hola que tal como'), by_id, vod_duration=10.0)
    except engine.CandidateError:
        return
    raise AssertionError('debio rechazar inicio fuera del VOD')


def test_short_moment_is_extended_to_minimum_duration():
    segs = make_segments(40, step=2.0, length=1.0)
    by_id = engine.index_by_id(segs)
    res = engine.resolve_candidate(cand('S00000', 'S00000', 'hola que tal como'), by_id)
    assert res['duration_seconds'] >= engine.MIN_CLIP_SECONDS
    assert len(res['source_segment_ids']) > 1


def test_long_moment_is_trimmed_to_maximum_duration_on_segment_boundary():
    segs = make_segments(200, step=5.0, length=4.0)
    by_id = engine.index_by_id(segs)
    res = engine.resolve_candidate(cand('S00000', 'S00150', 'hola que tal como'), by_id)
    assert res['duration_seconds'] <= engine.MAX_CLIP_SECONDS
    assert res['end_seconds'] == round(by_id[res['source_segment_ids'][-1]]['end'] + engine.POST_PAD_SECONDS, 2)


def test_dynamic_durations_differ_by_content():
    segs = make_segments(200, step=5.0, length=4.0)
    by_id = engine.index_by_id(segs)
    short = engine.resolve_candidate(cand('S00010', 'S00011', 'hola que tal como'), by_id)
    longer = engine.resolve_candidate(cand('S00010', 'S00025', 'hola que tal como'), by_id)
    assert short['duration_seconds'] != longer['duration_seconds']


# ---------------------------------------------------------------------------
# Citas: tokens completos, no subcadenas
# ---------------------------------------------------------------------------
def test_quote_must_be_present_and_nonempty():
    by_id = engine.index_by_id(make_segments(10))
    for quote in ('', '   ', None):
        try:
            engine.resolve_candidate(cand('S00001', 'S00002', quote), by_id)
        except engine.CandidateError:
            continue
        raise AssertionError(f'debio rechazar cita {quote!r}')


def test_substring_false_positive_is_rejected():
    segs = engine.assign_segment_ids(engine.build_segments([
        {'start': 0, 'end': 4, 'text': 'me voy a casarse el proximo mes'},
        {'start': 5, 'end': 9, 'text': 'nada que ver con eso hoy'},
    ]))
    by_id = engine.index_by_id(segs)
    # "casa" es subcadena de "casarse" pero no es el mismo token
    assert not engine.quote_is_supported('la casa grande de la playa', segs, 0.6)
    assert engine.quote_is_supported('me voy a casarse el proximo mes', segs, 0.6)


def test_quote_from_distant_segment_is_rejected():
    # El segmento con la cita esta a 46 s: no es contiguo, no entra en el clip.
    segs = engine.assign_segment_ids(engine.build_segments([
        {'start': 0, 'end': 4, 'text': 'hablamos del juego de ayer'},
        {'start': 50, 'end': 54, 'text': 'ahora de la musica nueva'},
    ]))
    by_id = engine.index_by_id(segs)
    try:
        engine.resolve_candidate(cand('S00000', 'S00000', 'la musica nueva es increible'), by_id)
    except engine.CandidateError as e:
        assert 'cita' in str(e)
        return
    raise AssertionError('la cita de otro segmento debio rechazarse')


# ---------------------------------------------------------------------------
# Puntuacion y descartes
# ---------------------------------------------------------------------------
def test_weights_sum_to_one():
    assert abs(sum(engine.WEIGHTS.values()) - 1.0) < 1e-9


def test_score_uses_weighted_mean_and_clamps_values():
    assert engine.score_candidate({k: 10 for k in engine.WEIGHTS}) == 10.0
    assert engine.score_candidate({k: 0 for k in engine.WEIGHTS}) == 0.0
    assert engine.score_candidate({'interest': 99}) == engine.WEIGHTS['interest'] * 10
    assert engine.score_candidate({'interest': 'x'}) == 0.0


def test_incomplete_moment_without_resolution_is_discarded():
    segs = make_segments(40)
    crit = dict(GOOD_CRITERIA, resolution=2)
    selected, discards = engine.process_candidates(
        [cand('S00010', 'S00014', 'hola que tal como', criteria=crit)], segs, max_moments=10)
    assert selected == [] and any('sin desenlace' in d['reason'] for d in discards)


def test_weak_score_is_discarded():
    segs = make_segments(40)
    weak = {k: 2 for k in engine.WEIGHTS}
    weak['resolution'] = 9  # hay desenlace, pero el resto es flojo
    selected, discards = engine.process_candidates(
        [cand('S00010', 'S00014', 'hola que tal como', criteria=weak)], segs, max_moments=10)
    assert selected == [] and any('puntuacion baja' in d['reason'] for d in discards)


def test_unknown_type_becomes_otro():
    segs = make_segments(40)
    selected, _ = engine.process_candidates(
        [cand('S00010', 'S00014', 'hola que tal como', type='inventado')], segs, max_moments=10)
    assert selected[0]['type'] == 'otro'


# ---------------------------------------------------------------------------
# Duplicados reales vs eventos distintos cercanos
# ---------------------------------------------------------------------------
def test_same_event_from_overlapping_windows_is_kept_once():
    segs = make_segments(120, step=5.0, length=4.0)
    a = cand('S00010', 'S00014', 'hola que tal como', tiktok_title='Titulo A')
    b = cand('S00011', 'S00015', 'hola que tal como', tiktok_title='Titulo B',
             criteria=dict(GOOD_CRITERIA, interest=6))
    selected, discards = engine.process_candidates([a, b], segs, max_moments=10)
    assert len(selected) == 1
    assert selected[0]['tiktok_title'] == 'Titulo A'  # conserva el de mayor puntuacion
    assert any('duplicados' in d['reason'] for d in discards)


def test_distinct_nearby_events_are_both_kept():
    segs = make_segments(200, step=5.0, length=4.0)
    first = cand('S00010', 'S00013', 'hola que tal como')
    second = cand('S00040', 'S00044', 'hola que tal como')  # ~150 s despues, sin solape
    selected, _ = engine.process_candidates([first, second], segs, max_moments=10)
    assert len(selected) == 2


def test_nearby_events_under_old_90s_rule_are_no_longer_merged():
    segs = make_segments(200, step=5.0, length=4.0)
    first = cand('S00100', 'S00102', 'hola que tal como')
    second = cand('S00112', 'S00114', 'hola que tal como')  # 60 s despues, sin solape
    selected, _ = engine.process_candidates([first, second], segs, max_moments=10)
    assert len(selected) == 2


# ---------------------------------------------------------------------------
# Ranking global y cupo
# ---------------------------------------------------------------------------
def test_selection_is_ranked_by_score_not_by_time():
    segs = make_segments(300, step=5.0, length=4.0)
    late_high = cand('S00200', 'S00203', 'hola que tal como', criteria=dict(GOOD_CRITERIA, interest=10))
    early_low = cand('S00010', 'S00013', 'hola que tal como', criteria=dict(GOOD_CRITERIA, interest=6))
    selected, _ = engine.process_candidates([early_low, late_high], segs, max_moments=10)
    assert selected[0]['start_seconds'] > selected[1]['start_seconds']
    assert [c['rank'] for c in selected] == [1, 2]


def test_max_moments_keeps_best_not_earliest():
    segs = make_segments(400, step=5.0, length=4.0)
    raws = []
    for k in range(10):
        start = 10 + k * 20
        raws.append(cand(f'S{start:05d}', f'S{start + 2:05d}', 'hola que tal como',
                         criteria=dict(GOOD_CRITERIA, interest=3 + k)))
    selected, _ = engine.process_candidates(raws, segs, max_moments=3)
    assert len(selected) == 3
    assert min(c['score'] for c in selected) > engine.score_candidate(dict(GOOD_CRITERIA, interest=3))


# ---------------------------------------------------------------------------
# Modelo: contrato y validacion
# ---------------------------------------------------------------------------
def test_model_response_must_be_json_with_candidates():
    for bad in ('no json', '{"moments": []}', '[]', '{"candidates": "x"}'):
        try:
            engine.parse_model_content(bad)
        except engine.CandidateError:
            continue
        raise AssertionError(f'debio rechazar {bad!r}')


def test_model_response_accepts_fenced_json():
    content = '```json\n{"candidates": []}\n```'
    assert engine.parse_model_content(content) == []


def test_prompts_never_ask_for_seconds():
    system, user = engine.build_prompts('jaap-z', {'start': 0, 'end': 900, 'segment_ids': ['S00000']},
                                        {'S00000': {'id': 'S00000', 'start': 1.0, 'end': 2.0, 'text': 'hola'}})
    assert 'NO calcules segundos' in system
    assert '"start_id"' in user and '"end_id"' in user
    assert 'timestamp_seconds' not in user


# ---------------------------------------------------------------------------
# Silencios (helper)
# ---------------------------------------------------------------------------
def test_parse_silences_from_ffmpeg_output():
    log = ('[silencedetect @ 0x1] silence_start: 12.5\n'
           '[silencedetect @ 0x1] silence_end: 14.0 | silence_duration: 1.5\n')
    assert engine.parse_silences(log) == [(12.5, 14.0)]


def test_snap_end_moves_to_nearby_silence_but_never_before_min_end():
    silences = [(20.0, 22.0), (40.0, 41.0)]
    assert engine.snap_end_to_silence(21.5, silences, min_end=0.0) == 21.0
    assert engine.snap_end_to_silence(21.5, silences, min_end=25.0) == 21.5  # 21 < min_end


# ---------------------------------------------------------------------------
# Evaluacion sobre etiquetas (la metrica, con datos sinteticos de ejemplo)
# ---------------------------------------------------------------------------
def test_evaluate_counts_coverage_precision_and_errors():
    labels = [{'start': 100, 'end': 160}, {'start': 500, 'end': 540}]
    preds = [
        {'start_seconds': 102, 'end_seconds': 158},  # cubre el evento 1
        {'start_seconds': 900, 'end_seconds': 950},  # falso positivo
    ]
    m = engine.evaluate(preds, labels)
    assert m['coverage'] == 0.5
    assert m['precision'] == 0.5
    assert m['start_error_median_s'] == 2.0
    assert m['end_error_median_s'] == 2.0


def test_evaluate_handles_empty_inputs():
    m = engine.evaluate([], [])
    assert m['coverage'] is None and m['precision'] is None
