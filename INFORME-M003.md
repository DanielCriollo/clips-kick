# MISIÓN 003 — Motor de selección y recorte de clips

**Estado:** implementado y probado con pruebas automatizadas en la copia de trabajo `clips-kick-m003/`. **No está desplegado, no tiene commits ni push, no consumió créditos de OpenAI.** La calidad real de los clips aún no está medida: eso requiere etiquetar un VOD real y correr `tools/evaluate.py`.

**Pruebas:** 97 pasan (`python -m pytest -q tests`): 64 del worker (`test_worker.py`: las 61 de M002, de las cuales 4 de detección se reescribieron para el contrato nuevo, más 3 nuevas) y 33 del motor (`test_engine.py`). Todas con datos sintéticos y mocks.

---

## 1. Qué mejora se espera

Lo que cambia y por qué:

| Problema anterior | Cambio en M003 | Mejora esperada |
|---|---|---|
| GPT calculaba segundos absolutos (`[45:30]` → `2730`), con errores de aritmética | GPT devuelve `start_id` y `end_id` (`S00042`). El backend toma inicio y fin de Whisper | Timestamps exactos al segmento. Error de minutos desaparece por diseño |
| Clips de 55 s que empezaban en `timestamp - 10` | Inicio = inicio del primer segmento − 1 s. Fin = fin del último segmento + 1 s. Duración entre 15 y 120 s | Menos frases cortadas y desenlaces incluidos |
| Momentos en el límite de 15 min se perdían | Ventanas de 15 min con 3 min de solapamiento | Momentos que cruzan una frontera quedan completos en una ventana |
| Cupo fijo por sección (`max(5, 50 // secciones)`) | Sin cupo por sección. GPT devuelve todos los que valgan la pena | Más cobertura en streams con mucho contenido |
| Deduplicación fija de 90 s (eliminaba eventos distintos cercanos) | Duplicado = mismo evento (IoU ≥ 0.5 o solapamiento ≥ 70% del más corto). Eventos distintos se conservan | Dos historias separadas por 60 s ya no se fusionan |
| Truncado cronológico antes del límite (perdía los últimos del stream) | Ranking global por puntuación, luego truncado | Se conservan los mejores, sin importar la hora |
| Validación de citas por subcadena (`casa` aceptaba `casarse`) | Tokens completos, con 60% de coincidencia dentro del rango del clip | Menos citas aceptadas por error |
| Sin criterios explícitos | 8 criterios con pesos explícitos (suman 1.0). Descarte por puntuación < 5 y por "sin desenlace" | Descartes auditables en `candidates.json` |

**Lo que no está garantizado:** que la puntuación prediga viralidad. No lo afirma ningún criterio del código. Los pesos son una decisión editorial, no un modelo entrenado.

## 2. Qué se ha verificado con pruebas

Todo esto está cubierto en `tests/test_engine.py` y `tests/test_worker.py`:

- **Timestamps correctos:** inicio y fin salen de los segmentos (`test_start_and_end_come_from_real_segments_with_padding`). Un ID inexistente o con formato inválido se rechaza (`test_unknown_or_malformed_ids_are_rejected`). Un rango invertido también.
- **Ausencia de cortes fuera del VOD:** `test_clip_never_starts_or_ends_outside_vod` y el recorte al final del VOD en `/clip`.
- **Detección en límites de fragmentos:** `test_windows_overlap_so_boundary_segments_appear_twice` y `test_windows_cover_every_segment_at_least_once`.
- **Eventos cercanos pero distintos:** `test_distinct_nearby_events_are_both_kept` y `test_nearby_events_under_old_90s_rule_are_no_longer_merged`.
- **Duplicados reales:** `test_same_event_from_overlapping_windows_is_kept_once` (conserva el de mayor puntuación).
- **Duraciones dinámicas:** `test_dynamic_durations_differ_by_content`, más extensión de clips cortos y recorte de largos (`test_long_moment_is_trimmed_to_maximum_duration_on_segment_boundary`).
- **Ordenamiento por calidad:** `test_selection_is_ranked_by_score_not_by_time` y `test_max_moments_keeps_best_not_earliest`.
- **Compatibilidad con n8n:** `test_detect_resolves_ids_to_real_times_and_keeps_fields_for_n8n` verifica `timestamp_seconds`, `clip_duration`, `vod_file` y `report_file`. Y `Parsear Momentos` se probó ejecutándolo con Node (ver sección 6).
- **Sin inventar contenido:** `test_detect_drops_hallucinated_quote` y `test_quote_from_distant_segment_is_rejected`.

**Un bug encontrado y corregido durante las pruebas:** al extender clips cortos, el código tomaba el siguiente segmento aunque estuviera a 46 s de distancia. Ahora solo extiende por segmentos contiguos (hueco ≤ 4 s). Lo detectó `test_quote_from_distant_segment_is_rejected`.

**Otro bug corregido:** un JSON de GPT sin la clave `candidates` causaba un `KeyError` no controlado. Ahora se rechaza como respuesta inválida y la ventana se registra como fallida.

## 3. Qué requiere validación con un VOD real

Ninguna de las siguientes mejoras está medida todavía:

- **Cobertura** (fracción de momentos buenos que el sistema encuentra).
- **Precisión** (fracción de clips seleccionados que son buenos).
- **Error de inicio y final** en segundos, medido contra momentos etiquetados a mano.
- **Si GPT-4o-mini distingue bien** historias completas de fragmentos sueltos. Esto depende del modelo, no del código. Un modelo más fuerte puede dar más mejora que cualquier cambio aquí, a más costo.

**Procedimiento:**
1. Corre el pipeline actual (v2) sobre un stream real y guarda la respuesta de `/detect_moments`.
2. Llena `tools/labels_template.json` con los momentos buenos que marques a mano en ese stream (tiempos en segundos desde el inicio del VOD).
3. Corre `python tools/evaluate.py --labels <archivo> --predictions <respuesta>`.
4. Repite con M003 sobre el mismo stream y compara los números.

**Limitación de la evaluación:** con pocas etiquetas el resultado tiene mucha varianza. Hacen falta al menos 2 o 3 streams para afirmar una mejora.

## 4. Contratos JSON

### `/transcribe` (cambio aditivo)
Campos existentes sin cambios: `transcript`, `transcript_file`, `audio_dir`, `job_id`, passthrough.
**Nuevos:**
- `segments_file`: ruta a `segments.json` dentro del job.
- `segments_count`: número de segmentos con ID.

`segments.json` contiene `{"version": 1, "chunk_minutes": N, "segments": [{"id": "S00000", "index": 0, "start": 1.234, "end": 3.456, "text": "..."}]}`. El tiempo es el del VOD completo (con el offset del chunk), sin redondear a segundos enteros.

### `/detect_moments` (cambio de contrato)
**Entrada:** igual que antes (`transcript`, `channel`, `max_moments`, `vod_file`, `audio_dir`). `max_moments` ahora acepta hasta 150 (antes 100). El defecto sigue siendo 50.

**Salida:** cada elemento de `moments` incluye:
- `start_seconds`, `end_seconds`, `duration_seconds`: resueltos por el backend.
- `score`, `rank`, `criteria` (los 8 criterios).
- `type`, `hook`, `reason`, `quote`.
- `source_segment_ids`: IDs de segmentos que componen el clip.
- `tiktok_title`, `tiktok_description`, `call_to_action`.
- Compatibilidad: `timestamp_seconds` (= `start_seconds + 10`) y `clip_duration` (= duración redondeada). Así un Parsear Momentos viejo sigue funcionando, aunque no aprovecha las duraciones dinámicas.

**Respuesta a nivel de respuesta:** `windows_analyzed`, `candidates_raw`, `discards_total`, `count`, `report_file`, `errors`, `job_id`.

**Archivos en el job:** `candidates.json` (todos los candidatos aceptados, descartados y con motivo) y `report.txt` (formato legible).

### `/clip` (cambio)
`start` y `duration` ahora aceptan decimales. El nombre del archivo sigue siendo `clip_<entero>s_<8 hex>.mp4`, y ffmpeg recibe la duración con dos decimales.

### Esquema de respuesta del modelo (GPT)
```json
{"candidates": [{
  "start_id": "S00042", "end_id": "S00047", "type": "historia",
  "hook": "...", "reason": "...", "quote": "frase literal entre start_id y end_id",
  "tiktok_title": "...", "tiktok_description": "...", "call_to_action": "...",
  "criteria": {"interest": 0-10, "clarity": 0-10, "emotion": 0-10, "originality": 0-10,
               "retention": 0-10, "resolution": 0-10, "start_quality": 0-10, "end_quality": 0-10}
}]}
```
GPT no calcula segundos. Si envía IDs inexistentes o una cita que no aparece en el rango, el candidato se descarta con motivo.

## 5. Prompts (M003.1)

Los prompts están en `worker/engine.py` (`SYSTEM_PROMPT_TEMPLATE` y `USER_PROMPT_TEMPLATE`). Cambios frente a la versión anterior:

- Se pide a GPT que **no calcule segundos**, sino IDs.
- Se piden **todos** los momentos que valgan la pena, sin cupo.
- Se exige **inicio, desarrollo y desenlace**, y se descarta lo repetitivo o incompleto.
- Se pide puntuar **8 criterios** con una escala explícita.
- Se prohíbe afirmar que la puntuación predice viralidad.

**Advertencia:** los prompts no se han evaluado todavía. Su efecto real se mide solo con la sección 3.

## 6. Cambios para n8n

Archivos nuevos en `workflows/` (no sobrescriben nada):

- `Kick Clips URL Manual v3 (m003).json`
- `Kick Auto Clips v6 v3 (m003).json`

**Único cambio funcional:** el nodo `Parsear Momentos` ahora usa `start_seconds` y `duration_seconds` cuando existen. Si no existen, usa el comportamiento anterior (`timestamp - 10` y 55 s). Se verificó ejecutándolo con Node en tres casos: formato nuevo, formato anterior y lista vacía. Los tres dan el resultado esperado.

`Crear Clip` no cambia: sigue leyendo `timestamp` y `clipDuration` de `Parsear Momentos`.

**Nota:** `timestamp` se redondea hacia abajo (`Math.floor`), así que el inicio puede quedar hasta 1 s antes del real. Eso se cubre con el margen de 1 s ya incluido en el backend.

**Transcribir Video Drive v2** no necesita cambios: `transcript_file` sigue existiendo.

**Referencias sin uso que quedaron:** en `Kick Auto Clips v6 v3` el nodo `Extraer URL del VOD` y en `Transcribir Video Drive v2` los nodos `Configuracion` y `Extraer ID Drive` todavía referencian `open_ai_key`. No contienen ningún valor real (en `Configuracion` es un marcador vacío). Pueden limpiarse en un paso aparte.

## 7. Costos de IA

- **Whisper:** sin cambios. Mismo número de chunks.
- **GPT:** una llamada por ventana en vez de por sección. Con ventanas de 15 min y 3 min de solapamiento, cada ventana avanza 12 min. Eso significa **~25% más llamadas** que antes para el mismo stream. `max_tokens` por llamada subió de 3000 a 4000 para acomodar más candidatos.
- **ffprobe:** una llamada extra por detección para leer la duración del VOD.

El costo real por stream no se conoce hasta correrlo. El costo de GPT es pequeño comparado con Whisper, pero conviene medirlo en la primera corrida real.

## 8. Limitaciones

- **Calidad no medida** (sección 3).
- **El modelo es el límite probable.** `gpt-4o-mini` es barato y rápido, pero puede no juzgar bien qué es un clip completo. Probar un modelo más fuerte es una decisión de costo.
- **Sin señal de audio** para los cortes. El helper `snap_end_to_silence` existe y tiene pruebas, pero no está conectado: decodificar silencios sobre el VOD completo tiene un costo que no se ha medido. Queda como siguiente paso.
- **Sin señal visual.** Un momento que depende de lo que se ve en pantalla no se detecta.
- **Pesos de puntuación** son una decisión editorial, no aprendida.
- **`max_moments` = 50** sigue siendo el valor que envía n8n. Si el stream tiene más momentos buenos, el ranking los corta. Conviene subirlo a 80 o 100 después de medir.
- **Clips de un solo segmento** se extienden a 15 s con segmentos contiguos. Si no hay contiguos, el clip queda más corto que 15 s (mínimo 1 s).
- **Sin cache de transcripción.** Correr de nuevo sobre el mismo VOD repite Whisper. Eso es parte de M002.1, no de esta misión.

## 9. Archivos

**Worker:**
- `worker/engine.py` — motor nuevo (módulo puro, sin red ni ffmpeg).
- `worker/server.py` — `/transcribe`, `/detect_moments` y `/clip` actualizados.
- `worker/Dockerfile` — ahora copia `engine.py` además de `server.py`. **Imprescindible**, o el contenedor no arranca.

**Pruebas:**
- `tests/test_engine.py` — 33 pruebas del motor.
- `tests/test_worker.py` — pruebas del worker actualizadas al nuevo contrato.

**Herramientas:**
- `tools/evaluate.py` — evaluación sobre etiquetas manuales.
- `tools/labels_template.json` — plantilla de etiquetas.
- `tools/convert_wf.py` — conversión de workflows (de M002).

**n8n:**
- `workflows/Kick Clips URL Manual v3 (m003).json`
- `workflows/Kick Auto Clips v6 v3 (m003).json`

## 10. Despliegue y rollback (para cuando se autorice)

**Despliegue:**
1. Copiar `worker/engine.py`, `worker/server.py` y `worker/Dockerfile` al repo.
2. Commit y push desde tu PC.
3. En el VPS: `git pull`, `docker compose build worker`, `docker compose up -d worker`.
4. Importar `Kick Clips URL Manual v3 (m003)` como workflow nuevo, con la credencial `Worker Token`.
5. Correr un stream de prueba con un VOD corto y guardar la respuesta de `/detect_moments`.
6. Medir con `tools/evaluate.py` y comparar con la versión anterior.

**Rollback:**
- El commit anterior es `1ff2090` (M002 desplegado). Volver a él con `git checkout 1ff2090 -- worker/ docker-compose.yml` y reconstruir el worker.
- Los workflows v2 siguen disponibles en el repo. Reactivar uno y desactivar el v3.
- Los jobs creados con M003 tienen `segments.json` y `candidates.json` extra. M002 los ignora, así que no hay conflicto.

## 11. Bloqueantes antes de desplegar

1. **Medición en un VOD real** con etiquetas manuales. Sin esto no se puede afirmar mejora.
2. **Dockerfile:** verificar que el build copie `engine.py` (ya está en la copia de trabajo).
3. **Decisión de costo:** aprobar ~25% más llamadas a GPT, y decidir si probar un modelo más fuerte.
4. **Decisión editorial:** confirmar los pesos de puntuación y el umbral de 5.
5. **Compatibilidad de n8n:** importar y probar el workflow v3 antes de activarlo.

Ningún bloqueante es técnico. Todos requieren una decisión tuya o una prueba con datos reales.
