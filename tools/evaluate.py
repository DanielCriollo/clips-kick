"""Evaluacion de seleccion de clips sobre momentos etiquetados a mano.

Uso:
    python tools/evaluate.py --labels tools/labels_template.json --predictions respuesta_detect.json

--labels       JSON con los eventos buenos marcados por una persona (ver labels_template.json).
--predictions  Respuesta JSON de POST /detect_moments (o un JSON con la clave "moments").

Imprime y guarda metricas: cobertura, precision y error mediano de inicio/final.
No llama a ninguna API ni consume creditos.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'worker'))

import engine  # noqa: E402


def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def run(labels_path, predictions_path, match_ratio=0.5):
    labels_doc = load_json(labels_path)
    events = labels_doc.get('good_events', [])
    if not events:
        raise SystemExit('El archivo de etiquetas no tiene "good_events".')
    labels = [{'start': float(e['start']), 'end': float(e['end'])}
              for e in events if float(e['end']) > float(e['start'])]

    pred_doc = load_json(predictions_path)
    moments = pred_doc.get('moments', pred_doc if isinstance(pred_doc, list) else [])
    predictions = [{'start_seconds': float(m['start_seconds']), 'end_seconds': float(m['end_seconds'])}
                   for m in moments if 'start_seconds' in m and 'end_seconds' in m]

    metrics = engine.evaluate(predictions, labels, match_ratio=match_ratio)
    metrics['stream_id'] = labels_doc.get('stream_id', '')
    metrics['match_ratio'] = match_ratio
    metrics['note'] = ('Metricas sobre etiquetas manuales. Con pocas etiquetas, '
                       'el resultado tiene mucha varianza.')
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--match-ratio', type=float, default=0.5)
    parser.add_argument('--out', default=None, help='ruta para guardar el resultado en JSON')
    args = parser.parse_args()

    metrics = run(args.labels, args.predictions, args.match_ratio)
    text = json.dumps(metrics, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            f.write(text + '\n')


if __name__ == '__main__':
    main()
