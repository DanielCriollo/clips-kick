import json, re, sys, copy

CRED = {"httpHeaderAuth": {"id": "REEMPLAZAR_CREDENCIAL_WORKER_TOKEN", "name": "Worker Token"}}
SECRET_FIELDS = {'open_ai_key', 'api_key'}

def is_worker(node):
    url = node.get('parameters', {}).get('url', '')
    return isinstance(url, str) and url.lstrip('=').startswith('http://worker:8080')

def convert(src, dst, fix_transcript_path=False):
    wf = json.load(open(src, encoding='utf-8'))
    report = []
    for n in wf['nodes']:
        p = n.get('parameters', {})
        if n.get('type') != 'n8n-nodes-base.httpRequest' or not is_worker(n):
            continue
        p['authentication'] = 'genericCredentialType'
        p['genericAuthType'] = 'httpHeaderAuth'
        n['credentials'] = copy.deepcopy(CRED)
        bp = p.get('bodyParameters', {}).get('parameters')
        if bp:
            before = len(bp)
            p['bodyParameters']['parameters'] = [x for x in bp if x.get('name') not in SECRET_FIELDS]
            if before != len(p['bodyParameters']['parameters']):
                report.append(f"{n['name']}: quitado campo de clave")
        jb = p.get('jsonBody')
        if isinstance(jb, str):
            new = re.sub(r'(api_key|open_ai_key): \$json\.open_ai_key, ?', '', jb)
            if new != jb:
                p['jsonBody'] = new
                report.append(f"{n['name']}: quitada clave del jsonBody")
        if fix_transcript_path and n['name'] == 'Leer Transcripcion':
            p['url'] = '=http://worker:8080/serve_file?path={{ $json.transcript_file }}'
            report.append(f"{n['name']}: ruta cambiada a transcript_file del job")
        report.append(f"{n['name']}: auth httpHeaderAuth")
    wf['name'] = wf['name'] + ' v2 (auth)'
    json.dump(wf, open(dst, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    return report

if __name__ == '__main__':
    src, dst = sys.argv[1], sys.argv[2]
    fix = 'Transcribir' in src
    for line in convert(src, dst, fix_transcript_path=fix):
        print('  -', line)
